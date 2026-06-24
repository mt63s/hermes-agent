"""Honcho Slice-2 reconciler.

This module repairs best-effort Honcho ingestion gaps by comparing the durable
Hermes ``state.db`` transcript with Honcho messages tagged by Slice 1 metadata.
It is intentionally standalone and off the gateway hot path: dry-run by default,
read-only against SQLite, and writes to Honcho only when ``apply=True``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence, cast

from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from plugins.memory.honcho import (
    HONCHO_METADATA_EPOCH_FILENAME,
    HONCHO_METADATA_VERSION,
    HONCHO_UNTAGGED_EXCLUSIONS_FILENAME,
    HonchoMemoryProvider,
    build_honcho_ingest_metadata,
    honcho_ingest_clean_hash,
    honcho_ingest_clean_text,
)
from plugins.memory.honcho.session import HonchoSessionManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StateMessage:
    """One eligible durable state.db message row."""

    id: int
    session_id: str
    role: str
    content: Any
    clean_content: str
    message_hash: str
    timestamp: float | None


@dataclass(frozen=True)
class StateSessionMetadata:
    """StateDB metadata needed to reproduce Honcho session-name resolution."""

    session_id: str
    cwd: str | None = None
    title: str | None = None


@dataclass(frozen=True)
class DriftMismatch:
    """A Honcho message exists for an id but with a different content hash."""

    session_id: str
    honcho_session_key: str
    hermes_message_id: int
    expected_hash: str
    present_hash: str


@dataclass(frozen=True)
class ReconcileFailure:
    """A non-fatal per-message/per-session reconcile failure."""

    session_id: str
    honcho_session_key: str
    hermes_message_id: int | None
    error: str


@dataclass
class ReconcileReport:
    """Summary of one reconciler run."""

    apply: bool
    epoch: int
    scanned: int = 0
    eligible: int = 0
    already_present: int = 0
    excluded: int = 0
    gaps: int = 0
    uploaded: int = 0
    failed: int = 0
    drift_mismatches: list[DriftMismatch] = field(default_factory=list)
    malformed_exclusions: list[str] = field(default_factory=list)
    failures: list[ReconcileFailure] = field(default_factory=list)
    per_session: list[dict[str, Any]] = field(default_factory=list)
    derive_queue_load: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReconcileOptions:
    """Runtime options for :func:`reconcile_honcho`."""

    hermes_home: Path | str | None = None
    state_db_path: Path | str | None = None
    apply: bool = False
    batch_size: int = 50
    pause_seconds: float = 0.0
    message_max_chars: int | None = None
    session_ids: tuple[str, ...] = ()
    honcho_session_key: str | None = None
    limit: int | None = None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_home(path: Path | str | None) -> Path:
    return Path(path) if path is not None else get_hermes_home()


def _normalize_state_db_path(home: Path, path: Path | str | None) -> Path:
    return Path(path) if path is not None else home / "state.db"


def load_metadata_epoch(hermes_home: Path | str | None = None) -> int:
    """Return the Slice-1 cutover id from ``honcho_metadata_epoch.json``.

    Missing/unreadable markers are hard errors: without the epoch, the
    reconciler cannot honestly distinguish guaranteed-replayable tagged-era rows
    from older best-effort untagged history.
    """
    home = _normalize_home(hermes_home)
    path = home / HONCHO_METADATA_EPOCH_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Honcho metadata epoch marker not found: {path}. "
            "Run Slice-1 live ingestion first; the reconciler will not guess."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exact JSON error is unimportant
        raise ValueError(f"Could not read Honcho metadata epoch marker {path}: {exc}") from exc
    epoch = _coerce_int(data.get("first_tagged_message_id") or data.get("hermes_ingest_epoch"))
    if epoch is None or epoch <= 0:
        raise ValueError(f"Honcho metadata epoch marker {path} does not contain a positive epoch")
    return epoch


def load_exclusion_ids(hermes_home: Path | str | None = None) -> tuple[set[int], list[str]]:
    """Load fail-open untagged post-epoch message ids.

    Missing file means the live path never recorded a successful untagged
    fallback, so the exclusion set is empty. Malformed lines are warnings, not
    fatal, because the reconciler can still safely process the valid records.
    """
    home = _normalize_home(hermes_home)
    path = home / HONCHO_UNTAGGED_EXCLUSIONS_FILENAME
    if not path.exists():
        return set(), []

    ids: set[int] = set()
    warnings: list[str] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            warnings.append(f"{path}:{line_no}: malformed JSON: {exc.msg}")
            continue
        message_id = _coerce_int(record.get("hermes_message_id")) if isinstance(record, dict) else None
        if message_id is None:
            warnings.append(f"{path}:{line_no}: missing/invalid hermes_message_id")
            continue
        ids.add(message_id)
    return ids, warnings


def _state_db_uri(path: Path) -> str:
    return f"file:{path}?mode=ro"


def select_eligible_state_messages(
    state_db_path: Path | str,
    *,
    epoch: int,
    session_ids: Sequence[str] = (),
    limit: int | None = None,
) -> tuple[list[StateMessage], int]:
    """Read eligible post-epoch state.db messages without taking write locks.

    Eligibility mirrors the live Honcho ingestion contract: active,
    uncompacted user/assistant rows at or above the metadata epoch, excluding
    assistant tool-call scaffolding and messages that sanitize to empty text.
    """
    db_path = Path(state_db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"state.db not found: {db_path}")

    where = ["id >= ?"]
    params: list[Any] = [int(epoch)]
    if session_ids:
        placeholders = ",".join("?" for _ in session_ids)
        where.append(f"session_id IN ({placeholders})")
        params.extend(session_ids)
    sql = (
        "SELECT id, session_id, role, content, tool_calls, timestamp, active, compacted "
        "FROM messages WHERE "
        + " AND ".join(where)
        + " ORDER BY session_id, id"
    )
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))

    conn = sqlite3.connect(_state_db_uri(db_path), uri=True, timeout=1.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        raw_rows = list(conn.execute(sql, params))
    finally:
        conn.close()

    rows: list[StateMessage] = []
    for row in raw_rows:
        if int(row["active"] or 0) != 1 or int(row["compacted"] or 0) != 0:
            continue
        role = row["role"]
        if role not in {"user", "assistant"}:
            continue
        if role == "assistant" and row["tool_calls"]:
            continue
        content = SessionDB._decode_content(row["content"])
        source_text = HonchoMemoryProvider._message_content_text({"content": content})
        clean = honcho_ingest_clean_text(source_text)
        if not clean:
            continue
        rows.append(
            StateMessage(
                id=int(row["id"]),
                session_id=str(row["session_id"]),
                role=str(role),
                content=content,
                clean_content=clean,
                message_hash=honcho_ingest_clean_hash(clean),
                timestamp=float(row["timestamp"]) if row["timestamp"] is not None else None,
            )
        )
    return rows, len(raw_rows)


def load_state_session_metadata(
    state_db_path: Path | str,
    session_ids: Sequence[str],
) -> dict[str, StateSessionMetadata]:
    """Load session title/cwd fields used by Honcho session resolution."""
    if not session_ids:
        return {}
    db_path = Path(state_db_path)
    placeholders = ",".join("?" for _ in session_ids)
    sql = f"SELECT id, cwd, title FROM sessions WHERE id IN ({placeholders})"
    conn = sqlite3.connect(_state_db_uri(db_path), uri=True, timeout=1.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        rows = list(conn.execute(sql, list(session_ids)))
    finally:
        conn.close()
    return {
        str(row["id"]): StateSessionMetadata(
            session_id=str(row["id"]),
            cwd=str(row["cwd"]) if row["cwd"] else None,
            title=str(row["title"]) if row["title"] else None,
        )
        for row in rows
    }


def resolve_honcho_session_key(
    config: Any,
    state_session_id: str,
    metadata: StateSessionMetadata | None = None,
) -> str:
    """Resolve the Honcho session key using the same config hook as live init."""
    resolver = getattr(config, "resolve_session_name", None) if config is not None else None
    if not callable(resolver):
        return state_session_id
    metadata = metadata or StateSessionMetadata(session_id=state_session_id)
    try:
        resolved = resolver(
            cwd=metadata.cwd,
            session_title=metadata.title,
            session_id=state_session_id,
        )
    except Exception:
        logger.debug("Honcho session-name resolver failed for %s", state_session_id, exc_info=True)
        return state_session_id
    return str(resolved or state_session_id)


def _metadata_from_message(message: Any) -> dict[str, Any]:
    metadata = message.get("metadata") if isinstance(message, dict) else getattr(message, "metadata", None)
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
    return metadata if isinstance(metadata, dict) else {}


def _list_existing_honcho_messages(remote_session: Any) -> list[Any]:
    messages_attr = getattr(remote_session, "messages", None)
    if callable(messages_attr):
        result = messages_attr()
    else:
        result = messages_attr
    if result is None:
        return []
    if isinstance(result, dict):
        items = result.get("items")
        if isinstance(items, list):
            return list(items)
    try:
        iterator = iter(cast(Iterable[Any], result))
    except TypeError:
        return []
    return [item for item in iterator]


def _present_honcho_ids(existing_messages: Iterable[Any]) -> tuple[set[int], dict[int, str]]:
    present: set[int] = set()
    hashes: dict[int, str] = {}
    for message in existing_messages:
        metadata = _metadata_from_message(message)
        message_id = _coerce_int(metadata.get("hermes_message_id"))
        if message_id is None:
            continue
        present.add(message_id)
        message_hash = metadata.get("hermes_message_hash")
        if isinstance(message_hash, str) and message_hash and message_id not in hashes:
            hashes[message_id] = message_hash
    return present, hashes


def _resolve_manager() -> HonchoSessionManager:
    from plugins.memory.honcho.client import HonchoClientConfig, get_honcho_client

    cfg = HonchoClientConfig.from_global_config()
    if not cfg.enabled or not (cfg.api_key or cfg.base_url):
        raise RuntimeError("Honcho is not configured/enabled for this profile")
    client = get_honcho_client(cfg)
    return HonchoSessionManager(honcho=client, config=cfg, context_tokens=cfg.context_tokens)


def _honcho_session_id(manager: Any, honcho_key: str) -> str:
    sanitizer = getattr(manager, "_sanitize_id", None)
    if callable(sanitizer):
        return str(sanitizer(honcho_key))
    return re.sub(r"[^a-zA-Z0-9_-]", "-", str(honcho_key))


def _resolve_remote_session_read_only(manager: Any, honcho_key: str) -> Any:
    """Return the remote Honcho session without get_or_create/add_peers.

    Dry-run must not mutate Honcho session peer configuration. The live manager's
    get_or_create() path intentionally calls add_peers(); reserve it for apply.
    """
    honcho_session_id = _honcho_session_id(manager, honcho_key)
    cache = getattr(manager, "_sessions_cache", {})
    if isinstance(cache, dict) and honcho_session_id in cache:
        return cache[honcho_session_id]
    honcho = getattr(manager, "honcho", None)
    session_factory = getattr(honcho, "session", None)
    if callable(session_factory):
        return session_factory(honcho_session_id)
    raise RuntimeError("Honcho manager does not expose a read-only session lookup")


def _resolve_remote_session(manager: Any, honcho_key: str) -> tuple[Any, Any, Any, Any]:
    local_session = manager.get_or_create(honcho_key)
    user_peer = manager._get_or_create_peer(local_session.user_peer_id)
    assistant_peer = manager._get_or_create_peer(local_session.assistant_peer_id)
    remote_session = getattr(manager, "_sessions_cache", {}).get(local_session.honcho_session_id)
    if remote_session is None:
        remote_session, _ = manager._get_or_create_honcho_session(
            local_session.honcho_session_id,
            user_peer,
            assistant_peer,
        )
    return local_session, remote_session, user_peer, assistant_peer


def _build_honcho_messages_for_state_message(
    state_message: StateMessage,
    *,
    user_peer: Any,
    assistant_peer: Any,
    epoch: int,
    profile: str,
    msg_limit: int,
) -> list[Any]:
    peer = user_peer if state_message.role == "user" else assistant_peer
    chunks = HonchoMemoryProvider._chunk_message(state_message.clean_content, msg_limit)
    created_at = HonchoSessionManager._coerce_created_at(state_message.timestamp)
    honcho_messages: list[Any] = []
    for idx, chunk in enumerate(chunks):
        metadata = build_honcho_ingest_metadata(
            ingest_path="reconcile",
            profile=profile,
            session_id=state_message.session_id,
            message_id=state_message.id,
            role=state_message.role,
            clean_content=state_message.clean_content,
            chunk=chunk,
            chunk_index=idx,
            chunk_count=len(chunks),
            ingest_epoch=epoch,
        )
        kwargs: dict[str, Any] = {"metadata": metadata}
        if created_at is not None:
            kwargs["created_at"] = created_at
        honcho_messages.append(peer.message(chunk, **kwargs))
    return honcho_messages


def _chunk_source_groups(
    groups: list[tuple[StateMessage, list[Any]]],
    batch_size: int,
) -> Iterable[list[tuple[StateMessage, list[Any]]]]:
    batch: list[tuple[StateMessage, list[Any]]] = []
    batch_message_count = 0
    for group in groups:
        group_count = len(group[1])
        if batch and batch_message_count + group_count > batch_size:
            yield batch
            batch = []
            batch_message_count = 0
        batch.append(group)
        batch_message_count += group_count
    if batch:
        yield batch


def reconcile_honcho(
    options: ReconcileOptions | None = None,
    *,
    manager: Any | None = None,
) -> ReconcileReport:
    """Run one Honcho reconciliation pass.

    The returned report is message-id based: ``gaps`` and ``uploaded`` count
    durable state.db rows, not Honcho chunk messages. ``apply=False`` still
    contacts Honcho to list existing metadata, but never calls ``add_messages``.
    """
    options = options or ReconcileOptions()
    home = _normalize_home(options.hermes_home)
    db_path = _normalize_state_db_path(home, options.state_db_path)
    epoch = load_metadata_epoch(home)
    exclusion_ids, malformed = load_exclusion_ids(home)
    state_messages, scanned = select_eligible_state_messages(
        db_path,
        epoch=epoch,
        session_ids=options.session_ids,
        limit=options.limit,
    )

    if options.honcho_session_key and not options.session_ids:
        distinct_sessions = {m.session_id for m in state_messages}
        if len(distinct_sessions) > 1:
            raise ValueError(
                "--honcho-session-key with multiple state sessions is ambiguous; "
                "also pass --session-id"
            )

    if manager is None:
        manager = _resolve_manager()

    config = getattr(manager, "_config", None)
    msg_limit = int(options.message_max_chars or getattr(config, "message_max_chars", 25000) or 25000)
    batch_size = max(1, min(int(options.batch_size), 100))
    profile = HonchoMemoryProvider._active_profile_name()

    report = ReconcileReport(
        apply=options.apply,
        epoch=epoch,
        scanned=scanned,
        eligible=len(state_messages),
        malformed_exclusions=malformed,
    )

    by_session: dict[str, list[StateMessage]] = defaultdict(list)
    for message in state_messages:
        by_session[message.session_id].append(message)
    session_metadata = load_state_session_metadata(db_path, tuple(by_session.keys()))

    for state_session_id, messages in sorted(by_session.items()):
        honcho_key = options.honcho_session_key or resolve_honcho_session_key(
            config,
            state_session_id,
            session_metadata.get(state_session_id),
        )
        session_summary: dict[str, Any] = {
            "session_id": state_session_id,
            "honcho_session_key": honcho_key,
            "eligible": len(messages),
            "already_present": 0,
            "excluded": 0,
            "gaps": 0,
            "uploaded": 0,
            "failed": 0,
            "drift_mismatches": 0,
        }
        try:
            remote_session = _resolve_remote_session_read_only(manager, honcho_key)
            present_ids, present_hash_by_id = _present_honcho_ids(
                _list_existing_honcho_messages(remote_session)
            )
        except Exception as exc:
            report.failed += len(messages)
            session_summary["failed"] += len(messages)
            report.failures.append(
                ReconcileFailure(state_session_id, honcho_key, None, f"session/list failed: {exc}")
            )
            report.per_session.append(session_summary)
            continue

        gap_messages: list[StateMessage] = []
        for message in messages:
            present = message.id in present_ids
            if present:
                report.already_present += 1
                session_summary["already_present"] += 1
                present_hash = present_hash_by_id.get(message.id)
                if present_hash and present_hash != message.message_hash:
                    mismatch = DriftMismatch(
                        session_id=state_session_id,
                        honcho_session_key=honcho_key,
                        hermes_message_id=message.id,
                        expected_hash=message.message_hash,
                        present_hash=present_hash,
                    )
                    report.drift_mismatches.append(mismatch)
                    session_summary["drift_mismatches"] += 1
                continue
            if message.id in exclusion_ids:
                report.excluded += 1
                session_summary["excluded"] += 1
                continue

            report.gaps += 1
            session_summary["gaps"] += 1
            gap_messages.append(message)

        if options.apply and gap_messages:
            try:
                _local_session, write_remote_session, user_peer, assistant_peer = _resolve_remote_session(
                    manager, honcho_key
                )
            except Exception as exc:
                report.failed += len(gap_messages)
                session_summary["failed"] += len(gap_messages)
                for state_message in gap_messages:
                    report.failures.append(
                        ReconcileFailure(
                            state_session_id,
                            honcho_key,
                            state_message.id,
                            f"session/write setup failed: {exc}",
                        )
                    )
                report.per_session.append(session_summary)
                continue

            payload_groups: list[tuple[StateMessage, list[Any]]] = []
            for message in gap_messages:
                try:
                    payload_groups.append(
                        (
                            message,
                            _build_honcho_messages_for_state_message(
                                message,
                                user_peer=user_peer,
                                assistant_peer=assistant_peer,
                                epoch=epoch,
                                profile=profile,
                                msg_limit=msg_limit,
                            ),
                        )
                    )
                except Exception as exc:
                    report.failed += 1
                    session_summary["failed"] += 1
                    report.failures.append(
                        ReconcileFailure(
                            state_session_id,
                            honcho_key,
                            message.id,
                            f"message build failed: {exc}",
                        )
                    )

            for batch_groups in _chunk_source_groups(payload_groups, batch_size):
                batch_messages = [honcho_msg for _state, msgs in batch_groups for honcho_msg in msgs]
                try:
                    write_remote_session.add_messages(batch_messages)
                except Exception as exc:
                    failed_count = len(batch_groups)
                    report.failed += failed_count
                    session_summary["failed"] += failed_count
                    for state_message, _msgs in batch_groups:
                        report.failures.append(
                            ReconcileFailure(
                                state_session_id,
                                honcho_key,
                                state_message.id,
                                f"add_messages failed: {exc}",
                            )
                        )
                    continue
                uploaded_count = len(batch_groups)
                report.uploaded += uploaded_count
                session_summary["uploaded"] += uploaded_count
                if options.pause_seconds > 0:
                    time.sleep(options.pause_seconds)

        report.per_session.append(session_summary)

    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconcile Hermes state.db messages into Honcho")
    parser.add_argument("--apply", action="store_true", help="write missing messages to Honcho (default: dry-run)")
    parser.add_argument("--hermes-home", type=Path, default=None, help="Hermes profile home (default: $HERMES_HOME)")
    parser.add_argument("--state-db", type=Path, default=None, help="state.db path (default: $HERMES_HOME/state.db)")
    parser.add_argument("--session-id", action="append", default=[], help="limit to one state.db session_id; repeatable")
    parser.add_argument("--honcho-session-key", default=None, help="override Honcho session key for a single selected state session")
    parser.add_argument("--batch-size", type=int, default=50, help="max Honcho messages per add_messages call (1-100)")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds to pause between write batches")
    parser.add_argument("--message-max-chars", type=int, default=None, help="override Honcho message chunk size")
    parser.add_argument("--limit", type=int, default=None, help="limit raw state.db rows scanned")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = reconcile_honcho(
        ReconcileOptions(
            hermes_home=args.hermes_home,
            state_db_path=args.state_db,
            apply=args.apply,
            batch_size=args.batch_size,
            pause_seconds=args.pause,
            message_max_chars=args.message_max_chars,
            session_ids=tuple(args.session_id or ()),
            honcho_session_key=args.honcho_session_key,
            limit=args.limit,
        )
    )
    data = report.to_dict()
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True, default=str))
    else:
        mode = "APPLY" if report.apply else "DRY-RUN"
        print(f"Honcho reconcile {mode}: epoch={report.epoch}")
        print(
            "scanned={scanned} eligible={eligible} already_present={already_present} "
            "excluded={excluded} gaps={gaps} uploaded={uploaded} failed={failed} "
            "drift_mismatches={drift}".format(
                scanned=report.scanned,
                eligible=report.eligible,
                already_present=report.already_present,
                excluded=report.excluded,
                gaps=report.gaps,
                uploaded=report.uploaded,
                failed=report.failed,
                drift=len(report.drift_mismatches),
            )
        )
        for warning in report.malformed_exclusions:
            print(f"warning: {warning}")
        for failure in report.failures:
            print(f"failure: {failure.session_id} {failure.hermes_message_id}: {failure.error}")
    return 0 if report.failed == 0 else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
