"""Slice-2 Honcho reconciler tests.

The reconciler repairs Honcho best-effort write gaps by diffing durable
state.db rows against Honcho message metadata.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from hermes_state import SessionDB
from plugins.memory.honcho import honcho_ingest_clean_hash, honcho_ingest_clean_text
from plugins.memory.honcho.reconcile import (
    ReconcileOptions,
    audit_identity_gaps,
    load_exclusion_ids,
    reconcile_honcho,
    select_eligible_state_messages,
)
from plugins.memory.honcho.session import HonchoSession


class FakeHonchoMessage:
    def __init__(self, peer_id: str, content: str, *, metadata=None, created_at=None):
        self.peer_id = peer_id
        self.content = content
        self.metadata = metadata or {}
        self.created_at = created_at


class FakePeer:
    def __init__(self, peer_id: str):
        self.id = peer_id

    def message(self, content: str, **kwargs):
        return FakeHonchoMessage(
            self.id,
            content,
            metadata=kwargs.get("metadata"),
            created_at=kwargs.get("created_at"),
        )


class FakeRemoteSession:
    def __init__(self, existing=None, session_id=""):
        self.id = session_id
        self.existing = list(existing or [])
        self.add_batches = []

    def messages(self):
        return list(self.existing)

    def add_messages(self, messages):
        batch = list(messages)
        self.add_batches.append(batch)
        self.existing.extend(batch)


class FakeHonchoClient:
    def __init__(self, manager):
        self.manager = manager

    def session(self, session_id: str):
        self.manager.session_lookup_calls += 1
        self.manager._sessions_cache.setdefault(session_id, FakeRemoteSession(session_id=session_id))
        return self.manager._sessions_cache[session_id]

    def sessions(self):
        return list(self.manager._sessions_cache.values())


class FakeHonchoManager:
    def __init__(self, *, existing_by_key=None, message_max_chars=25000, config=None):
        self._config = config or SimpleNamespace(message_max_chars=message_max_chars)
        self._cache = {}
        self._peers = {}
        self._sessions_cache = {}
        self.get_or_create_calls = 0
        self.get_or_create_keys = []
        self.session_lookup_calls = 0
        self.honcho = FakeHonchoClient(self)
        for key, messages in (existing_by_key or {}).items():
            sanitized = self._sanitize(key)
            self._sessions_cache[sanitized] = FakeRemoteSession(messages, session_id=sanitized)

    @staticmethod
    def _sanitize(value: str) -> str:
        return str(value).replace(":", "-")

    def _get_or_create_peer(self, peer_id):
        self._peers.setdefault(peer_id, FakePeer(peer_id))
        return self._peers[peer_id]

    def _get_or_create_honcho_session(self, honcho_session_id, user_peer, assistant_peer):
        self._sessions_cache.setdefault(honcho_session_id, FakeRemoteSession(session_id=honcho_session_id))
        return self._sessions_cache[honcho_session_id], []

    def get_or_create(self, key):
        self.get_or_create_calls += 1
        self.get_or_create_keys.append(key)
        if key not in self._cache:
            session = HonchoSession(
                key=key,
                user_peer_id=f"user-{self._sanitize(key)}",
                assistant_peer_id="hermes-assistant",
                honcho_session_id=self._sanitize(key),
            )
            self._cache[key] = session
            self._sessions_cache.setdefault(session.honcho_session_id, FakeRemoteSession())
        return self._cache[key]

    def remote(self, key):
        return self._sessions_cache[self._sanitize(key)]


def _write_epoch(home: Path, first_id: int = 1):
    (home / "honcho_metadata_epoch.json").write_text(
        json.dumps({"first_tagged_message_id": first_id, "hermes_ingest_epoch": first_id}),
        encoding="utf-8",
    )


def _make_db(tmp_path: Path, monkeypatch) -> tuple[SessionDB, Path]:
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    db_path = home / "state.db"
    return SessionDB(db_path=db_path), db_path


def test_select_eligible_state_messages_matches_live_ingestion_contract(tmp_path, monkeypatch):
    db, db_path = _make_db(tmp_path, monkeypatch)
    try:
        session_id = db.create_session("session-1", source="cli")
        pre_epoch = db.append_message(session_id, "user", "too old")
        keep_user = db.append_message(session_id, "user", "  hello\n")
        keep_multi = db.append_message(
            session_id,
            "user",
            cast(Any, [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": "x"}]),
        )
        keep_assistant = db.append_message(session_id, "assistant", "answer")
        assistant_tool_call = db.append_message(
            session_id,
            "assistant",
            "tool scaffold",
            tool_calls=[{"id": "c1", "type": "function"}],
        )
        tool_row = db.append_message(session_id, "tool", "tool result")
        inactive = db.append_message(session_id, "user", "inactive")
        compacted = db.append_message(session_id, "assistant", "compacted")
        blank = db.append_message(session_id, "user", "   ")
        synthetic_task = db.append_message(
            session_id,
            "user",
            "[Your active task list was preserved across context compression]\n- [ ] internal task",
        )
        synthetic_compaction = db.append_message(
            session_id,
            "assistant",
            "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted.",
        )
        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE messages SET active = 0 WHERE id = ?", (inactive,))
            conn.execute("UPDATE messages SET compacted = 1 WHERE id = ?", (compacted,))
    finally:
        db.close()

    rows, scanned = select_eligible_state_messages(db_path, epoch=keep_user)

    assert scanned >= 8
    assert [row.id for row in rows] == [keep_user, keep_multi, keep_assistant]
    assert pre_epoch not in {row.id for row in rows}
    assert assistant_tool_call not in {row.id for row in rows}
    assert tool_row not in {row.id for row in rows}
    assert blank not in {row.id for row in rows}
    assert synthetic_task not in {row.id for row in rows}
    assert synthetic_compaction not in {row.id for row in rows}
    assert rows[0].clean_content == "hello"
    assert rows[1].clean_content == "look\n[screenshot]"
    assert rows[0].message_hash == honcho_ingest_clean_hash("hello")


def test_load_exclusion_ids_missing_file_is_empty_and_malformed_lines_warn(tmp_path):
    missing_ids, missing_warnings = load_exclusion_ids(tmp_path)
    assert missing_ids == set()
    assert missing_warnings == []

    exclusions = tmp_path / "honcho_untagged_ingest_exclusions.jsonl"
    exclusions.write_text(
        "\n".join(
            [
                json.dumps({"hermes_message_id": 101}),
                "not json",
                json.dumps({"hermes_message_id": "102"}),
                json.dumps({"wrong": 103}),
                json.dumps({"hermes_message_id": 101}),
            ]
        ),
        encoding="utf-8",
    )

    ids, warnings = load_exclusion_ids(tmp_path)

    assert ids == {101, 102}
    assert len(warnings) == 2


def test_reconcile_dry_run_dedups_exclusions_and_reports_drift(tmp_path, monkeypatch):
    db, db_path = _make_db(tmp_path, monkeypatch)
    home = db_path.parent
    try:
        session_id = db.create_session("session-1", source="cli")
        first = db.append_message(session_id, "user", "already present", timestamp=1700000000.25)
        drift = db.append_message(session_id, "assistant", "changed local", timestamp=1700000001.5)
        excluded = db.append_message(session_id, "user", "untagged live fallback")
        gap = db.append_message(session_id, "assistant", "missing")
    finally:
        db.close()
    _write_epoch(home, first)
    (home / "honcho_untagged_ingest_exclusions.jsonl").write_text(
        json.dumps({"hermes_message_id": excluded}) + "\n", encoding="utf-8"
    )
    existing = [
        FakeHonchoMessage(
            "user-session-1",
            "already present",
            metadata={
                "hermes_message_id": first,
                "hermes_message_hash": honcho_ingest_clean_hash("already present"),
            },
        ),
        FakeHonchoMessage(
            "hermes-assistant",
            "old content",
            metadata={"hermes_message_id": drift, "hermes_message_hash": "wrong"},
        ),
    ]
    manager = FakeHonchoManager(existing_by_key={session_id: existing})

    report = reconcile_honcho(
        ReconcileOptions(hermes_home=home, state_db_path=db_path, apply=False),
        manager=manager,
    )

    assert report.scanned == 4
    assert report.eligible == 4
    assert report.already_present == 2
    assert report.excluded == 1
    assert report.gaps == 1
    assert report.uploaded == 0
    assert [m.hermes_message_id for m in report.drift_mismatches] == [drift]
    assert manager.get_or_create_calls == 0
    assert manager.remote(session_id).add_batches == []


def test_reconcile_skips_sessions_live_path_would_not_sync(tmp_path, monkeypatch):
    db, db_path = _make_db(tmp_path, monkeypatch)
    home = db_path.parent
    try:
        synced = db.create_session("synced-session", source="telegram")
        worker = db.create_session("worker-session", source="subagent")
        synced_msg = db.append_message(synced, "user", "human-facing gap")
        worker_msg = db.append_message(worker, "user", "worker should stay lean")
    finally:
        db.close()
    _write_epoch(home, synced_msg)
    manager = FakeHonchoManager(message_max_chars=25000)

    report = reconcile_honcho(
        ReconcileOptions(hermes_home=home, state_db_path=db_path, apply=False),
        manager=manager,
    )

    assert report.scanned == 2
    assert report.eligible == 1
    assert report.skipped_non_sync == 1
    assert report.gaps == 1
    assert report.per_session[0]["session_id"] == "synced-session"
    skipped = [s for s in report.per_session if s["session_id"] == "worker-session"][0]
    assert skipped["skipped_non_sync"] is True
    assert skipped["source"] == "subagent"
    assert worker_msg not in {failure.hermes_message_id for failure in report.failures}

    audit = audit_identity_gaps(
        ReconcileOptions(hermes_home=home, state_db_path=db_path, apply=False),
        manager=manager,
    )
    assert audit.eligible == 1
    assert audit.skipped_non_sync == 1
    assert audit.apparent_gaps == 1
    assert audit.mass_backfill_candidates == 1


def test_reconcile_uses_config_session_resolution_from_state_metadata(tmp_path, monkeypatch):
    db, db_path = _make_db(tmp_path, monkeypatch)
    home = db_path.parent
    try:
        session_id = db.create_session("session-with-title", source="telegram")
        first = db.append_message(session_id, "user", "present through title")
        db.set_session_title(session_id, "Resolved Title")
    finally:
        db.close()
    _write_epoch(home, first)

    class ResolvingConfig(SimpleNamespace):
        def resolve_session_name(self, *, cwd=None, session_title=None, session_id=None, gateway_session_key=None):
            assert session_title == "Resolved Title"
            return str(session_title).replace(" ", "-")

    manager = FakeHonchoManager(
        config=ResolvingConfig(message_max_chars=25000),
        existing_by_key={
            "Resolved-Title": [
                FakeHonchoMessage(
                    "user-Resolved-Title",
                    "present through title",
                    metadata={
                        "hermes_message_id": first,
                        "hermes_message_hash": honcho_ingest_clean_hash("present through title"),
                    },
                )
            ]
        },
    )

    report = reconcile_honcho(
        ReconcileOptions(hermes_home=home, state_db_path=db_path, apply=False),
        manager=manager,
    )

    assert report.already_present == 1
    assert report.gaps == 0
    assert report.per_session[0]["honcho_session_key"] == "Resolved-Title"



def test_reconcile_apply_uses_same_resolved_key_as_dry_run(tmp_path, monkeypatch):
    db, db_path = _make_db(tmp_path, monkeypatch)
    home = db_path.parent
    try:
        session_id = db.create_session("state-session", source="telegram")
        first = db.append_message(session_id, "user", "resolved gap")
        db.set_session_title(session_id, "Resolved Title")
    finally:
        db.close()
    _write_epoch(home, first)

    class ResolvingConfig(SimpleNamespace):
        def resolve_session_name(self, *, cwd=None, session_title=None, session_id=None, gateway_session_key=None):
            return str(session_title).replace(" ", "-")

    manager = FakeHonchoManager(config=ResolvingConfig(message_max_chars=25000))

    dry_report = reconcile_honcho(
        ReconcileOptions(hermes_home=home, state_db_path=db_path, apply=False),
        manager=manager,
    )
    assert dry_report.gaps == 1
    assert dry_report.per_session[0]["honcho_session_key"] == "Resolved-Title"
    assert manager.get_or_create_calls == 0

    apply_report = reconcile_honcho(
        ReconcileOptions(
            hermes_home=home,
            state_db_path=db_path,
            apply=True,
            session_ids=(session_id,),
        ),
        manager=manager,
    )

    assert apply_report.uploaded == 1
    assert apply_report.per_session[0]["honcho_session_key"] == "Resolved-Title"
    assert manager.get_or_create_keys == ["Resolved-Title"]
    assert manager.remote("Resolved-Title").add_batches
    assert "state-session" not in manager._sessions_cache


def test_identity_audit_partitions_key_drift_from_missing_everywhere(tmp_path, monkeypatch):
    db, db_path = _make_db(tmp_path, monkeypatch)
    home = db_path.parent
    try:
        drifted_session = db.create_session("state-session", source="telegram")
        drifted = db.append_message(drifted_session, "user", "present elsewhere")
        drifted_missing = db.append_message(drifted_session, "assistant", "missing but session drifted")
        db.set_session_title(drifted_session, "New Title")
        clean_session = db.create_session("clean-session", source="telegram")
        clean_missing = db.append_message(clean_session, "assistant", "clean missing everywhere")
        db.set_session_title(clean_session, "Clean Title")
        duplicate_session = db.create_session("duplicate-session", source="telegram")
        duplicate_missing = db.append_message(duplicate_session, "user", "same content already tagged elsewhere")
        db.set_session_title(duplicate_session, "Duplicate Title")
        safe_session = db.create_session("safe-session", source="telegram")
        safe_missing = db.append_message(safe_session, "assistant", "safe real gap")
        db.set_session_title(safe_session, "Safe Title")
    finally:
        db.close()
    _write_epoch(home, drifted)

    class ResolvingConfig(SimpleNamespace):
        def resolve_session_name(self, *, cwd=None, session_title=None, session_id=None, gateway_session_key=None):
            return str(session_title).replace(" ", "-")

    old_key = "Old-Title"
    manager = FakeHonchoManager(
        config=ResolvingConfig(message_max_chars=25000),
        existing_by_key={
            old_key: [
                FakeHonchoMessage(
                    "user-Old-Title",
                    "present elsewhere",
                    metadata={
                        "hermes_message_id": drifted,
                        "hermes_session_id": drifted_session,
                        "hermes_message_hash": honcho_ingest_clean_hash("present elsewhere"),
                        "hermes_ingest_path": "live",
                        "hermes_chunk_index": 0,
                        "hermes_chunk_count": 1,
                    },
                )
            ],
            "Clean-Title": [
                FakeHonchoMessage("hermes-assistant", "clean missing everywhere")
            ],
            "Other-Title": [
                FakeHonchoMessage(
                    "user-Other-Title",
                    "same content already tagged elsewhere",
                    metadata={
                        "hermes_message_id": 999001,
                        "hermes_session_id": "other-state-session",
                        "hermes_message_hash": honcho_ingest_clean_hash("same content already tagged elsewhere"),
                        "hermes_ingest_path": "live",
                    },
                )
            ],
        },
    )

    audit = audit_identity_gaps(
        ReconcileOptions(hermes_home=home, state_db_path=db_path, apply=False),
        manager=manager,
    )

    assert audit.apparent_gaps == 5
    assert audit.elsewhere_present == 1
    assert audit.missing_everywhere == 4
    assert audit.same_content_present == 2
    assert audit.untagged_present == 1
    assert audit.drifted_sessions == 1
    assert audit.drifted_session_missing_everywhere == 1
    assert audit.mass_backfill_candidates == 1
    summaries = {s["session_id"]: s for s in audit.per_session}
    drift_summary = summaries[drifted_session]
    clean_summary = summaries[clean_session]
    duplicate_summary = summaries[duplicate_session]
    safe_summary = summaries[safe_session]
    assert drift_summary["honcho_session_key"] == "New-Title"
    assert drift_summary["session_drifted"] is True
    assert drift_summary["missing_ids"] == [drifted_missing]
    assert drift_summary["mass_backfill_candidate_ids"] == []
    assert drift_summary["elsewhere_locations"][0]["hermes_message_id"] == drifted
    assert drift_summary["elsewhere_locations"][0]["locations"][0]["honcho_session_key"] == old_key
    assert clean_summary["session_drifted"] is False
    assert clean_summary["same_content_present_ids"] == [clean_missing]
    assert clean_summary["untagged_present_ids"] == [clean_missing]
    assert clean_summary["mass_backfill_candidate_ids"] == []
    assert duplicate_summary["session_drifted"] is False
    assert duplicate_summary["same_content_present_ids"] == [duplicate_missing]
    assert duplicate_summary["untagged_present_ids"] == []
    assert duplicate_summary["same_content_locations"][0]["locations"][0]["hermes_message_id"] == 999001
    assert duplicate_summary["same_content_locations"][0]["locations"][0]["honcho_session_key"] == "Other-Title"
    assert duplicate_summary["mass_backfill_candidate_ids"] == []
    assert safe_summary["session_drifted"] is False
    assert safe_summary["untagged_present_ids"] == []
    assert safe_summary["mass_backfill_candidate_ids"] == [safe_missing]


def test_reconcile_apply_requires_explicit_session_scope(tmp_path, monkeypatch):
    db, db_path = _make_db(tmp_path, monkeypatch)
    home = db_path.parent
    try:
        session_id = db.create_session("session-unscoped", source="cli")
        first = db.append_message(session_id, "user", "would be unsafe as mass apply")
    finally:
        db.close()
    _write_epoch(home, first)
    manager = FakeHonchoManager(message_max_chars=25000)

    with pytest.raises(ValueError, match="Unscoped --apply is disabled"):
        reconcile_honcho(
            ReconcileOptions(hermes_home=home, state_db_path=db_path, apply=True),
            manager=manager,
        )


def test_reconcile_apply_uploads_tagged_batches_and_rerun_is_idempotent(tmp_path, monkeypatch):
    db, db_path = _make_db(tmp_path, monkeypatch)
    home = db_path.parent
    try:
        session_id = db.create_session("session-2", source="cli")
        first = db.append_message(session_id, "user", "one", timestamp=1700000000.25)
        second = db.append_message(session_id, "assistant", "two", timestamp=1700000001.5)
    finally:
        db.close()
    _write_epoch(home, first)
    manager = FakeHonchoManager(message_max_chars=25000)

    first_report = reconcile_honcho(
        ReconcileOptions(
            hermes_home=home,
            state_db_path=db_path,
            apply=True,
            batch_size=1,
            session_ids=(session_id,),
        ),
        manager=manager,
    )

    remote = manager.remote(session_id)
    assert first_report.gaps == 2
    assert first_report.uploaded == 2
    assert [len(batch) for batch in remote.add_batches] == [1, 1]
    uploaded = [msg for batch in remote.add_batches for msg in batch]
    assert [msg.peer_id for msg in uploaded] == ["user-session-2", "hermes-assistant"]
    assert [msg.metadata["hermes_message_id"] for msg in uploaded] == [first, second]
    assert all(msg.metadata["hermes_ingest_path"] == "reconcile" for msg in uploaded)
    assert all(msg.metadata["hermes_ingest_epoch"] == first for msg in uploaded)
    assert uploaded[0].metadata["hermes_message_hash"] == honcho_ingest_clean_hash("one")
    assert uploaded[0].created_at == datetime.fromtimestamp(1700000000.25, tz=timezone.utc)

    second_report = reconcile_honcho(
        ReconcileOptions(
            hermes_home=home,
            state_db_path=db_path,
            apply=True,
            session_ids=(session_id,),
        ),
        manager=manager,
    )

    assert second_report.gaps == 0
    assert second_report.uploaded == 0
    assert len(remote.add_batches) == 2
