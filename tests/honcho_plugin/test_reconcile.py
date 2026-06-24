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

from hermes_state import SessionDB
from plugins.memory.honcho import honcho_ingest_clean_hash, honcho_ingest_clean_text
from plugins.memory.honcho.reconcile import (
    ReconcileOptions,
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
    def __init__(self, existing=None):
        self.existing = list(existing or [])
        self.add_batches = []

    def messages(self):
        return list(self.existing)

    def add_messages(self, messages):
        batch = list(messages)
        self.add_batches.append(batch)
        self.existing.extend(batch)


class FakeHonchoManager:
    def __init__(self, *, existing_by_key=None, message_max_chars=25000, config=None):
        self._config = config or SimpleNamespace(message_max_chars=message_max_chars)
        self._cache = {}
        self._peers = {}
        self._sessions_cache = {}
        for key, messages in (existing_by_key or {}).items():
            self._sessions_cache[self._sanitize(key)] = FakeRemoteSession(messages)

    @staticmethod
    def _sanitize(value: str) -> str:
        return str(value).replace(":", "-")

    def _get_or_create_peer(self, peer_id):
        self._peers.setdefault(peer_id, FakePeer(peer_id))
        return self._peers[peer_id]

    def _get_or_create_honcho_session(self, honcho_session_id, user_peer, assistant_peer):
        self._sessions_cache.setdefault(honcho_session_id, FakeRemoteSession())
        return self._sessions_cache[honcho_session_id], []

    def get_or_create(self, key):
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
    assert manager.remote(session_id).add_batches == []


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
        ReconcileOptions(hermes_home=home, state_db_path=db_path, apply=True),
        manager=manager,
    )

    assert second_report.gaps == 0
    assert second_report.uploaded == 0
    assert len(remote.add_batches) == 2
