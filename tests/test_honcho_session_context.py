"""Tests for Honcho session context peer resolution."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from honcho.api_types import MessageCreateParams
from honcho.http.exceptions import UnprocessableEntityError
from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager


class _FakeSummary:
    content = "summary"


class _FakeContext:
    summary = _FakeSummary()
    peer_representation = "representation"
    peer_card = ["fact"]
    messages = []


class _RecordingHonchoSession:
    def __init__(self):
        self.calls = []

    def context(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeContext()


def _manager_with_cached_session(*, ai_observe_others=True):
    cfg = SimpleNamespace(
        write_frequency="turn",
        dialectic_reasoning_level="low",
        dialectic_dynamic=True,
        dialectic_max_chars=600,
        observation_mode="directional",
        user_observe_me=True,
        user_observe_others=True,
        ai_observe_me=True,
        ai_observe_others=ai_observe_others,
        message_max_chars=25000,
        dialectic_max_input_chars=10000,
    )
    mgr = HonchoSessionManager(honcho=SimpleNamespace(), config=cfg)
    session = HonchoSession(
        key="test-session",
        user_peer_id="chris",
        assistant_peer_id="hermes",
        honcho_session_id="test-session",
    )
    fake_honcho_session = _RecordingHonchoSession()
    mgr._cache[session.key] = session
    mgr._sessions_cache[session.honcho_session_id] = fake_honcho_session
    return mgr, fake_honcho_session


def test_session_context_user_alias_uses_assistant_observer_when_ai_can_observe_others():
    mgr, fake = _manager_with_cached_session(ai_observe_others=True)

    result = mgr.get_session_context("test-session", peer="user")

    assert result["summary"] == "summary"
    assert fake.calls == [
        {
            "summary": True,
            "peer_target": "chris",
            "peer_perspective": "hermes",
        }
    ]


def test_session_context_explicit_user_peer_matches_user_alias():
    mgr, fake = _manager_with_cached_session(ai_observe_others=True)

    mgr.get_session_context("test-session", peer="chris")

    assert fake.calls == [
        {
            "summary": True,
            "peer_target": "chris",
            "peer_perspective": "hermes",
        }
    ]


def test_session_context_user_alias_uses_user_self_observer_when_ai_cannot_observe_others():
    mgr, fake = _manager_with_cached_session(ai_observe_others=False)

    mgr.get_session_context("test-session", peer="user")

    assert fake.calls == [
        {
            "summary": True,
            "peer_target": "chris",
            "peer_perspective": "chris",
        }
    ]


def test_flush_session_batches_messages_and_preserves_metadata_created_at():
    """The live path flushes all pending chunks through one add_messages call;
    metadata and created_at must survive from the provider cache to the SDK.
    """
    mgr, _ = _manager_with_cached_session(ai_observe_others=True)
    session = mgr._cache["test-session"]
    created_at = datetime.fromtimestamp(1700000000.25, tz=timezone.utc)
    session.messages = [
        {
            "role": "user",
            "content": "hello",
            "metadata": {"hermes_message_id": 101},
            "created_at": created_at,
        },
        {
            "role": "assistant",
            "content": "world",
            "metadata": {"hermes_message_id": 102},
            "created_at": created_at,
        },
    ]

    class RecordingPeer:
        def __init__(self, peer_id):
            self.id = peer_id

        def message(self, content, **kwargs):
            return {"peer_id": self.id, "content": content, **kwargs}

    class RecordingHonchoSession:
        def __init__(self):
            self.calls = []

        def add_messages(self, messages):
            self.calls.append(messages)

    recording_honcho_session = RecordingHonchoSession()
    mgr._get_or_create_peer = lambda peer_id: RecordingPeer(peer_id)
    mgr._sessions_cache[session.honcho_session_id] = recording_honcho_session

    assert mgr._flush_session(session) is True

    assert len(recording_honcho_session.calls) == 1
    batch = recording_honcho_session.calls[0]
    assert [message["content"] for message in batch] == ["hello", "world"]
    assert batch[0]["metadata"] == {"hermes_message_id": 101}
    assert batch[0]["created_at"] == created_at
    assert batch[1]["metadata"] == {"hermes_message_id": 102}
    assert all(message["_synced"] for message in session.messages)


def test_flush_session_retries_untagged_if_tagged_message_construction_fails(
    monkeypatch, tmp_path
):
    """Metadata rejection must degrade to legacy untagged ingestion."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    mgr, _ = _manager_with_cached_session(ai_observe_others=True)
    session = mgr._cache["test-session"]
    tagged_metadata = {
        "hermes_message_id": 101,
        "hermes_ingest_epoch": 101,
        "hermes_message_hash": "abc123",
    }
    session.messages = [
        {
            "role": "user",
            "content": "hello",
            "metadata": tagged_metadata,
            "created_at": "not-a-date",
        }
    ]

    class RecordingPeer:
        id = "chris"

        def __init__(self):
            self.calls = []

        def message(self, content, **kwargs):
            self.calls.append((content, kwargs))
            if kwargs:
                raise ValueError("metadata rejected")
            return {"content": content}

    class RecordingHonchoSession:
        def __init__(self):
            self.calls = []

        def add_messages(self, messages):
            self.calls.append(messages)

    peer = RecordingPeer()
    recording_honcho_session = RecordingHonchoSession()
    mgr._get_or_create_peer = lambda peer_id: peer
    mgr._sessions_cache[session.honcho_session_id] = recording_honcho_session

    assert mgr._flush_session(session) is True

    assert peer.calls == [
        ("hello", {"metadata": tagged_metadata, "created_at": "not-a-date"}),
        ("hello", {}),
    ]
    assert recording_honcho_session.calls == [[{"content": "hello"}]]
    assert session.messages[0]["_synced"] is True
    exclusion_path = tmp_path / "honcho_untagged_ingest_exclusions.jsonl"
    records = [json.loads(line) for line in exclusion_path.read_text().splitlines()]
    assert records[0]["hermes_message_id"] == 101
    assert records[0]["reason"] == "tagged_message_construction_failed"
    assert records[0]["hermes_ingest_epoch"] == 101


def test_flush_session_retries_untagged_if_tagged_batch_is_rejected_before_write(
    monkeypatch, tmp_path
):
    """A metadata-shape 4xx from add_messages is pre-write, so retry untagged."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    mgr, _ = _manager_with_cached_session(ai_observe_others=True)
    session = mgr._cache["test-session"]
    created_at = datetime.fromtimestamp(1700000000.25, tz=timezone.utc)
    session.messages = [
        {
            "role": "user",
            "content": "hello",
            "metadata": {"hermes_message_id": 101, "hermes_ingest_epoch": 101},
            "created_at": created_at,
        },
        {
            "role": "assistant",
            "content": "world",
            "metadata": {"hermes_message_id": 102, "hermes_ingest_epoch": 101},
            "created_at": created_at,
        },
    ]

    class RecordingPeer:
        def __init__(self, peer_id):
            self.id = peer_id

        def message(self, content, **kwargs):
            return {"peer_id": self.id, "content": content, **kwargs}

    class RejectOnceHonchoSession:
        def __init__(self):
            self.calls = []

        def add_messages(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 1:
                raise UnprocessableEntityError("metadata rejected")

    recording_honcho_session = RejectOnceHonchoSession()
    mgr._get_or_create_peer = lambda peer_id: RecordingPeer(peer_id)
    mgr._sessions_cache[session.honcho_session_id] = recording_honcho_session

    assert mgr._flush_session(session) is True

    assert len(recording_honcho_session.calls) == 2
    tagged_batch, untagged_batch = recording_honcho_session.calls
    assert all("metadata" in message for message in tagged_batch)
    assert [message["content"] for message in untagged_batch] == ["hello", "world"]
    assert all("metadata" not in message for message in untagged_batch)
    assert all("created_at" not in message for message in untagged_batch)
    assert all(message["_synced"] for message in session.messages)
    exclusion_path = tmp_path / "honcho_untagged_ingest_exclusions.jsonl"
    records = [json.loads(line) for line in exclusion_path.read_text().splitlines()]
    assert [record["hermes_message_id"] for record in records] == [101, 102]
    assert {record["reason"] for record in records} == {"tagged_batch_rejected_untagged_retry"}
    assert all(record["hermes_ingest_epoch"] == 101 for record in records)


def test_flush_session_does_not_double_record_when_construction_fallback_batch_rejected(
    monkeypatch, tmp_path
):
    """Batch 4xx fallback should supersede per-message construction fallback markers."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    mgr, _ = _manager_with_cached_session(ai_observe_others=True)
    session = mgr._cache["test-session"]
    created_at = datetime.fromtimestamp(1700000000.25, tz=timezone.utc)
    session.messages = [
        {
            "role": "user",
            "content": "hello",
            "metadata": {
                "hermes_message_id": 101,
                "hermes_ingest_epoch": 101,
                "hermes_message_hash": "user-hash",
            },
            "created_at": created_at,
        },
        {
            "role": "assistant",
            "content": "world",
            "metadata": {
                "hermes_message_id": 102,
                "hermes_ingest_epoch": 101,
                "hermes_message_hash": "assistant-hash",
            },
            "created_at": created_at,
        },
    ]

    class RecordingPeer:
        def __init__(self, peer_id):
            self.id = peer_id

        def message(self, content, **kwargs):
            if self.id == "chris" and kwargs:
                raise ValueError("user tagged payload rejected locally")
            return {"peer_id": self.id, "content": content, **kwargs}

    class RejectTaggedBatchOnceHonchoSession:
        def __init__(self):
            self.calls = []

        def add_messages(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 1:
                raise UnprocessableEntityError("metadata rejected")

    recording_honcho_session = RejectTaggedBatchOnceHonchoSession()
    mgr._get_or_create_peer = lambda peer_id: RecordingPeer(peer_id)
    mgr._sessions_cache[session.honcho_session_id] = recording_honcho_session

    assert mgr._flush_session(session) is True

    records = [
        json.loads(line)
        for line in (tmp_path / "honcho_untagged_ingest_exclusions.jsonl").read_text().splitlines()
    ]
    assert [record["hermes_message_id"] for record in records] == [101, 102]
    assert {record["reason"] for record in records} == {"tagged_batch_rejected_untagged_retry"}


def test_honcho_sdk_serializes_created_at_as_utc_iso8601():
    created_at = datetime.fromtimestamp(1700000000.25, tz=timezone.utc)
    msg = MessageCreateParams(peer_id="chris", content="hello", created_at=created_at)

    assert msg.model_dump(mode="json", exclude_none=True)["created_at"] == (
        "2023-11-14T22:13:20.250000Z"
    )
