"""Regression tests for Honcho startup fail-open behavior."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

from plugins.memory.honcho import HonchoMemoryProvider, honcho_ingest_message_hash


class _FakeHonchoConfig(SimpleNamespace):
    def resolve_session_name(self, **kwargs):
        return "test-session"


def _configured_hybrid_config() -> _FakeHonchoConfig:
    return _FakeHonchoConfig(
        enabled=True,
        api_key=None,
        base_url="http://127.0.0.1:8000",
        recall_mode="hybrid",
        init_on_session_start=False,
        dialectic_depth=1,
        dialectic_depth_levels=None,
        reasoning_heuristic=True,
        reasoning_level_cap="high",
        context_tokens=None,
        message_max_chars=25000,
        session_strategy="per-directory",
    )


def _configured_tools_config(*, init_on_session_start: bool = False) -> _FakeHonchoConfig:
    cfg = _configured_hybrid_config()
    cfg.recall_mode = "tools"
    cfg.init_on_session_start = init_on_session_start
    return cfg


def test_honcho_hybrid_initialize_returns_without_waiting_for_session_init(monkeypatch):
    """Slow Honcho session creation must not block agent startup."""
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    started = threading.Event()
    release = threading.Event()

    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )

    def slow_session_init(self, cfg, session_id, **kwargs):
        started.set()
        release.wait(timeout=5)
        self._session_initialized = True

    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", slow_session_init)

    start = time.perf_counter()
    provider.initialize("session-1", platform="cli")
    elapsed = time.perf_counter() - start

    try:
        assert elapsed < 0.5
        assert started.wait(timeout=1)
        assert provider._session_key == "test-session"
    finally:
        release.set()
        init_thread = getattr(provider, "_init_thread", None)
        if init_thread:
            init_thread.join(timeout=1)


def test_honcho_background_init_rechecks_state_after_lock_race():
    """Startup should not spawn/crash if init completes while waiting for lock."""
    provider = HonchoMemoryProvider()
    provider._config = _configured_hybrid_config()
    provider._lazy_init_kwargs = {"platform": "cli"}
    provider._lazy_init_session_id = "session-1"

    class RacingLock:
        def __enter__(self):
            provider._session_initialized = True
            provider._lazy_init_kwargs = None
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    provider._init_lock = RacingLock()

    provider._start_session_init_background()

    assert provider._init_thread is None
    assert provider._session_initialized is True


def test_honcho_prefetch_returns_without_waiting_for_first_context_fetch():
    """First-turn context injection must fail open when Honcho is slow."""
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    cfg.timeout = 0.1
    fetch_started = threading.Event()

    class SlowManager:
        def get_prefetch_context(self, session_key, user_message=None):
            fetch_started.set()
            time.sleep(5)
            return {"representation": "late"}

        def prefetch_context(self, session_key, user_message=None):
            fetch_started.set()

        def pop_context_result(self, session_key):
            return {}

    provider._config = cfg
    provider._manager = SlowManager()
    provider._session_key = "test-session"
    provider._session_initialized = True
    provider._turn_count = 1

    start = time.perf_counter()
    result = provider.prefetch("what do you know about me?")
    elapsed = time.perf_counter() - start

    assert result == ""
    assert elapsed < 0.5
    assert fetch_started.is_set()



def test_honcho_sync_turn_does_not_start_network_write_before_session_init():
    """Session-end sync must not create a blocking writer before init finishes."""
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    get_started = threading.Event()
    background_started = threading.Event()
    release_init = threading.Event()

    class SlowManager:
        def get_or_create(self, session_key):
            get_started.set()
            time.sleep(5)
            return SimpleNamespace()

        def _flush_session(self, session):
            pass

    provider._config = cfg
    provider._manager = SlowManager()
    provider._session_key = "test-session"
    provider._session_initialized = False
    provider._start_session_init_background = background_started.set
    provider._init_thread = threading.Thread(
        target=lambda: release_init.wait(timeout=5), daemon=True
    )
    provider._init_thread.start()

    try:
        provider.sync_turn("hello", "world")

        assert provider._sync_thread is None
        assert background_started.is_set()
        assert not get_started.wait(timeout=0.1)
    finally:
        release_init.set()
        provider._init_thread.join(timeout=1)


def test_honcho_sync_turn_waits_for_full_background_startup(monkeypatch):
    """Manager assignment alone is not readiness while background init continues."""
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    session_created = threading.Event()
    migration_started = threading.Event()
    release_migration = threading.Event()
    get_calls = []

    class StartupManager:
        def __init__(self, *args, **kwargs):
            pass

        def get_or_create(self, session_key):
            get_calls.append(session_key)
            session_created.set()
            return SimpleNamespace(messages=[])

        def migrate_memory_files(self, session_key, mem_dir):
            migration_started.set()
            release_migration.wait(timeout=5)

        def prefetch_context(self, session_key, user_message=None):
            pass

        def _flush_session(self, session):
            pass

    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )
    monkeypatch.setattr("plugins.memory.honcho.client.get_honcho_client", lambda cfg: object())
    monkeypatch.setattr("plugins.memory.honcho.session.HonchoSessionManager", StartupManager)

    provider.initialize("session-1", platform="cli")
    try:
        assert session_created.wait(timeout=1)
        assert migration_started.wait(timeout=1)
        assert provider._manager is not None
        assert provider._session_initialized is False

        provider.sync_turn("hello", "world")

        assert provider._sync_thread is None
        assert get_calls == ["test-session"]
    finally:
        release_migration.set()
        init_thread = getattr(provider, "_init_thread", None)
        if init_thread:
            init_thread.join(timeout=1)
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=1)

    assert provider._session_initialized is True


def test_honcho_sync_turn_tags_live_messages_with_state_db_metadata(monkeypatch, tmp_path):
    """Post-epoch live Honcho writes must carry the durable state.db replay
    key and timestamp so a later reconciler can use pure metadata dedup.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    cfg.message_max_chars = 25000
    captured = {}

    class RecordingSession(SimpleNamespace):
        def __init__(self):
            super().__init__(messages=[])

        def add_message(self, role, content, **kwargs):
            self.messages.append({"role": role, "content": content, **kwargs})

    recording_session = RecordingSession()

    class RecordingManager:
        def get_or_create(self, session_key):
            captured["session_key"] = session_key
            return recording_session

        def _flush_session(self, session):
            captured["flushed"] = list(session.messages)

    provider._config = cfg
    provider._manager = RecordingManager()
    provider._session_key = "test-session"
    provider._session_initialized = True

    provider.sync_turn(
        "hello",
        "world",
        messages=[
            {
                "role": "user",
                "content": "hello",
                "_session_db_message_id": 101,
                "_session_db_timestamp": 1700000000.25,
            },
            {
                "role": "assistant",
                "content": "world",
                "_session_db_message_id": 102,
                "_session_db_timestamp": 1700000001.5,
            },
        ],
    )
    provider._sync_thread.join(timeout=1)

    assert captured["session_key"] == "test-session"
    assert len(captured["flushed"]) == 2
    user_msg, assistant_msg = captured["flushed"]
    assert user_msg["metadata"]["hermes_message_id"] == 101
    assert user_msg["metadata"]["hermes_ingest_epoch"] == 101
    assert user_msg["metadata"]["hermes_message_role"] == "user"
    assert user_msg["metadata"]["hermes_message_hash"] == honcho_ingest_message_hash("hello")
    assert user_msg["metadata"]["hermes_chunk_index"] == 0
    assert user_msg["metadata"]["hermes_chunk_count"] == 1
    assert user_msg["created_at"] == datetime.fromtimestamp(1700000000.25, tz=timezone.utc)
    assert assistant_msg["metadata"]["hermes_message_id"] == 102
    assert assistant_msg["metadata"]["hermes_ingest_epoch"] == 101
    assert assistant_msg["metadata"]["hermes_message_role"] == "assistant"
    assert assistant_msg["created_at"] == datetime.fromtimestamp(1700000001.5, tz=timezone.utc)
    epoch_marker = json.loads((tmp_path / "honcho_metadata_epoch.json").read_text())
    assert epoch_marker["first_tagged_message_id"] == 101
    assert epoch_marker["hermes_ingest_epoch"] == 101


def test_honcho_ingest_epoch_marker_is_global_write_once(monkeypatch, tmp_path):
    """The cutover marker is a global messages.id watermark, not per-session state."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = HonchoMemoryProvider()
    provider._session_key = "first-session"

    assert provider._ensure_ingest_epoch(101, session_id="first-session") == 101
    provider._session_key = "second-session"
    assert provider._ensure_ingest_epoch(250, session_id="second-session") == 101

    marker = json.loads((tmp_path / "honcho_metadata_epoch.json").read_text())
    assert marker["first_tagged_message_id"] == 101
    assert marker["hermes_ingest_epoch"] == 101
    assert marker["session_id"] == "first-session"


def test_honcho_sync_turn_without_messages_keeps_legacy_untagged_ingestion():
    """messages=None must preserve the pre-Slice-1 soft-fail behavior."""
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    cfg.message_max_chars = 25000
    captured = {}

    class RecordingSession(SimpleNamespace):
        def __init__(self):
            super().__init__(messages=[])

        def add_message(self, role, content, **kwargs):
            self.messages.append({"role": role, "content": content, **kwargs})

    class RecordingManager:
        def __init__(self):
            self.session = RecordingSession()

        def get_or_create(self, session_key):
            return self.session

        def _flush_session(self, session):
            captured["flushed"] = list(session.messages)

    provider._config = cfg
    provider._manager = RecordingManager()
    provider._session_key = "test-session"
    provider._session_initialized = True

    provider.sync_turn("hello", "world")
    provider._sync_thread.join(timeout=1)

    assert captured["flushed"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]


def test_honcho_sync_turn_metadata_mapping_failure_records_untagged_exclusions(
    monkeypatch, tmp_path
):
    """Untagged above-epoch fallback must be excluded from Slice-2 gap fill."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    cfg.message_max_chars = 25000
    captured = {}

    class RecordingSession(SimpleNamespace):
        def __init__(self):
            super().__init__(messages=[])

        def add_message(self, role, content, **kwargs):
            self.messages.append({"role": role, "content": content, **kwargs})

    class RecordingManager:
        def __init__(self):
            self.session = RecordingSession()

        def get_or_create(self, session_key):
            return self.session

        def _flush_session(self, session):
            captured["flushed"] = list(session.messages)

    def broken_mapping(*args, **kwargs):
        raise RuntimeError("mapping exploded")

    monkeypatch.setattr(HonchoMemoryProvider, "_find_source_message", broken_mapping)
    provider._config = cfg
    provider._manager = RecordingManager()
    provider._session_key = "test-session"
    provider._session_initialized = True

    provider.sync_turn(
        "hello",
        "world",
        messages=[
            {"role": "user", "content": "hello", "_session_db_message_id": 101},
            {"role": "assistant", "content": "world", "_session_db_message_id": 102},
        ],
    )
    provider._sync_thread.join(timeout=1)

    assert captured["flushed"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    exclusion_path = tmp_path / "honcho_untagged_ingest_exclusions.jsonl"
    records = [json.loads(line) for line in exclusion_path.read_text().splitlines()]
    assert [record["hermes_message_id"] for record in records] == [101, 102]
    assert {record["reason"] for record in records} == {"metadata_source_mapping_failed"}
    assert all(record["hermes_ingest_epoch"] == 101 for record in records)


def test_honcho_sync_turn_metadata_construction_failure_records_untagged_exclusions(
    monkeypatch, tmp_path
):
    """If tag construction fails after mapping, Slice 2 must skip the untagged ids."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    cfg.message_max_chars = 25000
    captured = {}

    class RecordingSession(SimpleNamespace):
        def __init__(self):
            super().__init__(messages=[])

        def add_message(self, role, content, **kwargs):
            self.messages.append({"role": role, "content": content, **kwargs})

    class RecordingManager:
        def __init__(self):
            self.session = RecordingSession()

        def get_or_create(self, session_key):
            return self.session

        def _flush_session(self, session):
            captured["flushed"] = list(session.messages)

    def broken_metadata(*args, **kwargs):
        raise RuntimeError("metadata exploded")

    monkeypatch.setattr(HonchoMemoryProvider, "_chunk_metadata", broken_metadata)
    provider._config = cfg
    provider._manager = RecordingManager()
    provider._session_key = "test-session"
    provider._session_initialized = True

    provider.sync_turn(
        "hello",
        "world",
        messages=[
            {"role": "user", "content": "hello", "_session_db_message_id": 101},
            {"role": "assistant", "content": "world", "_session_db_message_id": 102},
        ],
    )
    provider._sync_thread.join(timeout=1)

    assert captured["flushed"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    exclusion_path = tmp_path / "honcho_untagged_ingest_exclusions.jsonl"
    records = [json.loads(line) for line in exclusion_path.read_text().splitlines()]
    assert [record["hermes_message_id"] for record in records] == [101, 102]
    assert {record["reason"] for record in records} == {"metadata_construction_failed"}
    assert all(record["hermes_ingest_epoch"] == 101 for record in records)


def test_honcho_system_prompt_advertises_active_while_background_init_runs(monkeypatch):
    """Prompt metadata should not require a completed network session."""
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    release = threading.Event()

    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )

    def slow_session_init(self, cfg, session_id, **kwargs):
        release.wait(timeout=5)
        self._session_initialized = True

    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", slow_session_init)

    provider.initialize("session-1", platform="cli")
    try:
        prompt = provider.system_prompt_block()
        assert "Honcho Memory" in prompt
        assert "hybrid mode" in prompt
    finally:
        release.set()
        init_thread = getattr(provider, "_init_thread", None)
        if init_thread:
            init_thread.join(timeout=1)


def test_honcho_tools_eager_init_still_ready_on_return(monkeypatch):
    """tools + initOnSessionStart=true keeps its ready-on-return contract."""
    provider = HonchoMemoryProvider()
    cfg = _configured_tools_config(init_on_session_start=True)

    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )

    def fake_session_init(self, cfg, session_id, **kwargs):
        self._manager = SimpleNamespace()
        self._session_key = "test-session"
        self._session_initialized = True

    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", fake_session_init)

    provider.initialize("session-1", platform="cli")

    assert provider._session_initialized is True
    assert provider._manager is not None
    assert provider._init_thread is None


def test_honcho_tools_eager_init_failure_does_not_leave_ready_manager(monkeypatch):
    """Failed eager tools startup must not leave hooks seeing a ready session."""
    provider = HonchoMemoryProvider()
    cfg = _configured_tools_config(init_on_session_start=True)

    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )

    def failing_session_init(self, cfg, session_id, **kwargs):
        self._manager = SimpleNamespace()
        self._session_key = "test-session"
        raise RuntimeError("boom")

    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", failing_session_init)

    provider.initialize("session-1", platform="cli")
    assert provider._session_initialized is False
    assert provider._manager is None

    background_started = threading.Event()
    provider._start_session_init_background = background_started.set
    provider.sync_turn("hello", "world")
    provider.on_memory_write("add", "user", "prefers safe Honcho startup")

    assert provider._sync_thread is None
    assert not background_started.is_set()

    result = json.loads(provider.handle_tool_call("honcho_profile", {"peer": "user"}))
    assert "could not be initialized" in result["error"]
    assert provider._manager is None


def test_honcho_tools_lazy_hooks_do_not_prestart_background_init(monkeypatch):
    """tools lazy mode lets the first tool call own session initialization."""
    provider = HonchoMemoryProvider()
    cfg = _configured_tools_config(init_on_session_start=False)

    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )

    provider.initialize("session-1", platform="cli")
    background_started = threading.Event()
    provider._start_session_init_background = background_started.set

    provider.prefetch("what do you know?")
    provider.queue_prefetch("what do you know?")
    provider.sync_turn("hello", "world")
    provider.on_memory_write("add", "user", "prefers fail-open memory")

    assert not background_started.is_set()
    assert provider._session_initialized is False

    class ToolManager:
        def get_peer_card(self, session_key, peer="user"):
            return ["ready"]

    init_calls = []

    def fake_session_init(self, cfg, session_id, **kwargs):
        init_calls.append(session_id)
        self._manager = ToolManager()
        self._session_key = "test-session"
        self._session_initialized = True

    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", fake_session_init)

    result = json.loads(provider.handle_tool_call("honcho_profile", {"peer": "user"}))

    assert result == {"result": ["ready"]}
    assert init_calls == ["session-1"]
    assert not background_started.is_set()
