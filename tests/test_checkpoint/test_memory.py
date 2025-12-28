"""Tests for InMemorySaver."""

import time

from myagent.checkpoint.base import Checkpoint, CheckpointMetadata
from myagent.checkpoint.id import create_checkpoint_id
from myagent.checkpoint.memory import InMemorySaver, MemorySaver
from myagent_core.runnable.config import RunnableConfig


class TestInMemorySaver:
    def test_checkpoint_ids_are_sortable(self):
        ids = [create_checkpoint_id() for _ in range(3)]
        assert ids == sorted(ids)

    def test_alias(self):
        assert MemorySaver is InMemorySaver

    def test_put_and_get(self):
        saver = InMemorySaver()
        config: RunnableConfig = {"configurable": {"thread_id": "t1"}}
        checkpoint: Checkpoint = {
            "v": 1,
            "id": create_checkpoint_id(),
            "ts": "2024-01-01T00:00:00+00:00",
            "channel_values": {"x": 42},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        metadata: CheckpointMetadata = {"source": "loop", "step": 0, "writes": {}, "parents": {}}

        saver.put(config, checkpoint, metadata)
        result = saver.get_tuple(config)
        assert result is not None
        assert result.checkpoint["channel_values"]["x"] == 42

    def test_get_latest(self):
        saver = InMemorySaver()
        config: RunnableConfig = {"configurable": {"thread_id": "t1"}}

        for i in range(3):
            checkpoint: Checkpoint = {
                "v": 1,
                "id": create_checkpoint_id(),
                "ts": f"2024-01-0{i + 1}T00:00:00+00:00",
                "channel_values": {"step": i},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            }
            metadata: CheckpointMetadata = {"source": "loop", "step": i, "writes": {}, "parents": {}}
            saver.put(config, checkpoint, metadata)

        result = saver.get_tuple(config)
        assert result is not None
        assert result.checkpoint["channel_values"]["step"] == 2

    def test_list(self):
        saver = InMemorySaver()
        config: RunnableConfig = {"configurable": {"thread_id": "t1"}}

        for i in range(3):
            checkpoint: Checkpoint = {
                "v": 1,
                "id": create_checkpoint_id(),
                "ts": f"2024-01-0{i + 1}T00:00:00+00:00",
                "channel_values": {"step": i},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            }
            metadata: CheckpointMetadata = {"source": "loop", "step": i, "writes": {}, "parents": {}}
            saver.put(config, checkpoint, metadata)

        items = list(saver.list(config))
        assert len(items) == 3
        # Should be sorted newest first
        assert items[0].checkpoint["channel_values"]["step"] == 2

    def test_list_with_limit(self):
        saver = InMemorySaver()
        config: RunnableConfig = {"configurable": {"thread_id": "t1"}}

        for i in range(5):
            checkpoint: Checkpoint = {
                "v": 1,
                "id": create_checkpoint_id(),
                "ts": f"2024-01-0{i + 1}T00:00:00+00:00",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            }
            metadata: CheckpointMetadata = {"source": "loop", "step": i, "writes": {}, "parents": {}}
            saver.put(config, checkpoint, metadata)

        items = list(saver.list(config, limit=2))
        assert len(items) == 2

    def test_list_before_uses_checkpoint_order(self):
        saver = InMemorySaver()
        config: RunnableConfig = {"configurable": {"thread_id": "t1"}}
        checkpoint_ids = []

        for i in range(3):
            checkpoint: Checkpoint = {
                "v": 1,
                "id": create_checkpoint_id(),
                "ts": f"2024-01-0{i + 1}T00:00:00+00:00",
                "channel_values": {"step": i},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            }
            checkpoint_ids.append(checkpoint["id"])
            metadata: CheckpointMetadata = {"source": "loop", "step": i, "writes": {}, "parents": {}}
            saver.put(config, checkpoint, metadata)
            time.sleep(0.000001)

        items = list(
            saver.list(config, before={"configurable": {"thread_id": "t1", "checkpoint_id": checkpoint_ids[1]}})
        )
        assert [item.checkpoint["channel_values"]["step"] for item in items] == [0]

    def test_list_before_without_thread_id_finds_checkpoint(self):
        saver = InMemorySaver()
        config: RunnableConfig = {"configurable": {"thread_id": "t1"}}

        first: Checkpoint = {
            "v": 1,
            "id": create_checkpoint_id(),
            "ts": "2024-01-01T00:00:00+00:00",
            "channel_values": {"step": 0},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        second: Checkpoint = {
            "v": 1,
            "id": create_checkpoint_id(),
            "ts": "2024-01-02T00:00:00+00:00",
            "channel_values": {"step": 1},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        metadata: CheckpointMetadata = {"source": "loop", "step": 0, "writes": {}, "parents": {}}
        saver.put(config, first, metadata)
        saver.put(config, second, metadata)

        items = list(saver.list(None, before={"configurable": {"checkpoint_id": second["id"]}}))
        assert [item.checkpoint["channel_values"]["step"] for item in items] == [0]

    def test_empty_get(self):
        saver = InMemorySaver()
        config: RunnableConfig = {"configurable": {"thread_id": "nonexistent"}}
        assert saver.get_tuple(config) is None
