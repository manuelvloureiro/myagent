from __future__ import annotations

from typing import Any, Iterator, Optional

from myagent_core.runnable.config import RunnableConfig

from myagent.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)


class InMemorySaver(BaseCheckpointSaver):
    """In-memory checkpoint saver backed by a dict."""

    def __init__(self):
        self.storage: dict[str, dict[str, CheckpointTuple]] = {}

    def _key(self, config: RunnableConfig) -> str:
        return config.get("configurable", {}).get("thread_id", "")

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[dict[str, int]] = None,
    ) -> RunnableConfig:
        thread_id = self._key(config)
        if thread_id not in self.storage:
            self.storage[thread_id] = {}

        checkpoint_id = checkpoint["id"]
        parent_id = config.get("configurable", {}).get("checkpoint_id")

        new_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }
        parent_config: Optional[RunnableConfig] = None
        if parent_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": parent_id,
                }
            }

        self.storage[thread_id][checkpoint_id] = CheckpointTuple(
            config=new_config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
        )
        return new_config

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = self._key(config)
        checkpoints = self.storage.get(thread_id, {})
        if not checkpoints:
            return None

        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")
        if checkpoint_id:
            return checkpoints.get(checkpoint_id)

        # Return the latest checkpoint (by timestamp)
        latest = max(checkpoints.values(), key=lambda t: t.checkpoint["ts"])
        return latest

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        if config is None:
            all_tuples = []
            for thread_checkpoints in self.storage.values():
                all_tuples.extend(thread_checkpoints.values())
        else:
            thread_id = self._key(config)
            all_tuples = list(self.storage.get(thread_id, {}).values())

        all_tuples.sort(key=lambda t: t.checkpoint["ts"], reverse=True)

        if before:
            before_tuple = self._resolve_before_tuple(before)
            if before_tuple:
                before_ts = before_tuple.checkpoint["ts"]
                before_id = before_tuple.checkpoint["id"]
                all_tuples = [
                    t for t in all_tuples if (t.checkpoint["ts"], t.checkpoint["id"]) < (before_ts, before_id)
                ]

        if limit:
            all_tuples = all_tuples[:limit]

        yield from all_tuples

    def _resolve_before_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        before_id = config.get("configurable", {}).get("checkpoint_id")
        if not before_id:
            return None

        thread_id = self._key(config)
        if thread_id:
            return self.storage.get(thread_id, {}).get(before_id)

        for thread_checkpoints in self.storage.values():
            checkpoint_tuple = thread_checkpoints.get(before_id)
            if checkpoint_tuple is not None:
                return checkpoint_tuple

        return None


MemorySaver = InMemorySaver
