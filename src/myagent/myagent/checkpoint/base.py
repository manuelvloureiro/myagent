from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional, TypedDict

from myagent_core.runnable.config import RunnableConfig


class CheckpointMetadata(TypedDict, total=False):
    source: str  # "input" | "loop" | "update"
    step: int
    writes: dict[str, Any]
    parents: dict[str, str]
    next: tuple[str, ...]


class Checkpoint(TypedDict):
    v: int
    id: str
    ts: str
    channel_values: dict[str, Any]
    channel_versions: dict[str, int]
    versions_seen: dict[str, dict[str, int]]
    pending_sends: list[Any]


class CheckpointTuple:
    def __init__(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: Optional[CheckpointMetadata] = None,
        parent_config: Optional[RunnableConfig] = None,
        pending_writes: Optional[list[tuple[str, str, Any]]] = None,
    ):
        self.config = config
        self.checkpoint = checkpoint
        self.metadata = metadata or {}
        self.parent_config = parent_config
        self.pending_writes = pending_writes or []


class BaseCheckpointSaver(ABC):
    """Abstract base class for checkpoint savers."""

    @abstractmethod
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[dict[str, int]] = None,
    ) -> RunnableConfig: ...

    @abstractmethod
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]: ...

    @abstractmethod
    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]: ...

    def put_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        task_id: str,
    ) -> None:
        pass

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[dict[str, int]] = None,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return self.get_tuple(config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ):
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        task_id: str,
    ) -> None:
        self.put_writes(config, writes, task_id)
