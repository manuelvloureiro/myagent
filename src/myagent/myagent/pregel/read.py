from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from myagent.types import RetryPolicy


class PregelNode:
    """Wraps a node callable with its channel mappings and retry policy."""

    def __init__(
        self,
        *,
        name: str,
        bound: Callable,
        triggers: Sequence[str] = (),
        channels: Optional[dict[str, str]] = None,
        mapper: Optional[Callable] = None,
        writers: Sequence[Any] = (),
        retry_policy: Optional[RetryPolicy] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.name = name
        self.bound = bound
        self.triggers = list(triggers)
        self.channels = channels or {}
        self.mapper = mapper
        self.writers = list(writers)
        self.retry_policy = retry_policy
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"PregelNode(name={self.name!r})"
