from __future__ import annotations

from typing import (
    Any,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Union,
)


class RetryPolicy(NamedTuple):
    initial_interval: float = 0.5
    backoff_factor: float = 2.0
    max_interval: float = 128.0
    max_attempts: int = 3
    jitter: bool = True
    retry_on: type[Exception] | tuple[type[Exception], ...] = Exception


StreamMode = Literal["values", "updates"]

All = Literal["*"]

Checkpointer = Any  # type alias for BaseCheckpointSaver


class Send:
    """Send a value to a specific node."""

    def __init__(self, node: str, arg: Any):
        self.node = node
        self.arg = arg

    def __repr__(self) -> str:
        return f"Send(node={self.node!r}, arg={self.arg!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Send):
            return self.node == other.node and self.arg == other.arg
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.node, id(self.arg)))


class Command:
    """Command to control graph execution."""

    def __init__(
        self,
        *,
        goto: Optional[Union[str, Sequence[str], Send, Sequence[Send]]] = None,
        update: Optional[dict[str, Any]] = None,
        resume: Optional[Any] = None,
    ):
        self.goto = goto
        self.update = update
        self.resume = resume

    def __repr__(self) -> str:
        parts = []
        if self.goto is not None:
            parts.append(f"goto={self.goto!r}")
        if self.update is not None:
            parts.append(f"update={self.update!r}")
        return f"Command({', '.join(parts)})"


class StateSnapshot:
    """Snapshot of the graph state."""

    def __init__(
        self,
        values: dict[str, Any],
        next: tuple[str, ...],
        config: Any,
        metadata: Any = None,
        created_at: Optional[str] = None,
        parent_config: Any = None,
    ):
        self.values = values
        self.next = next
        self.config = config
        self.metadata = metadata
        self.created_at = created_at
        self.parent_config = parent_config

    def __repr__(self) -> str:
        return f"StateSnapshot(values={self.values!r}, next={self.next!r}, config={self.config!r})"
