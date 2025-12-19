from __future__ import annotations

from typing import Any, Optional, Sequence

from myagent.channels.base import BaseChannel
from myagent.errors import EmptyChannelError

_EMPTY = object()


class EphemeralValue(BaseChannel[Any, Any, None]):
    """Channel that holds a value for one superstep, then auto-clears."""

    def __init__(self, typ: type = object, guard: bool = True):
        self.typ = typ
        self.guard = guard
        self.value: Any = _EMPTY

    def update(self, values: Sequence[Any]) -> bool:
        if len(values) == 0:
            return False
        if self.guard and len(values) > 1:
            from myagent.errors import InvalidUpdateError

            raise InvalidUpdateError(f"EphemeralValue received {len(values)} values, expected at most 1")
        self.value = values[-1]
        return True

    def get(self) -> Any:
        if self.value is _EMPTY:
            raise EmptyChannelError("Ephemeral channel is empty")
        return self.value

    def checkpoint(self) -> None:
        return None

    def from_checkpoint(self, checkpoint: Optional[None] = None) -> EphemeralValue:
        return EphemeralValue(self.typ, self.guard)

    def clear(self) -> None:
        self.value = _EMPTY
