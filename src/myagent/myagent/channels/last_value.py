from __future__ import annotations

from typing import Any, Generic, Optional, Sequence, TypeVar

from myagent.channels.base import BaseChannel
from myagent.errors import EmptyChannelError, InvalidUpdateError

T = TypeVar("T")
_EMPTY = object()


class LastValue(BaseChannel[T, T, T], Generic[T]):
    """Channel that stores the last value written. Default for plain TypedDict fields."""

    def __init__(self, typ: type[T] = object):
        self.typ = typ
        self.value: Any = _EMPTY

    def update(self, values: Sequence[T]) -> bool:
        if len(values) == 0:
            return False
        if len(values) > 1:
            raise InvalidUpdateError(f"LastValue channel received {len(values)} values, expected at most 1")
        old = self.value
        self.value = values[0]
        return self.value is not old

    def get(self) -> T:
        if self.value is _EMPTY:
            raise EmptyChannelError("Channel has no value")
        return self.value

    def checkpoint(self) -> Any:
        if self.value is _EMPTY:
            return _EMPTY
        return self.value

    def from_checkpoint(self, checkpoint: Optional[T]) -> LastValue[T]:
        new = LastValue(self.typ)
        if checkpoint is not None and checkpoint is not _EMPTY:
            new.value = checkpoint
        return new
