from __future__ import annotations

from typing import Any, Callable, Generic, Optional, Sequence, TypeVar

from myagent.channels.base import BaseChannel
from myagent.errors import EmptyChannelError

T = TypeVar("T")
_EMPTY = object()


class BinaryOperatorAggregate(BaseChannel[T, T, T], Generic[T]):
    """Channel that aggregates values using a binary operator (reducer)."""

    def __init__(self, typ: type[T] = object, operator: Optional[Callable[[T, T], T]] = None):
        self.typ = typ
        self.operator = operator
        self.value: Any = _EMPTY

    def update(self, values: Sequence[T]) -> bool:
        if len(values) == 0:
            return False
        new_value = self.value
        for val in values:
            if new_value is _EMPTY:
                new_value = val
            elif self.operator is not None:
                new_value = self.operator(new_value, val)
        old = self.value
        self.value = new_value
        return self.value is not old

    def get(self) -> T:
        if self.value is _EMPTY:
            raise EmptyChannelError("Channel has no value")
        return self.value

    def checkpoint(self) -> Any:
        if self.value is _EMPTY:
            return _EMPTY
        return self.value

    def from_checkpoint(self, checkpoint: Optional[T]) -> BinaryOperatorAggregate[T]:
        new = BinaryOperatorAggregate(self.typ, self.operator)
        if checkpoint is not None and checkpoint is not _EMPTY:
            new.value = checkpoint
        return new
