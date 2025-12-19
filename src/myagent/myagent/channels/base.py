from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Optional, Sequence, TypeVar

Value = TypeVar("Value")
Update = TypeVar("Update")
C = TypeVar("C")


class BaseChannel(ABC, Generic[Value, Update, C]):
    """Abstract base class for state channels."""

    @abstractmethod
    def update(self, values: Sequence[Update]) -> bool:
        """Apply updates. Returns True if channel value changed."""
        ...

    @abstractmethod
    def get(self) -> Value:
        """Get current value. Raises EmptyChannelError if empty."""
        ...

    @abstractmethod
    def checkpoint(self) -> C:
        """Return serializable checkpoint of current state."""
        ...

    @abstractmethod
    def from_checkpoint(self, checkpoint: Optional[C]) -> BaseChannel:
        """Restore channel from checkpoint."""
        ...
