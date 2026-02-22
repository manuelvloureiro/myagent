"""Message types compatible with langchain_core.messages.

Provides the standard message hierarchy used throughout the LangChain ecosystem::

    from myagent_core.messages import HumanMessage, AIMessage, SystemMessage

    messages = [
        SystemMessage(content="You are helpful."),
        HumanMessage(content="Hello!"),
    ]
"""

from __future__ import annotations

import uuid
from typing import Any, Literal, Optional, Union


class BaseMessage:
    """Base class for all message types.

    Attributes:
        content: The message text (or list of content blocks).
        type: Message type identifier (e.g. "human", "ai", "system", "tool").
        id: Unique message identifier.
        name: Optional name for the message sender.
        additional_kwargs: Extra provider-specific data.
        response_metadata: Metadata from model responses.
    """

    type: str = "base"

    def __init__(
        self,
        content: Union[str, list[Any]],
        *,
        id: Optional[str] = None,
        name: Optional[str] = None,
        additional_kwargs: Optional[dict[str, Any]] = None,
        response_metadata: Optional[dict[str, Any]] = None,
    ):
        self.content = content
        self.id = id or str(uuid.uuid4())
        self.name = name
        self.additional_kwargs: dict[str, Any] = additional_kwargs or {}
        self.response_metadata: dict[str, Any] = response_metadata or {}

    def __repr__(self) -> str:
        content_repr = repr(self.content) if len(repr(self.content)) < 80 else repr(self.content)[:77] + "..."
        return f"{self.__class__.__name__}(content={content_repr})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseMessage):
            return NotImplemented
        return self.type == other.type and self.content == other.content and self.id == other.id

    def __hash__(self) -> int:
        return hash((self.type, self.id))

    def dict(self) -> dict[str, Any]:
        """Serialize to dict (langchain-core compatible)."""
        d: dict[str, Any] = {
            "type": self.type,
            "content": self.content,
            "id": self.id,
            "additional_kwargs": self.additional_kwargs,
            "response_metadata": self.response_metadata,
        }
        if self.name is not None:
            d["name"] = self.name
        return d


class HumanMessage(BaseMessage):
    """A message from a human user."""

    type: str = "human"


class AIMessage(BaseMessage):
    """A message from an AI model.

    Attributes:
        tool_calls: List of tool calls requested by the model.
        usage_metadata: Token usage information.
    """

    type: str = "ai"

    def __init__(
        self,
        content: Union[str, list[Any]] = "",
        *,
        tool_calls: Optional[list[ToolCall]] = None,
        usage_metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(content, **kwargs)
        self.tool_calls: list[ToolCall] = tool_calls or []
        self.usage_metadata: Optional[dict[str, Any]] = usage_metadata

    def dict(self) -> dict[str, Any]:
        d = super().dict()
        if self.tool_calls:
            d["tool_calls"] = [tc.dict() for tc in self.tool_calls]
        if self.usage_metadata:
            d["usage_metadata"] = self.usage_metadata
        return d


class SystemMessage(BaseMessage):
    """A system instruction message."""

    type: str = "system"


class ToolMessage(BaseMessage):
    """A message containing the result of a tool invocation.

    Attributes:
        tool_call_id: The ID of the tool call this message responds to.
    """

    type: str = "tool"

    def __init__(self, content: Union[str, list[Any]], *, tool_call_id: str, **kwargs: Any):
        super().__init__(content, **kwargs)
        self.tool_call_id = tool_call_id

    def dict(self) -> dict[str, Any]:
        d = super().dict()
        d["tool_call_id"] = self.tool_call_id
        return d


class ToolCall:
    """Represents a tool call from an AI model.

    Attributes:
        name: The name of the tool to call.
        args: The arguments to pass.
        id: Unique identifier for this call.
        type: Always "tool_call".
    """

    def __init__(self, name: str, args: dict[str, Any], id: Optional[str] = None):
        self.name = name
        self.args = args
        self.id = id or str(uuid.uuid4())
        self.type: Literal["tool_call"] = "tool_call"

    def __repr__(self) -> str:
        return f"ToolCall(name={self.name!r}, args={self.args!r}, id={self.id!r})"

    def dict(self) -> dict[str, Any]:
        return {"name": self.name, "args": self.args, "id": self.id, "type": self.type}


# Type alias matching langchain-core
AnyMessage = Union[HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage]


def _message_from_tuple(role: str, content: str) -> BaseMessage:
    """Convert a (role, content) tuple to a message object."""
    role_map: dict[str, type[BaseMessage]] = {
        "human": HumanMessage,
        "user": HumanMessage,
        "ai": AIMessage,
        "assistant": AIMessage,
        "system": SystemMessage,
    }
    cls = role_map.get(role)
    if cls is None:
        raise ValueError(f"Unknown message role: {role!r}. Expected one of {list(role_map.keys())}")
    return cls(content=content)
