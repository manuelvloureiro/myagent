"""Base chat model interface compatible with langchain_core.language_models.

Provides ``BaseChatModel`` - the abstract Runnable that takes messages and
returns an ``AIMessage``. Also provides ``FakeListChatModel`` for testing::

    from myagent_core.chat_models import FakeListChatModel

    model = FakeListChatModel(responses=["Hello!", "Goodbye!"])
    model.invoke([HumanMessage(content="Hi")])  # AIMessage(content="Hello!")
    model.invoke([HumanMessage(content="Bye")])  # AIMessage(content="Goodbye!")
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from copy import deepcopy
from typing import Any, AsyncIterator, Iterator, Optional, Sequence

from myagent_core.messages import AIMessage, BaseMessage, HumanMessage, ToolCall
from myagent_core.runnable.base import Runnable
from myagent_core.runnable.config import RunnableConfig, ensure_config


class BaseChatModel(Runnable[Any, AIMessage]):
    """Abstract base class for chat models.

    Subclasses must implement :meth:`_generate`.  The ``invoke`` / ``ainvoke``
    methods handle input coercion (str -> HumanMessage) and config propagation.

    Compatible with ``langchain_core.language_models.BaseChatModel``.
    """

    def __init__(self, **kwargs: Any):
        self.name = kwargs.get("name", self.__class__.__name__)
        self._bound_tools: list[Any] = []

    @abstractmethod
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Generate a response from the given messages.

        Args:
            messages: Input messages.
            stop: Optional stop sequences.

        Returns:
            The model's response as an AIMessage.
        """
        ...

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Async version of _generate. Default runs sync in executor."""
        return await asyncio.get_event_loop().run_in_executor(None, self._generate, messages, stop)

    def _coerce_input(self, input: Any) -> list[BaseMessage]:
        """Coerce input to a list of messages."""
        if isinstance(input, str):
            return [HumanMessage(content=input)]
        if isinstance(input, BaseMessage):
            return [input]
        if isinstance(input, list):
            result: list[BaseMessage] = []
            for item in input:
                if isinstance(item, BaseMessage):
                    result.append(item)
                elif isinstance(item, str):
                    result.append(HumanMessage(content=item))
                elif isinstance(item, dict):
                    result.append(HumanMessage(content=str(item.get("content", item))))
                else:
                    result.append(HumanMessage(content=str(item)))
            return result
        return [HumanMessage(content=str(input))]

    def invoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        config = ensure_config(config)
        messages = self._coerce_input(input)
        return self._generate(messages, stop=stop, **kwargs)

    async def ainvoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        config = ensure_config(config)
        messages = self._coerce_input(input)
        return await self._agenerate(messages, stop=stop, **kwargs)

    def stream(self, input: Any, config: Optional[RunnableConfig] = None) -> Iterator[AIMessage]:
        yield self.invoke(input, config)

    async def astream(self, input: Any, config: Optional[RunnableConfig] = None) -> AsyncIterator[AIMessage]:
        yield await self.ainvoke(input, config)

    def bind_tools(self, tools: Sequence[Any]) -> BaseChatModel:
        """Return a copy of this model with tools bound.

        Args:
            tools: List of tool definitions (BaseTool instances or dicts).

        Returns:
            A new model instance with tools available for calling.
        """
        clone = deepcopy(self)
        clone._bound_tools = list(tools)
        return clone


class FakeListChatModel(BaseChatModel):
    """Chat model that cycles through canned responses. For testing.

    Args:
        responses: List of response strings to cycle through.
    """

    def __init__(self, responses: list[str], **kwargs: Any):
        super().__init__(**kwargs)
        self.responses = responses
        self._call_count = 0

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        response = self.responses[self._call_count % len(self.responses)]
        self._call_count += 1
        if stop:
            for s in stop:
                if s in response:
                    response = response[: response.index(s)]
        return AIMessage(content=response)


class FakeToolCallingModel(BaseChatModel):
    """Chat model that returns pre-configured tool calls. For testing.

    Args:
        tool_calls: List of ToolCall objects to return.
        content: Optional text content alongside tool calls.
    """

    def __init__(self, tool_calls: list[ToolCall], content: str = "", **kwargs: Any):
        super().__init__(**kwargs)
        self._tool_calls = tool_calls
        self._content = content

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        return AIMessage(content=self._content, tool_calls=self._tool_calls)
