"""Prompt templates compatible with langchain_core.prompts.

Provides ChatPromptTemplate and PromptTemplate that produce messages
suitable for chat model input::

    from myagent_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a {role}."),
        ("human", "{question}"),
    ])
    messages = prompt.invoke({"role": "helpful assistant", "question": "Hi!"})
"""

from __future__ import annotations

from string import Formatter
from typing import Any, Optional, Sequence, Union

from myagent_core.messages import BaseMessage, _message_from_tuple
from myagent_core.runnable.base import Runnable
from myagent_core.runnable.config import RunnableConfig


class PromptTemplate(Runnable[dict, str]):
    """String prompt template with variable substitution.

    Uses Python's ``str.format_map`` for variable interpolation::

        prompt = PromptTemplate(template="Hello, {name}!")
        prompt.invoke({"name": "world"})  # "Hello, world!"

    Args:
        template: Format string with ``{variable}`` placeholders.
        input_variables: Optional explicit list of variable names.
            If not provided, extracted automatically from the template.
    """

    def __init__(self, template: str, *, input_variables: Optional[list[str]] = None):
        self.template = template
        self.input_variables = input_variables or [
            fname for _, fname, _, _ in Formatter().parse(template) if fname is not None
        ]
        self.name = "PromptTemplate"

    @classmethod
    def from_template(cls, template: str) -> PromptTemplate:
        """Create a PromptTemplate from a format string."""
        return cls(template=template)

    def format(self, **kwargs: Any) -> str:
        """Format the template with the given variables."""
        return self.template.format_map(kwargs)

    def invoke(self, input: dict, config: Optional[RunnableConfig] = None) -> str:
        return self.format(**input)

    async def ainvoke(self, input: dict, config: Optional[RunnableConfig] = None) -> str:
        return self.format(**input)


class ChatPromptTemplate(Runnable[dict, list[BaseMessage]]):
    """Chat prompt template that produces a list of messages.

    Each message template is either:
    - A ``(role, template_string)`` tuple
    - A ``BaseMessage`` instance (used as-is, no formatting)

    Usage::

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a {role}."),
            ("human", "{question}"),
        ])
        messages = prompt.invoke({"role": "assistant", "question": "Hi!"})
    """

    def __init__(self, messages: Sequence[Union[tuple[str, str], BaseMessage]]):
        self.message_templates: list[Union[tuple[str, str], BaseMessage]] = list(messages)
        self.input_variables = self._extract_variables()
        self.name = "ChatPromptTemplate"

    def _extract_variables(self) -> list[str]:
        variables: list[str] = []
        for tmpl in self.message_templates:
            if isinstance(tmpl, tuple):
                _, content_template = tmpl
                for _, fname, _, _ in Formatter().parse(content_template):
                    if fname is not None and fname not in variables:
                        variables.append(fname)
        return variables

    @classmethod
    def from_messages(cls, messages: Sequence[Union[tuple[str, str], BaseMessage]]) -> ChatPromptTemplate:
        """Create a ChatPromptTemplate from a list of message specs."""
        return cls(messages=messages)

    @classmethod
    def from_template(cls, template: str) -> ChatPromptTemplate:
        """Create a single human-message prompt from a template string."""
        return cls(messages=[("human", template)])

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Format all message templates with the given variables."""
        result: list[BaseMessage] = []
        for tmpl in self.message_templates:
            if isinstance(tmpl, BaseMessage):
                result.append(tmpl)
            else:
                role, content_template = tmpl
                content = content_template.format_map(kwargs)
                result.append(_message_from_tuple(role, content))
        return result

    def invoke(self, input: dict, config: Optional[RunnableConfig] = None) -> list[BaseMessage]:
        return self.format_messages(**input)

    async def ainvoke(self, input: dict, config: Optional[RunnableConfig] = None) -> list[BaseMessage]:
        return self.format_messages(**input)
