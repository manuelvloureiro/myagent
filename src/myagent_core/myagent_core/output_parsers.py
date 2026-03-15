"""Output parsers compatible with langchain_core.output_parsers.

Provides parsers that extract structured data from model output::

    from myagent_core.output_parsers import StrOutputParser

    chain = prompt | model | StrOutputParser()
    result = chain.invoke({"question": "Hi"})  # returns a plain string
"""

from __future__ import annotations

import json
from typing import Any, Optional

from myagent_core.messages import BaseMessage
from myagent_core.runnable.base import Runnable
from myagent_core.runnable.config import RunnableConfig


class BaseOutputParser(Runnable[Any, Any]):
    """Base class for output parsers."""

    def __init__(self) -> None:
        self.name = self.__class__.__name__

    def _get_text(self, input: Any) -> str:
        """Extract text content from various input types."""
        if isinstance(input, str):
            return input
        if isinstance(input, BaseMessage):
            if isinstance(input.content, str):
                return input.content
            return str(input.content)
        if isinstance(input, dict) and "content" in input:
            return str(input["content"])
        return str(input)


class StrOutputParser(BaseOutputParser):
    """Extracts string content from an AIMessage or passes strings through.

    Commonly used as the final step in a chain::

        chain = prompt | model | StrOutputParser()
    """

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        return self._get_text(input)

    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        return self._get_text(input)


class JsonOutputParser(BaseOutputParser):
    """Parses JSON from model output.

    Extracts JSON from the text content, handling optional markdown
    code fences::

        parser = JsonOutputParser()
        parser.invoke(AIMessage(content='```json\\n{"key": "value"}\\n```'))
        # {"key": "value"}
    """

    def _parse(self, text: str) -> Any:
        text = text.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```) and last line (```)
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return json.loads(text)

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        return self._parse(self._get_text(input))

    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        return self._parse(self._get_text(input))


class PydanticOutputParser(BaseOutputParser):
    """Parses JSON output and validates it against a Pydantic model.

    Args:
        pydantic_object: The Pydantic model class to validate against.

    Usage::

        from pydantic import BaseModel

        class Answer(BaseModel):
            text: str
            confidence: float

        parser = PydanticOutputParser(pydantic_object=Answer)
        parser.invoke('{"text": "hello", "confidence": 0.95}')
        # Answer(text='hello', confidence=0.95)
    """

    def __init__(self, pydantic_object: type) -> None:
        super().__init__()
        self.pydantic_object = pydantic_object

    def _parse(self, text: str) -> Any:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        data = json.loads(text)
        return self.pydantic_object(**data)

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        return self._parse(self._get_text(input))

    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        return self._parse(self._get_text(input))
