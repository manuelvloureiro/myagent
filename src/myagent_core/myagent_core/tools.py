"""Tool abstractions compatible with langchain_core.tools.

Provides ``BaseTool`` and the ``@tool`` decorator for wrapping functions
as tools that can be bound to chat models::

    from myagent_core.tools import tool

    @tool
    def search(query: str) -> str:
        \"\"\"Search the web for information.\"\"\"
        return f"Results for: {query}"

    search.invoke({"query": "LangChain"})
"""

from __future__ import annotations

import inspect
from abc import abstractmethod
from typing import Any, Callable, Optional, get_type_hints

from myagent_core.runnable.base import Runnable
from myagent_core.runnable.config import RunnableConfig


class BaseTool(Runnable[Any, Any]):
    """Abstract base class for tools.

    Compatible with ``langchain_core.tools.BaseTool``.

    Attributes:
        name: The tool name (used in tool_calls).
        description: What this tool does (sent to the model).
        args_schema: Dict describing expected arguments.
    """

    name: Optional[str] = ""
    description: str = ""
    args_schema: dict[str, Any] = {}

    @abstractmethod
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool (sync)."""
        ...

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool (async). Default delegates to sync."""
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(None, lambda: self._run(*args, **kwargs))

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        if isinstance(input, dict):
            return self._run(**input)
        if isinstance(input, str):
            return self._run(input)
        return self._run(input)

    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        if isinstance(input, dict):
            return await self._arun(**input)
        if isinstance(input, str):
            return await self._arun(input)
        return await self._arun(input)

    def to_dict(self) -> dict[str, Any]:
        """Serialize tool definition for model binding."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema,
            },
        }


class StructuredTool(BaseTool):
    """A tool created from a function.

    Typically created via the :func:`tool` decorator rather than directly.
    """

    def __init__(
        self,
        func: Callable,
        name: str,
        description: str,
        args_schema: dict[str, Any],
        afunc: Optional[Callable] = None,
    ):
        self.func = func
        self.afunc = afunc
        self.name = name
        self.description = description
        self.args_schema = args_schema

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        if self.afunc is not None:
            return await self.afunc(*args, **kwargs)
        return await super()._arun(*args, **kwargs)


def _build_args_schema(func: Callable) -> dict[str, Any]:
    """Extract a JSON-schema-like args description from function signature."""
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    properties: dict[str, Any] = {}
    required: list[str] = []
    for param_name, param in sig.parameters.items():
        prop: dict[str, Any] = {}
        hint = hints.get(param_name)
        if hint in type_map:
            prop["type"] = type_map[hint]
        else:
            prop["type"] = "string"
        properties[param_name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def tool(func: Optional[Callable] = None, *, name: Optional[str] = None, description: Optional[str] = None) -> Any:
    """Decorator to create a :class:`StructuredTool` from a function.

    Can be used with or without arguments::

        @tool
        def search(query: str) -> str:
            \"\"\"Search the web.\"\"\"
            return f"Results for: {query}"

        @tool(name="my_search", description="Custom search tool")
        def search(query: str) -> str:
            return f"Results for: {query}"
    """

    def decorator(f: Callable) -> StructuredTool:
        tool_name = name or f.__name__
        tool_desc = description or (f.__doc__ or "").strip()
        args_schema = _build_args_schema(f)
        return StructuredTool(
            func=f,
            name=tool_name,
            description=tool_desc,
            args_schema=args_schema,
        )

    if func is not None:
        return decorator(func)
    return decorator
