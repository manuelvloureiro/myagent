"""RunnablePassthrough - passes input through unchanged.

Usage::

    from myagent_core import RunnablePassthrough

    # Simple passthrough
    chain = RunnablePassthrough()
    chain.invoke({"key": "value"})  # {"key": "value"}

    # Assign: add computed keys to the input dict
    chain = RunnablePassthrough.assign(upper=lambda x: x["text"].upper())
    chain.invoke({"text": "hello"})  # {"text": "hello", "upper": "HELLO"}
"""

from __future__ import annotations

from typing import Any, Optional

from myagent_core.runnable.base import Runnable
from myagent_core.runnable.config import RunnableConfig, ensure_config


class RunnablePassthrough(Runnable[Any, Any]):
    """Identity Runnable that passes input through unchanged.

    Use :meth:`assign` to create a variant that merges computed keys
    into the input dict.
    """

    def __init__(self) -> None:
        self.name = "RunnablePassthrough"

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        return input

    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        return input

    @staticmethod
    def assign(**kwargs: Any) -> RunnableAssign:
        """Create a Runnable that passes input through and adds computed keys.

        Each kwarg value should be a Runnable or callable that takes the input
        and returns the value for that key.

        Returns:
            A :class:`RunnableAssign` that merges original input with computed keys.
        """
        return RunnableAssign(**kwargs)


class RunnableAssign(Runnable[dict, dict]):
    """Passes input dict through and merges in computed keys."""

    def __init__(self, **kwargs: Any) -> None:
        from myagent_core.runnable.utils import coerce_to_runnable

        self.mapping: dict[str, Runnable] = {k: coerce_to_runnable(v) for k, v in kwargs.items()}
        self.name = "RunnableAssign"

    def invoke(self, input: dict, config: Optional[RunnableConfig] = None) -> dict:
        config = ensure_config(config)
        result = dict(input)
        for key, runnable in self.mapping.items():
            result[key] = runnable.invoke(input, config)
        return result

    async def ainvoke(self, input: dict, config: Optional[RunnableConfig] = None) -> dict:
        import asyncio

        config = ensure_config(config)
        result = dict(input)
        computed = await asyncio.gather(*(r.ainvoke(input, config) for r in self.mapping.values()))
        for key, value in zip(self.mapping.keys(), computed):
            result[key] = value
        return result
