"""RunnableLambda - wraps plain functions as Runnables.

This is the primary way user-defined functions enter the Runnable
ecosystem.  ``StateGraph.add_node`` wraps node functions internally,
but ``RunnableLambda`` can also be used standalone::

    r = RunnableLambda(lambda x: x * 2)
    r.invoke(5)  # -> 10

Config injection
----------------
If the wrapped function accepts **two positional parameters**, the second
is automatically filled with the ``RunnableConfig``::

    def my_fn(state, config):
        thread = config["configurable"]["thread_id"]
        ...

    r = RunnableLambda(my_fn)
    r.invoke(state, {"configurable": {"thread_id": "t1"}})

Async support
-------------
Pass an ``afunc`` to provide a native async implementation.  If omitted,
``ainvoke`` falls back to running the sync ``func`` in a thread executor::

    async def async_fn(x):
        return x * 2

    r = RunnableLambda(lambda x: x * 2, afunc=async_fn)
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, AsyncIterator, Callable, Iterator, Optional

from myagent_core.runnable.base import Runnable
from myagent_core.runnable.config import RunnableConfig, ensure_config


class RunnableLambda(Runnable[Any, Any]):
    """Wraps a plain function (sync or async) as a Runnable.

    Args:
        func: The synchronous function to wrap.
        afunc: Optional native async function.  Used by ``ainvoke`` / ``astream``
            when provided; otherwise the sync ``func`` is run in an executor.
        name: Display name.  Defaults to ``func.__name__``.
    """

    def __init__(self, func: Callable, afunc: Optional[Callable] = None, name: Optional[str] = None):
        self.func = func
        self.afunc = afunc
        self.name = name or getattr(func, "__name__", "RunnableLambda")
        self._accepts_config = _accepts_config(func)
        if afunc:
            self._async_accepts_config = _accepts_config(afunc)
        else:
            self._async_accepts_config = False

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        config = ensure_config(config)
        if self._accepts_config:
            return self.func(input, config)
        return self.func(input)

    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        config = ensure_config(config)
        if self.afunc is not None:
            if self._async_accepts_config:
                return await self.afunc(input, config)
            return await self.afunc(input)
        return await asyncio.get_event_loop().run_in_executor(None, self.invoke, input, config)

    def stream(self, input: Any, config: Optional[RunnableConfig] = None) -> Iterator[Any]:
        yield self.invoke(input, config)

    async def astream(self, input: Any, config: Optional[RunnableConfig] = None) -> AsyncIterator[Any]:
        yield await self.ainvoke(input, config)


def _accepts_config(func: Callable) -> bool:
    """Return True if *func* has ≥2 positional parameters (input + config)."""
    try:
        sig = inspect.signature(func)
        params = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        return len(params) >= 2
    except (ValueError, TypeError):
        return False
