"""RunnableParallel - runs multiple Runnables concurrently, returns a dict.

Usage::

    from myagent_core import RunnableParallel, RunnablePassthrough

    chain = RunnableParallel(
        original=RunnablePassthrough(),
        upper=lambda x: x.upper(),
    )
    chain.invoke("hello")  # {"original": "hello", "upper": "HELLO"}

Also created implicitly when piping a dict::

    chain = some_runnable | {"key1": runnable1, "key2": runnable2}
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator, Iterator, Optional

from myagent_core.runnable.base import Runnable
from myagent_core.runnable.config import RunnableConfig, ensure_config


class RunnableParallel(Runnable[Any, dict]):
    """Runs multiple Runnables in parallel and returns a dict of results.

    Accepts either keyword arguments or a single dict mapping keys to Runnables.
    Callables are automatically wrapped in RunnableLambda.
    """

    def __init__(self, steps: Optional[dict[str, Any]] = None, **kwargs: Any):
        from myagent_core.runnable.utils import coerce_to_runnable

        raw = steps or {}
        raw.update(kwargs)
        self.steps: dict[str, Runnable] = {k: coerce_to_runnable(v) for k, v in raw.items()}
        self.name = "RunnableParallel"

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> dict:
        config = ensure_config(config)
        if len(self.steps) == 1:
            key, runnable = next(iter(self.steps.items()))
            return {key: runnable.invoke(input, config)}
        with ThreadPoolExecutor() as executor:
            futures = {key: executor.submit(runnable.invoke, input, config) for key, runnable in self.steps.items()}
            return {key: future.result() for key, future in futures.items()}

    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None) -> dict:
        config = ensure_config(config)
        results = await asyncio.gather(*(runnable.ainvoke(input, config) for runnable in self.steps.values()))
        return dict(zip(self.steps.keys(), results))

    def stream(self, input: Any, config: Optional[RunnableConfig] = None) -> Iterator[dict]:
        yield self.invoke(input, config)

    async def astream(self, input: Any, config: Optional[RunnableConfig] = None) -> AsyncIterator[dict]:
        yield await self.ainvoke(input, config)
