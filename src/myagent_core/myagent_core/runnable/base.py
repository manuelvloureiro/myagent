"""Base Runnable abstraction.

``Runnable[Input, Output]`` is the universal execution interface shared by every
component in the myagent stack.  It mirrors the ``langchain_core.runnables.Runnable``
contract so that any class implementing it can be used interchangeably.

Execution modes
---------------
Every Runnable exposes four execution patterns:

* **invoke / ainvoke** – single input -> single output (the primary path).
* **stream / astream** – single input -> iterator of output chunks.
  The base implementation yields a single chunk (the full invoke result).
  Subclasses like ``Pregel`` override this to yield per-superstep snapshots.
* **batch / abatch** – multiple inputs -> list of outputs, executed
  concurrently via ``ThreadPoolExecutor`` (sync) or ``asyncio.gather`` (async).

Default async behaviour
-----------------------
``ainvoke`` delegates to ``invoke`` inside ``run_in_executor`` so that
subclasses only *need* to implement ``invoke``.  Subclasses with native
async I/O (e.g. Pregel) override ``ainvoke`` / ``astream`` directly.

Config propagation
------------------
Every method accepts an optional ``RunnableConfig`` that carries tags,
metadata, recursion limits, and the ``configurable`` dict used for
thread IDs and checkpoint routing.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    AsyncIterator,
    Generic,
    Iterator,
    Optional,
    Sequence,
    TypeVar,
)

from myagent_core.runnable.config import RunnableConfig

Input = TypeVar("Input")
Output = TypeVar("Output")


class Runnable(ABC, Generic[Input, Output]):
    """Abstract base for all executable components.

    Subclasses must implement :meth:`invoke`.  All other methods have
    sensible defaults built on top of it.

    Type Parameters
    ---------------
    Input
        The type accepted by :meth:`invoke`.
    Output
        The type returned by :meth:`invoke`.
    """

    name: Optional[str] = None

    @abstractmethod
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        """Execute on a single input and return the result.

        This is the only method subclasses *must* implement.

        Args:
            input: The input value.
            config: Optional execution configuration.

        Returns:
            The computed output.
        """
        ...

    async def ainvoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        """Async version of :meth:`invoke`.

        Default implementation runs ``invoke`` in a thread executor.
        Override for native async behaviour.
        """
        return await asyncio.get_event_loop().run_in_executor(None, self.invoke, input, config)

    def stream(self, input: Input, config: Optional[RunnableConfig] = None) -> Iterator[Output]:
        """Yield output chunks for a single input.

        The base implementation yields the full ``invoke`` result as a
        single chunk.  ``Pregel`` overrides this to yield state after
        each superstep.
        """
        yield self.invoke(input, config)

    async def astream(self, input: Input, config: Optional[RunnableConfig] = None) -> AsyncIterator[Output]:
        """Async version of :meth:`stream`."""
        yield await self.ainvoke(input, config)

    def batch(
        self,
        inputs: Sequence[Input],
        config: Optional[RunnableConfig | Sequence[RunnableConfig]] = None,
    ) -> list[Output]:
        """Execute on multiple inputs concurrently using threads.

        Args:
            inputs: Sequence of input values.
            config: A single config applied to all inputs, or one config per input.

        Returns:
            List of outputs, one per input.
        """
        configs = [config] * len(inputs) if not isinstance(config, list) else config
        with ThreadPoolExecutor() as executor:
            return list(executor.map(self.invoke, inputs, configs))

    async def abatch(
        self,
        inputs: Sequence[Input],
        config: Optional[RunnableConfig | Sequence[RunnableConfig]] = None,
    ) -> list[Output]:
        """Async version of :meth:`batch` using ``asyncio.gather``."""
        configs: Sequence[Optional[RunnableConfig]]
        if isinstance(config, list):
            configs = config
        elif isinstance(config, dict) or config is None:
            configs = [config] * len(inputs)
        else:
            configs = list(config)
        return await asyncio.gather(*(self.ainvoke(inp, c) for inp, c in zip(inputs, configs)))

    def pipe(self, *others: Any) -> Runnable:
        """Chain this Runnable with others in sequence.

        Equivalent to ``self | other1 | other2 | ...``.
        """
        from myagent_core.runnable.sequence import RunnableSequence
        from myagent_core.runnable.utils import coerce_to_runnable

        steps = [self] + [coerce_to_runnable(o) for o in others]
        return RunnableSequence(steps_list=steps)

    def __or__(self, other: Any) -> Runnable:
        """Pipe operator: ``self | other``."""
        from myagent_core.runnable.sequence import RunnableSequence
        from myagent_core.runnable.utils import coerce_to_runnable

        other = coerce_to_runnable(other)
        # Flatten nested sequences
        left = self.steps if isinstance(self, RunnableSequence) else [self]
        right = other.steps if isinstance(other, RunnableSequence) else [other]
        return RunnableSequence(steps_list=left + right)

    def __ror__(self, other: Any) -> Runnable:
        """Reverse pipe operator: ``other | self``."""
        from myagent_core.runnable.sequence import RunnableSequence
        from myagent_core.runnable.utils import coerce_to_runnable

        other = coerce_to_runnable(other)
        left = other.steps if isinstance(other, RunnableSequence) else [other]
        right = self.steps if isinstance(self, RunnableSequence) else [self]
        return RunnableSequence(steps_list=left + right)
