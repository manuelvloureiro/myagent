"""RunnableSequence - chains Runnables in series.

Created automatically by the ``|`` pipe operator::

    chain = prompt | model | parser
    result = chain.invoke({"question": "hello"})

Each step's output becomes the next step's input.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, Optional

from myagent_core.runnable.base import Runnable
from myagent_core.runnable.config import RunnableConfig, ensure_config


class RunnableSequence(Runnable[Any, Any]):
    """Executes a sequence of Runnables, piping output -> input.

    Args:
        steps: Ordered list of Runnables to execute.
    """

    def __init__(self, *steps: Runnable, steps_list: Optional[list[Runnable]] = None):
        if steps_list is not None:
            self.steps: list[Runnable] = steps_list
        else:
            self.steps = list(steps)
        self.name = " | ".join(s.name or s.__class__.__name__ for s in self.steps)

    @property
    def first(self) -> Runnable:
        return self.steps[0]

    @property
    def middle(self) -> list[Runnable]:
        return self.steps[1:-1]

    @property
    def last(self) -> Runnable:
        return self.steps[-1]

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        config = ensure_config(config)
        result = input
        for step in self.steps:
            result = step.invoke(result, config)
        return result

    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        config = ensure_config(config)
        result = input
        for step in self.steps:
            result = await step.ainvoke(result, config)
        return result

    def stream(self, input: Any, config: Optional[RunnableConfig] = None) -> Iterator[Any]:
        config = ensure_config(config)
        # Run all steps except last to get intermediate result
        result = input
        for step in self.steps[:-1]:
            result = step.invoke(result, config)
        # Stream the last step
        yield from self.steps[-1].stream(result, config)

    async def astream(self, input: Any, config: Optional[RunnableConfig] = None) -> AsyncIterator[Any]:
        config = ensure_config(config)
        result = input
        for step in self.steps[:-1]:
            result = await step.ainvoke(result, config)
        async for chunk in self.steps[-1].astream(result, config):
            yield chunk

    def __or__(self, other: Any) -> RunnableSequence:
        from myagent_core.runnable.utils import coerce_to_runnable

        other = coerce_to_runnable(other)
        if isinstance(other, RunnableSequence):
            return RunnableSequence(steps_list=self.steps + other.steps)
        return RunnableSequence(steps_list=self.steps + [other])

    def __ror__(self, other: Any) -> RunnableSequence:
        from myagent_core.runnable.utils import coerce_to_runnable

        other = coerce_to_runnable(other)
        if isinstance(other, RunnableSequence):
            return RunnableSequence(steps_list=other.steps + self.steps)
        return RunnableSequence(steps_list=[other] + self.steps)
