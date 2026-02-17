"""RunnableBranch - conditional routing of inputs.

Usage::

    from myagent_core import RunnableBranch, RunnableLambda

    branch = RunnableBranch(
        (lambda x: x["type"] == "question", handle_question),
        (lambda x: x["type"] == "statement", handle_statement),
        handle_default,  # fallback
    )
    branch.invoke({"type": "question", "text": "What?"})
"""

from __future__ import annotations

from typing import Any, Optional, Union

from myagent_core.runnable.base import Runnable
from myagent_core.runnable.config import RunnableConfig, ensure_config


class RunnableBranch(Runnable[Any, Any]):
    """Routes input to different Runnables based on conditions.

    Args:
        *branches: Alternating (condition, runnable) tuples followed by a
            default runnable. Conditions are callables that take the input
            and return a bool.
    """

    def __init__(self, *branches: Union[tuple[Any, Any], Any]):
        from myagent_core.runnable.utils import coerce_to_runnable

        if len(branches) < 2:
            raise ValueError("RunnableBranch requires at least one (condition, runnable) pair and a default.")

        self.branches: list[tuple[Any, Runnable]] = []
        for branch in branches[:-1]:
            if not isinstance(branch, tuple) or len(branch) != 2:
                raise ValueError(f"Expected (condition, runnable) tuple, got {type(branch)}")
            condition, runnable = branch
            self.branches.append((condition, coerce_to_runnable(runnable)))

        self.default = coerce_to_runnable(branches[-1])
        self.name = "RunnableBranch"

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        config = ensure_config(config)
        for condition, runnable in self.branches:
            if condition(input):
                return runnable.invoke(input, config)
        return self.default.invoke(input, config)

    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        config = ensure_config(config)
        for condition, runnable in self.branches:
            if condition(input):
                return await runnable.ainvoke(input, config)
        return await self.default.ainvoke(input, config)
