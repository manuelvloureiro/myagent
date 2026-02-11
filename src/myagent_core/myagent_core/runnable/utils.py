"""Utilities for coercing values into Runnables."""

from __future__ import annotations

from typing import Any

from myagent_core.runnable.base import Runnable


def coerce_to_runnable(thing: Any) -> Runnable:
    """Convert a value into a Runnable.

    - Runnable -> returned as-is
    - callable -> wrapped in RunnableLambda
    - dict -> wrapped in RunnableParallel

    Raises:
        TypeError: If the value cannot be converted.
    """
    if isinstance(thing, Runnable):
        return thing
    if isinstance(thing, dict):
        from myagent_core.runnable.parallel import RunnableParallel

        return RunnableParallel(steps=thing)
    if callable(thing):
        from myagent_core.runnable.lambda_ import RunnableLambda

        return RunnableLambda(thing)
    raise TypeError(f"Cannot coerce {type(thing).__name__} to Runnable")
