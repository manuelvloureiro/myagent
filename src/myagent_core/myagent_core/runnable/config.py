"""Execution configuration for Runnables.

``RunnableConfig`` is a TypedDict that flows through every ``invoke``,
``stream``, and ``batch`` call.  It carries:

* **tags / metadata** – arbitrary labels for tracing and filtering.
* **callbacks** – reserved for LangChain callback integration (unused here).
* **run_id** – unique identifier for this execution.
* **configurable** – the main extension point.  Keys like ``thread_id``
  and ``checkpoint_id`` are used by the checkpoint system to route state.
* **recursion_limit** – maximum supersteps before ``GraphRecursionError``
  is raised (default: 25).
* **max_concurrency** – optional cap on parallel node execution.

Helper functions
----------------
* ``ensure_config(cfg)`` – fills in defaults for any missing keys.
* ``merge_configs(*cfgs)`` – combines multiple configs; tags and metadata
  are merged additively, everything else is last-writer-wins.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional, TypedDict


class RunnableConfig(TypedDict, total=False):
    """Execution configuration passed through the Runnable call chain.

    All fields are optional (``total=False``).  Use :func:`ensure_config`
    to obtain a copy with defaults filled in.
    """

    tags: list[str]
    metadata: dict[str, Any]
    callbacks: Any
    run_name: str
    run_id: str
    configurable: dict[str, Any]
    recursion_limit: int
    max_concurrency: Optional[int]


DEFAULT_RECURSION_LIMIT = 25


def ensure_config(config: Optional[RunnableConfig] = None) -> RunnableConfig:
    """Return a ``RunnableConfig`` with all defaults filled in.

    Safe to call with ``None`` – returns a fresh config with a new
    ``run_id`` and the default recursion limit.
    """
    if config is None:
        config = {}
    return {
        "tags": config.get("tags", []),
        "metadata": config.get("metadata", {}),
        "callbacks": config.get("callbacks"),
        "run_name": config.get("run_name", ""),
        "run_id": config.get("run_id", str(uuid.uuid4())),
        "configurable": config.get("configurable", {}),
        "recursion_limit": config.get("recursion_limit", DEFAULT_RECURSION_LIMIT),
        "max_concurrency": config.get("max_concurrency"),
    }


def merge_configs(*configs: Optional[RunnableConfig]) -> RunnableConfig:
    """Merge multiple configs into one.

    * **tags** are concatenated.
    * **metadata** and **configurable** are shallow-merged (later wins per key).
    * All other fields use last-writer-wins.
    """
    base = ensure_config()
    for config in configs:
        if config is None:
            continue
        if "tags" in config:
            base["tags"] = [*base.get("tags", []), *config["tags"]]
        if "metadata" in config:
            base["metadata"] = {**base.get("metadata", {}), **config["metadata"]}
        if "callbacks" in config:
            base["callbacks"] = config["callbacks"]
        if "run_name" in config:
            base["run_name"] = config["run_name"]
        if "run_id" in config:
            base["run_id"] = config["run_id"]
        if "configurable" in config:
            base["configurable"] = {
                **base.get("configurable", {}),
                **config["configurable"],
            }
        if "recursion_limit" in config:
            base["recursion_limit"] = config["recursion_limit"]
        if "max_concurrency" in config:
            base["max_concurrency"] = config["max_concurrency"]
    return base
