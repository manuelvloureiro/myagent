"""Pregel - the superstep execution engine.

``Pregel`` is the runtime that powers ``CompiledStateGraph``.  It implements
the LangGraph execution model: a loop of **supersteps** where, in each step,
eligible nodes execute concurrently, read a consistent state snapshot, and
write updates back through typed channels.

Superstep algorithm
-------------------
::

    invoke(input, config):
        1. Load checkpoint (if checkpointer + thread_id present)
        2. Initialise channels from checkpoint (or empty)
        3. Write *input* to channels
        4. Loop (step < recursion_limit):
           a. PLAN  - find nodes whose incoming edges are satisfied
           b. If no nodes to fire -> break
           c. Check interrupt_before -> checkpoint & break if matched
           d. SNAPSHOT - read full state (all channels) for nodes
           e. EXECUTE - run planned nodes concurrently
              • Sync: ThreadPoolExecutor
              • Async: asyncio.gather
              • Per-node RetryPolicy applied
           f. UPDATE - apply each node's result dict to channels
              • LastValue: overwrite
              • BinaryOperatorAggregate: fold via reducer
           g. STREAM - yield state (values mode) or updates (updates mode)
           h. Check interrupt_after -> checkpoint & break if matched
           i. Clear ephemeral channels
           j. If END triggered -> break
           k. Checkpoint
        5. Return final state

State visibility
----------------
Nodes always see the **full** state (all channels), even when
``output_schema`` is set.  The output schema only filters what
``invoke`` / ``stream`` return to the caller and what ``get_state``
exposes.  This separation is handled by ``_read_full_state`` (for
nodes and conditional-edge routers) vs ``_read_output`` (for
external consumers).

Checkpointing
-------------
When a ``BaseCheckpointSaver`` is provided and the config contains a
``thread_id``, Pregel saves a checkpoint after each superstep and at the
end of execution.  On subsequent invocations with the same ``thread_id``,
channels are restored from the latest checkpoint, enabling multi-turn
conversations.

Interrupt support
-----------------
``interrupt_before`` and ``interrupt_after`` accept lists of node names.
When a planned node matches ``interrupt_before``, the engine checkpoints
and exits *before* that node runs.  ``interrupt_after`` does the same
*after* the node completes.  Resume-from-interrupt requires re-invoking
with the same ``thread_id``.
"""

from __future__ import annotations

import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Iterator, Optional

from myagent_core.runnable.base import Runnable
from myagent_core.runnable.config import RunnableConfig, ensure_config

from myagent.channels.base import BaseChannel
from myagent.channels.ephemeral import EphemeralValue
from myagent.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
)
from myagent.checkpoint.id import create_checkpoint_id
from myagent.constants import END, START
from myagent.errors import EmptyChannelError, GraphRecursionError
from myagent.pregel.read import PregelNode
from myagent.pregel.retry import arun_with_retry, run_with_retry
from myagent.types import StreamMode


class Pregel(Runnable[dict, dict]):
    """Superstep-based graph execution engine.

    This is the core runtime behind ``CompiledStateGraph``.  It is not
    normally instantiated directly - use ``StateGraph.compile()`` instead.

    Args:
        nodes: Mapping of node name -> ``PregelNode``.
        channels: Mapping of channel name -> ``BaseChannel`` instance.
        input_channels: Channel names to write input values into.
        output_channels: Channel names to include in output / ``get_state``.
        edges: Direct edges as ``{source: [target, ...]}``.
        conditional_edges: Conditional edges as
            ``{source: [(router_fn, path_map | None), ...]}``.
        checkpointer: Optional checkpoint saver for state persistence.
        interrupt_before: Node names to pause *before* executing.
        interrupt_after: Node names to pause *after* executing.
        stream_mode: Default stream mode (``"values"`` or ``"updates"``).
    """

    def __init__(
        self,
        *,
        nodes: dict[str, PregelNode],
        channels: dict[str, BaseChannel],
        input_channels: list[str],
        output_channels: list[str],
        edges: dict[str, list[str]],
        conditional_edges: dict[str, list[tuple[Any, dict[str, str] | None]]],
        checkpointer: Optional[BaseCheckpointSaver] = None,
        interrupt_before: Optional[list[str]] = None,
        interrupt_after: Optional[list[str]] = None,
        stream_mode: StreamMode = "values",
    ):
        self.nodes = nodes
        self.channels = channels
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.edges = edges
        self.conditional_edges = conditional_edges
        self.checkpointer = checkpointer
        self.interrupt_before = interrupt_before or []
        self.interrupt_after = interrupt_after or []
        self._stream_mode = stream_mode

    # ------------------------------------------------------------------
    # Public execution API
    # ------------------------------------------------------------------

    def invoke(
        self,
        input: dict,
        config: Optional[RunnableConfig] = None,
        *,
        stream_mode: Optional[StreamMode] = None,
    ) -> dict:
        """Run the graph to completion and return the final state.

        Args:
            input: Initial state values (written to input channels).
            config: Execution config.  Must include
                ``{"configurable": {"thread_id": ...}}`` when using a
                checkpointer.
            stream_mode: Ignored for invoke (consumed internally).

        Returns:
            The final state dict (filtered to ``output_channels``).

        Raises:
            GraphRecursionError: If the recursion limit is exceeded.
        """
        result = None
        for chunk in self.stream(input, config, stream_mode=stream_mode or "values"):
            result = chunk
        return result or {}

    async def ainvoke(
        self,
        input: dict,
        config: Optional[RunnableConfig] = None,
        *,
        stream_mode: Optional[StreamMode] = None,
    ) -> dict:
        """Async version of :meth:`invoke`."""
        result = None
        async for chunk in self.astream(input, config, stream_mode=stream_mode or "values"):
            result = chunk
        return result or {}

    def stream(
        self,
        input: dict,
        config: Optional[RunnableConfig] = None,
        *,
        stream_mode: Optional[StreamMode] = None,
    ) -> Iterator[dict]:
        """Execute the graph, yielding state after each superstep.

        Args:
            input: Initial state values.
            config: Execution config.
            stream_mode: ``"values"`` yields the full output state after
                each step.  ``"updates"`` yields ``{node_name: result_dict}``
                for each step.

        Yields:
            State dicts (values mode) or update dicts (updates mode).

        Raises:
            GraphRecursionError: If the recursion limit is exceeded.
        """
        config = ensure_config(config)
        mode = stream_mode or self._stream_mode
        recursion_limit = config.get("recursion_limit", 25)
        max_concurrency = config.get("max_concurrency")

        # Restore channels from checkpoint
        channels = self._init_channels(config)

        # Write input to channels
        self._apply_writes(channels, START, input)
        completed: set[str] = set()
        completed.add(START)

        if mode == "values":
            yield self._read_output(channels)

        final_next: Optional[tuple[str, ...]] = None
        for step in range(recursion_limit):
            # PLAN: determine which nodes to fire
            to_fire = self._plan(channels, completed)

            if not to_fire:
                break

            # Check interrupt_before
            interrupted = [n for n in to_fire if n in self.interrupt_before]
            if interrupted:
                final_next = tuple(interrupted)
                self._save_checkpoint(config, channels, step, completed, next_nodes=final_next)
                break

            # SNAPSHOT: read full state for nodes to consume
            state = self._read_full_state(channels)

            # EXECUTE: run planned nodes concurrently
            updates: dict[str, dict] = {}
            with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                futures = {}
                for node_name in to_fire:
                    node = self.nodes[node_name]

                    def _run(n=node, s=state):
                        return run_with_retry(lambda: _run_sync_node(n, s), n.retry_policy)

                    futures[node_name] = executor.submit(_run)
                for node_name, future in futures.items():
                    result = future.result()
                    if result is not None:
                        updates[node_name] = result

            # UPDATE: apply node outputs to channels
            new_completed: set[str] = set()
            for node_name, update in updates.items():
                if isinstance(update, dict):
                    self._apply_writes(channels, node_name, update)
                new_completed.add(node_name)

            completed = new_completed

            # STREAM: yield to caller
            if mode == "values":
                yield self._read_output(channels)
            elif mode == "updates":
                yield updates

            # Check interrupt_after
            interrupted_after = [n for n in completed if n in self.interrupt_after]
            if interrupted_after:
                final_next = tuple(self._plan(channels, completed))
                self._save_checkpoint(config, channels, step + 1, completed, next_nodes=final_next)
                break

            # Clear ephemeral channels between supersteps
            for ch in channels.values():
                if isinstance(ch, EphemeralValue):
                    ch.clear()

            # Check if END was triggered
            if END in self._get_triggered_nodes(channels, completed):
                break

            # Save checkpoint
            self._save_checkpoint(config, channels, step + 1, completed)
        else:
            raise GraphRecursionError(f"Recursion limit of {recursion_limit} reached without hitting a stop condition")

        if final_next is None:
            self._save_checkpoint(config, channels, -1, completed, next_nodes=())

    async def astream(
        self,
        input: dict,
        config: Optional[RunnableConfig] = None,
        *,
        stream_mode: Optional[StreamMode] = None,
    ) -> AsyncIterator[dict]:
        """Async version of :meth:`stream`.

        Nodes are executed concurrently via ``asyncio.gather``.  Async
        node functions are awaited natively; sync functions are wrapped
        automatically.
        """
        config = ensure_config(config)
        mode = stream_mode or self._stream_mode
        recursion_limit = config.get("recursion_limit", 25)
        max_concurrency = config.get("max_concurrency")

        channels = await self._ainit_channels(config)
        self._apply_writes(channels, START, input)
        completed: set[str] = set()
        completed.add(START)

        if mode == "values":
            yield self._read_output(channels)

        final_next: Optional[tuple[str, ...]] = None
        for step in range(recursion_limit):
            to_fire = self._plan(channels, completed)
            if not to_fire:
                break

            interrupted = [n for n in to_fire if n in self.interrupt_before]
            if interrupted:
                final_next = tuple(interrupted)
                await self._asave_checkpoint(config, channels, step, completed, next_nodes=final_next)
                break

            state = self._read_full_state(channels)

            # Execute nodes concurrently via asyncio
            updates: dict[str, dict] = {}

            async def run_node(name: str, node: PregelNode):
                return await _run_async_node(node, state, max_concurrency=max_concurrency)

            tasks = {name: asyncio.create_task(run_node(name, self.nodes[name])) for name in to_fire}
            for node_name, task in tasks.items():
                result = await task
                if result is not None:
                    updates[node_name] = result

            new_completed: set[str] = set()
            for node_name, update in updates.items():
                if isinstance(update, dict):
                    self._apply_writes(channels, node_name, update)
                new_completed.add(node_name)

            completed = new_completed

            if mode == "values":
                yield self._read_output(channels)
            elif mode == "updates":
                yield updates

            interrupted_after = [n for n in completed if n in self.interrupt_after]
            if interrupted_after:
                final_next = tuple(self._plan(channels, completed))
                await self._asave_checkpoint(config, channels, step + 1, completed, next_nodes=final_next)
                break

            for ch in channels.values():
                if isinstance(ch, EphemeralValue):
                    ch.clear()

            if END in self._get_triggered_nodes(channels, completed):
                break

            await self._asave_checkpoint(config, channels, step + 1, completed)
        else:
            raise GraphRecursionError(f"Recursion limit of {recursion_limit} reached without hitting a stop condition")

        if final_next is None:
            await self._asave_checkpoint(config, channels, -1, completed, next_nodes=())

    # ------------------------------------------------------------------
    # State management (require checkpointer)
    # ------------------------------------------------------------------

    def get_state(self, config: RunnableConfig) -> Any:
        """Read the current state from the latest checkpoint.

        Args:
            config: Must contain ``thread_id`` in ``configurable``.

        Returns:
            A ``StateSnapshot`` with ``.values`` (output-filtered state),
            ``.next`` (tuple of pending node names), ``.config``, and
            ``.metadata``.

        Raises:
            ValueError: If no checkpointer is configured.
        """
        from myagent.types import StateSnapshot

        if not self.checkpointer:
            raise ValueError("No checkpointer configured")

        checkpoint_tuple = self.checkpointer.get_tuple(config)
        if checkpoint_tuple is None:
            return StateSnapshot(
                values={},
                next=(),
                config=config,
            )

        channels = {}
        saved = checkpoint_tuple.checkpoint.get("channel_values", {})
        for name, ch in self.channels.items():
            if name in saved:
                channels[name] = ch.from_checkpoint(saved[name])
            else:
                channels[name] = ch.from_checkpoint(None)

        state = self._read_output(channels)

        next_nodes = tuple(checkpoint_tuple.metadata.get("next", ()))
        metadata = {k: v for k, v in checkpoint_tuple.metadata.items() if k != "next"}

        return StateSnapshot(
            values=state,
            next=next_nodes,
            config=checkpoint_tuple.config,
            metadata=metadata,
            created_at=checkpoint_tuple.checkpoint.get("ts"),
            parent_config=checkpoint_tuple.parent_config,
        )

    async def aget_state(self, config: RunnableConfig) -> Any:
        """Async version of :meth:`get_state`."""
        from myagent.types import StateSnapshot

        if not self.checkpointer:
            raise ValueError("No checkpointer configured")

        checkpoint_tuple = await self.checkpointer.aget_tuple(config)
        if checkpoint_tuple is None:
            return StateSnapshot(values={}, next=(), config=config)

        channels = {}
        saved = checkpoint_tuple.checkpoint.get("channel_values", {})
        for name, ch in self.channels.items():
            if name in saved:
                channels[name] = ch.from_checkpoint(saved[name])
            else:
                channels[name] = ch.from_checkpoint(None)

        state = self._read_output(channels)
        next_nodes = tuple(checkpoint_tuple.metadata.get("next", ()))
        metadata = {k: v for k, v in checkpoint_tuple.metadata.items() if k != "next"}

        return StateSnapshot(
            values=state,
            next=next_nodes,
            config=checkpoint_tuple.config,
            metadata=metadata,
            created_at=checkpoint_tuple.checkpoint.get("ts"),
            parent_config=checkpoint_tuple.parent_config,
        )

    def update_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any],
        as_node: Optional[str] = None,
    ) -> RunnableConfig:
        """Externally update the graph state through the checkpoint.

        This loads the current checkpoint, applies *values* through the
        channel system (respecting reducers), and saves a new checkpoint.

        Args:
            config: Must contain ``thread_id``.
            values: State updates to apply.
            as_node: Optional node name to attribute the update to.

        Returns:
            The new checkpoint config.

        Raises:
            ValueError: If no checkpointer is configured.
        """
        if not self.checkpointer:
            raise ValueError("No checkpointer configured")

        checkpoint_tuple = self.checkpointer.get_tuple(config)
        if checkpoint_tuple is None:
            channels = {name: ch.from_checkpoint(None) for name, ch in self.channels.items()}
        else:
            saved = checkpoint_tuple.checkpoint.get("channel_values", {})
            channels = {}
            for name, ch in self.channels.items():
                if name in saved:
                    channels[name] = ch.from_checkpoint(saved[name])
                else:
                    channels[name] = ch.from_checkpoint(None)

        self._apply_writes(channels, as_node or "__update__", values)

        channel_values = {}
        for name, ch in channels.items():
            try:
                channel_values[name] = ch.checkpoint()
            except Exception:
                pass

        checkpoint_id = create_checkpoint_id()
        checkpoint: Checkpoint = {
            "v": 1,
            "id": checkpoint_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "channel_values": channel_values,
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }

        metadata: CheckpointMetadata = {
            "source": "update",
            "step": -1,
            "writes": values,
            "parents": {},
        }

        thread_id = config.get("configurable", {}).get("thread_id", "")
        parent_id = ""
        if checkpoint_tuple:
            parent_id = checkpoint_tuple.checkpoint.get("id", "")

        return self.checkpointer.put(
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": parent_id,
                }
            },
            checkpoint,
            metadata,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_channels(self, config: RunnableConfig) -> dict[str, BaseChannel]:
        """Initialise channels, restoring from checkpoint if available."""
        channels = {}
        checkpoint_tuple = None
        if self.checkpointer:
            thread_id = config.get("configurable", {}).get("thread_id")
            if thread_id:
                checkpoint_tuple = self.checkpointer.get_tuple(config)

        if checkpoint_tuple:
            saved = checkpoint_tuple.checkpoint.get("channel_values", {})
            for name, ch in self.channels.items():
                if name in saved:
                    channels[name] = ch.from_checkpoint(saved[name])
                else:
                    channels[name] = ch.from_checkpoint(None)
        else:
            for name, ch in self.channels.items():
                channels[name] = ch.from_checkpoint(None)

        return channels

    async def _ainit_channels(self, config: RunnableConfig) -> dict[str, BaseChannel]:
        """Async version of :meth:`_init_channels`."""
        channels = {}
        checkpoint_tuple = None
        if self.checkpointer:
            thread_id = config.get("configurable", {}).get("thread_id")
            if thread_id:
                checkpoint_tuple = await self.checkpointer.aget_tuple(config)

        if checkpoint_tuple:
            saved = checkpoint_tuple.checkpoint.get("channel_values", {})
            for name, ch in self.channels.items():
                if name in saved:
                    channels[name] = ch.from_checkpoint(saved[name])
                else:
                    channels[name] = ch.from_checkpoint(None)
        else:
            for name, ch in self.channels.items():
                channels[name] = ch.from_checkpoint(None)

        return channels

    def _read_full_state(self, channels: dict[str, BaseChannel]) -> dict:
        """Read all channels - used by nodes and conditional-edge routers."""
        state = {}
        for name, ch in channels.items():
            try:
                state[name] = ch.get()
            except EmptyChannelError:
                pass
        return state

    def _read_output(self, channels: dict[str, BaseChannel]) -> dict:
        """Read only output channels - used for invoke results and get_state."""
        state = {}
        for name in self.output_channels:
            if name in channels:
                try:
                    state[name] = channels[name].get()
                except EmptyChannelError:
                    pass
        return state

    def _apply_writes(
        self,
        channels: dict[str, BaseChannel],
        source: str,
        values: dict,
    ) -> None:
        """Apply a dict of writes to the channels.

        Each value is wrapped in a single-element list and passed to
        ``channel.update([value])``.  LastValue channels overwrite;
        BinaryOperatorAggregate channels fold via the reducer.
        """
        if not isinstance(values, dict):
            return
        for key, value in values.items():
            if key in channels:
                channels[key].update([value])

    def _plan(
        self,
        channels: dict[str, BaseChannel],
        completed: set[str],
    ) -> list[str]:
        """Determine which nodes to fire this superstep.

        A node fires if any of its incoming edges (direct or conditional)
        have a source in the *completed* set.
        """
        to_fire: list[str] = []
        seen: set[str] = set()

        for node_name in self.nodes:
            if node_name in seen:
                continue
            triggered = self._is_triggered(node_name, channels, completed)
            if triggered:
                to_fire.append(node_name)
                seen.add(node_name)

        return to_fire

    def _is_triggered(
        self,
        node_name: str,
        channels: dict[str, BaseChannel],
        completed: set[str],
    ) -> bool:
        """Check if *node_name* should fire given the set of completed nodes."""
        # Direct edges
        for source, targets in self.edges.items():
            if node_name in targets and source in completed:
                return True

        # Conditional edges
        for source, cond_list in self.conditional_edges.items():
            if source not in completed:
                continue
            for router_fn, mapping in cond_list:
                for resolved in self._resolve_conditional_targets(channels, router_fn, mapping):
                    if resolved == node_name:
                        return True

        return False

    def _get_triggered_nodes(
        self,
        channels: dict[str, BaseChannel],
        completed: set[str],
    ) -> set[str]:
        """Return all nodes (including END) that would be triggered."""
        triggered: set[str] = set()

        for source, targets in self.edges.items():
            if source in completed:
                for t in targets:
                    triggered.add(t)

        for source, cond_list in self.conditional_edges.items():
            if source not in completed:
                continue
            for router_fn, mapping in cond_list:
                for resolved in self._resolve_conditional_targets(channels, router_fn, mapping):
                    triggered.add(resolved)

        return triggered

    def _resolve_conditional_targets(
        self,
        channels: dict[str, BaseChannel],
        router_fn: Any,
        mapping: Optional[dict[str, str]],
    ) -> list[Any]:
        state = self._read_full_state(channels) if channels else {}
        result = router_fn(state)
        targets = result if isinstance(result, list) else [result]
        resolved_targets: list[Any] = []
        for target in targets:
            resolved_targets.append(mapping.get(target, target) if mapping else target)
        return resolved_targets

    def _save_checkpoint(
        self,
        config: RunnableConfig,
        channels: dict[str, BaseChannel],
        step: int,
        completed: set[str],
        next_nodes: Optional[tuple[str, ...]] = None,
    ) -> Optional[RunnableConfig]:
        if not self.checkpointer:
            return None
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return None

        channel_values = {}
        for name, ch in channels.items():
            try:
                channel_values[name] = ch.checkpoint()
            except Exception:
                pass

        checkpoint_id = create_checkpoint_id()
        checkpoint: Checkpoint = {
            "v": 1,
            "id": checkpoint_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "channel_values": channel_values,
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }

        metadata: CheckpointMetadata = {
            "source": "loop",
            "step": step,
            "writes": {},
            "parents": {},
            "next": next_nodes or (),
        }

        parent_id = config.get("configurable", {}).get("checkpoint_id")
        new_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }
        if parent_id:
            new_config["configurable"]["checkpoint_id"] = checkpoint_id

        return self.checkpointer.put(
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": parent_id or "",
                }
            },
            checkpoint,
            metadata,
        )

    async def _asave_checkpoint(
        self,
        config: RunnableConfig,
        channels: dict[str, BaseChannel],
        step: int,
        completed: set[str],
        next_nodes: Optional[tuple[str, ...]] = None,
    ) -> Optional[RunnableConfig]:
        if not self.checkpointer:
            return None
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return None

        channel_values = {}
        for name, ch in channels.items():
            try:
                channel_values[name] = ch.checkpoint()
            except Exception:
                pass

        checkpoint_id = create_checkpoint_id()
        checkpoint: Checkpoint = {
            "v": 1,
            "id": checkpoint_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "channel_values": channel_values,
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }

        metadata: CheckpointMetadata = {
            "source": "loop",
            "step": step,
            "writes": {},
            "parents": {},
            "next": next_nodes or (),
        }

        parent_id = config.get("configurable", {}).get("checkpoint_id")
        return await self.checkpointer.aput(
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": parent_id or "",
                }
            },
            checkpoint,
            metadata,
        )


async def _maybe_await(result):
    """Await a result if it's a coroutine, otherwise return it directly."""
    if asyncio.iscoroutine(result):
        return await result
    return result


async def _run_async_node(
    node: PregelNode,
    state: dict[str, Any],
    *,
    max_concurrency: Optional[int] = None,
) -> Any:
    if max_concurrency is None:
        return await arun_with_retry(
            lambda: _maybe_await(node.bound(state)),
            node.retry_policy,
        )

    async with _get_concurrency_semaphore(max_concurrency):
        return await arun_with_retry(
            lambda: _maybe_await(node.bound(state)),
            node.retry_policy,
        )


def _get_concurrency_semaphore(limit: int) -> asyncio.Semaphore:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.Semaphore(limit)

    semaphores = getattr(loop, "_myagent_concurrency_semaphores", None)
    if semaphores is None:
        semaphores = {}
        setattr(loop, "_myagent_concurrency_semaphores", semaphores)

    semaphore = semaphores.get(limit)
    if semaphore is None:
        semaphore = asyncio.Semaphore(limit)
        semaphores[limit] = semaphore
    return semaphore


def _run_sync_node(node: PregelNode, state: dict[str, Any]) -> Any:
    """Execute a node in the sync runtime."""
    result = node.bound(state)
    if asyncio.iscoroutine(result):
        result.close()
        raise TypeError(f'No synchronous function provided to "{node.name}". Use ainvoke() or astream() instead.')
    if inspect.isawaitable(result):
        close = getattr(result, "close", None)
        if callable(close):
            close()
        raise TypeError(f'No synchronous function provided to "{node.name}". Use ainvoke() or astream() instead.')
    return result


async def _coerce_awaitable(result: Any) -> Any:
    return await result
