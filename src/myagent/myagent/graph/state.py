from __future__ import annotations

from typing import (
    Any,
    Callable,
    Optional,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from myagent.channels.base import BaseChannel
from myagent.channels.binop import BinaryOperatorAggregate
from myagent.channels.last_value import LastValue
from myagent.checkpoint.base import BaseCheckpointSaver
from myagent.graph._compiled import CompiledStateGraph
from myagent.pregel.read import PregelNode
from myagent.types import RetryPolicy

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated


class StateGraph:
    """Graph builder that introspects a TypedDict state schema to create channels."""

    def __init__(
        self,
        state_schema: Type,
        *,
        input_schema: Optional[Type] = None,
        output_schema: Optional[Type] = None,
    ):
        self.state_schema = state_schema
        self.input_schema = input_schema
        self.output_schema = output_schema
        self._nodes: dict[str, Callable] = {}
        self._node_retry_policies: dict[str, Optional[RetryPolicy]] = {}
        self._edges: dict[str, list[str]] = {}  # source -> [target]
        self._conditional_edges: dict[str, list[tuple[Callable, Optional[dict[str, str]]]]] = {}
        self._channels: dict[str, BaseChannel] = {}
        self._subgraphs: dict[str, tuple[Any, Callable, Callable]] = {}

        # Introspect state schema to build channels
        self._build_channels()

    def _build_channels(self) -> None:
        """Introspect TypedDict annotations to create appropriate channels."""
        try:
            hints = get_type_hints(self.state_schema, include_extras=True)
        except Exception:
            hints = getattr(self.state_schema, "__annotations__", {})

        for name, hint in hints.items():
            if get_origin(hint) is Annotated:
                args = get_args(hint)
                base_type = args[0]
                reducer = args[1] if len(args) > 1 else None
                if reducer is not None and callable(reducer):
                    self._channels[name] = BinaryOperatorAggregate(base_type, reducer)
                else:
                    self._channels[name] = LastValue(hint)
            else:
                self._channels[name] = LastValue(hint)

    def add_node(
        self,
        node: Union[str, Callable],
        action: Optional[Callable] = None,
        *,
        retry: Optional[RetryPolicy] = None,
        retry_policy: Optional[RetryPolicy] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> StateGraph:
        """Add a node to the graph."""
        if callable(node) and action is None:
            name = getattr(node, "__name__", str(node))
            action = node
        elif isinstance(node, str) and action is not None:
            name = node
        else:
            raise ValueError("add_node expects either (name, fn) or (fn,) with name inferred")

        policy = retry_policy or retry
        self._nodes[name] = action
        self._node_retry_policies[name] = policy
        return self

    def add_subgraph(
        self,
        name: str,
        compiled_graph: Any,
        *,
        input_map: Callable,
        output_map: Callable,
    ) -> StateGraph:
        """Add a compiled sub-graph as a node."""
        self._subgraphs[name] = (compiled_graph, input_map, output_map)
        return self

    def add_edge(
        self,
        source: Union[str, list[str]],
        target: str,
    ) -> StateGraph:
        """Add a direct edge from source to target."""
        if isinstance(source, list):
            # Fan-in: multiple sources to one target
            for s in source:
                self._edges.setdefault(s, []).append(target)
        else:
            self._edges.setdefault(source, []).append(target)
        return self

    def add_conditional_edges(
        self,
        source: str,
        path: Callable,
        path_map: Optional[dict[str, str]] = None,
        then: Optional[str] = None,
    ) -> StateGraph:
        """Add conditional edges from source using a router function."""
        if then is not None:
            raise NotImplementedError("add_conditional_edges(..., then=...) is not supported in myagent")
        self._conditional_edges.setdefault(source, []).append((path, path_map))
        return self

    def compile(
        self,
        *,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        interrupt_before: Optional[list[str]] = None,
        interrupt_after: Optional[list[str]] = None,
    ) -> CompiledStateGraph:
        """Compile the graph into a runnable CompiledStateGraph."""
        # Determine output channels
        if self.output_schema:
            try:
                output_hints = get_type_hints(self.output_schema, include_extras=True)
            except Exception:
                output_hints = getattr(self.output_schema, "__annotations__", {})
            output_channels = list(output_hints.keys())
        else:
            output_channels = list(self._channels.keys())

        # Determine input channels
        if self.input_schema:
            try:
                input_hints = get_type_hints(self.input_schema, include_extras=True)
            except Exception:
                input_hints = getattr(self.input_schema, "__annotations__", {})
            input_channels = list(input_hints.keys())
        else:
            input_channels = list(self._channels.keys())

        # Build PregelNodes
        nodes: dict[str, PregelNode] = {}
        for name, action in self._nodes.items():
            nodes[name] = PregelNode(
                name=name,
                bound=action,
                retry_policy=self._node_retry_policies.get(name),
            )

        for name, (compiled, input_map, output_map) in self._subgraphs.items():
            nodes[name] = PregelNode(
                name=name,
                bound=_make_subgraph_wrapper(compiled, input_map, output_map),
            )

        return CompiledStateGraph(
            nodes=nodes,
            channels=self._channels,
            input_channels=input_channels,
            output_channels=output_channels,
            edges=self._edges,
            conditional_edges=self._conditional_edges,
            checkpointer=checkpointer,
            interrupt_before=interrupt_before or [],
            interrupt_after=interrupt_after or [],
            builder=self,
        )


class _SubgraphNode:
    """Callable node that invokes a compiled sub-graph.

    Provides both sync and async entry points so the Pregel engine can
    call ``invoke()`` or ``ainvoke()`` without runtime context detection.
    The sync ``__call__`` delegates to ``compiled_graph.invoke()``; the
    async path is exposed via the ``ainvoke`` coroutine attribute which
    ``_maybe_await`` picks up in the async engine.
    """

    __slots__ = ("_graph", "_input_map", "_output_map")

    def __init__(self, compiled_graph: Any, input_map: Callable, output_map: Callable) -> None:
        self._graph = compiled_graph
        self._input_map = input_map
        self._output_map = output_map

    def __call__(self, state: dict) -> dict:
        sub_input = self._input_map(state)
        result = self._graph.invoke(sub_input)
        return self._output_map(result, state)

    async def ainvoke(self, state: dict) -> dict:
        sub_input = self._input_map(state)
        result = await self._graph.ainvoke(sub_input)
        return self._output_map(result, state)


def _make_subgraph_wrapper(
    compiled_graph: Any,
    input_map: Callable,
    output_map: Callable,
) -> _SubgraphNode:
    """Create a node callable that invokes a compiled sub-graph."""
    return _SubgraphNode(compiled_graph, input_map, output_map)
