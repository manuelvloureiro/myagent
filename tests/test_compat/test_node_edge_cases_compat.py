"""Import-swappable: node return value edge cases.

Verifies that both implementations handle nodes returning None,
empty dicts, or partial state updates identically.
"""

import operator
from typing import Annotated, TypedDict


class SimpleState(TypedDict):
    value: str


class MultiState(TypedDict):
    a: str
    b: int
    c: Annotated[list, operator.add]


class TestNodeEdgeCasesCompat:
    def test_node_returns_none(self, StateGraph, START, END):
        """A node returning None should not alter state."""

        def noop(state: SimpleState):
            return None

        graph = StateGraph(SimpleState)
        graph.add_node("noop", noop)
        graph.add_edge(START, "noop")
        graph.add_edge("noop", END)
        app = graph.compile()

        result = app.invoke({"value": "unchanged"})
        assert result["value"] == "unchanged"

    def test_node_returns_empty_dict(self, StateGraph, START, END):
        """A node returning {} should not alter state."""

        def empty(state: SimpleState) -> dict:
            return {}

        graph = StateGraph(SimpleState)
        graph.add_node("empty", empty)
        graph.add_edge(START, "empty")
        graph.add_edge("empty", END)
        app = graph.compile()

        result = app.invoke({"value": "unchanged"})
        assert result["value"] == "unchanged"

    def test_node_returns_partial_state(self, StateGraph, START, END):
        """A node updating only some fields leaves others intact."""

        def partial(state: MultiState) -> dict:
            return {"a": "new_a"}

        graph = StateGraph(MultiState)
        graph.add_node("partial", partial)
        graph.add_edge(START, "partial")
        graph.add_edge("partial", END)
        app = graph.compile()

        result = app.invoke({"a": "old", "b": 42, "c": []})
        assert result["a"] == "new_a"
        assert result["b"] == 42
        assert result["c"] == []
