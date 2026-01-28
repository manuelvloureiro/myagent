"""Tests for Annotated state with reducers."""

import operator
from typing import Annotated, TypedDict

from myagent.graph import END, START, StateGraph


class ListState(TypedDict):
    items: Annotated[list, operator.add]
    count: int


class TestReducers:
    def test_list_reducer(self):
        """Test operator.add reducer for list accumulation."""

        def node_a(state: ListState) -> dict:
            return {"items": ["a"], "count": 1}

        def node_b(state: ListState) -> dict:
            return {"items": ["b"], "count": 2}

        graph = StateGraph(ListState)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        app = graph.compile()

        result = app.invoke({"items": [], "count": 0})
        # items should be accumulated via operator.add
        assert "a" in result["items"]
        assert "b" in result["items"]
        # count should be overwritten (LastValue)
        assert result["count"] == 2

    def test_custom_reducer(self):
        """Test custom reducer function."""

        def max_reducer(a: int, b: int) -> int:
            return max(a, b)

        class MaxState(TypedDict):
            score: Annotated[int, max_reducer]

        def node_a(state: MaxState) -> dict:
            return {"score": 10}

        def node_b(state: MaxState) -> dict:
            return {"score": 5}

        graph = StateGraph(MaxState)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        app = graph.compile()

        result = app.invoke({"score": 0})
        assert result["score"] == 10  # max(10, 5) = 10

    def test_initial_list_preserved(self):
        """Test that initial list values are used as starting point for reducer."""

        def node_a(state: ListState) -> dict:
            return {"items": ["new"], "count": 1}

        graph = StateGraph(ListState)
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)
        app = graph.compile()

        result = app.invoke({"items": ["existing"], "count": 0})
        assert "existing" in result["items"]
        assert "new" in result["items"]
