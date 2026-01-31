"""Import-swappable: reducer tests."""

import operator
from typing import Annotated, TypedDict


class ListState(TypedDict):
    items: Annotated[list, operator.add]
    count: int


class TestReducersCompat:
    def test_list_accumulation(self, StateGraph, START, END):
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
        assert "a" in result["items"]
        assert "b" in result["items"]
        assert result["count"] == 2

    def test_initial_value_preserved(self, StateGraph, START, END):
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
