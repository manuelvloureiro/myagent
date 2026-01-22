"""Tests for fan-in edges (multiple sources converging on one target)."""

import operator
from typing import Annotated, TypedDict

from myagent.graph import END, START, StateGraph


class FanInState(TypedDict):
    items: Annotated[list, operator.add]


class TestFanIn:
    def test_two_sources_fan_in(self):
        """START -> [a, b] both feed into c -> END via reducer."""

        def node_a(state: FanInState) -> dict:
            return {"items": ["from_a"]}

        def node_b(state: FanInState) -> dict:
            return {"items": ["from_b"]}

        def node_c(state: FanInState) -> dict:
            return {"items": ["from_c"]}

        graph = StateGraph(FanInState)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_node("c", node_c)
        graph.add_edge(START, "a")
        graph.add_edge(START, "b")
        graph.add_edge(["a", "b"], "c")
        graph.add_edge("c", END)
        app = graph.compile()

        result = app.invoke({"items": []})
        assert "from_a" in result["items"]
        assert "from_b" in result["items"]
        assert "from_c" in result["items"]

    def test_fan_in_list_syntax(self):
        """Test add_edge with list source syntax."""

        def node_a(state: FanInState) -> dict:
            return {"items": ["a"]}

        def node_b(state: FanInState) -> dict:
            return {"items": ["b"]}

        def joiner(state: FanInState) -> dict:
            return {"items": [f"joined({len(state['items'])})"]}

        graph = StateGraph(FanInState)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_node("joiner", joiner)
        graph.add_edge(START, "a")
        graph.add_edge(START, "b")
        graph.add_edge(["a", "b"], "joiner")
        graph.add_edge("joiner", END)
        app = graph.compile()

        result = app.invoke({"items": []})
        assert "a" in result["items"]
        assert "b" in result["items"]
        assert any("joined" in item for item in result["items"])
