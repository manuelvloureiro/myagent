"""Import-swappable: fan-in edge tests.

Verifies that add_edge([a, b], c) fan-in syntax works identically.
"""

import operator
from typing import Annotated, TypedDict


class FanInState(TypedDict):
    items: Annotated[list, operator.add]


class TestFanInCompat:
    def test_two_sources_fan_in(self, StateGraph, START, END):
        """Two parallel nodes feed into a single joiner via list edge syntax."""

        def node_a(state: FanInState) -> dict:
            return {"items": ["from_a"]}

        def node_b(state: FanInState) -> dict:
            return {"items": ["from_b"]}

        def joiner(state: FanInState) -> dict:
            return {"items": ["joined"]}

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
        assert "from_a" in result["items"]
        assert "from_b" in result["items"]
        assert "joined" in result["items"]
