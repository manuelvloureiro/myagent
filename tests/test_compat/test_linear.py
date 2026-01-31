"""Import-swappable: linear graph tests."""

from typing import TypedDict


class SimpleState(TypedDict):
    value: str


class TestLinearCompat:
    def test_single_node(self, StateGraph, START, END):
        def node_a(state: SimpleState) -> dict:
            return {"value": state["value"] + " processed"}

        graph = StateGraph(SimpleState)
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)
        app = graph.compile()

        result = app.invoke({"value": "hello"})
        assert result["value"] == "hello processed"

    def test_two_nodes_chain(self, StateGraph, START, END):
        def node_a(state: SimpleState) -> dict:
            return {"value": state["value"] + " A"}

        def node_b(state: SimpleState) -> dict:
            return {"value": state["value"] + " B"}

        graph = StateGraph(SimpleState)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        app = graph.compile()

        result = app.invoke({"value": "x"})
        assert result["value"] == "x A B"

    def test_inferred_node_name(self, StateGraph, START, END):
        def my_node(state: SimpleState) -> dict:
            return {"value": state["value"] + " inferred"}

        graph = StateGraph(SimpleState)
        graph.add_node(my_node)
        graph.add_edge(START, "my_node")
        graph.add_edge("my_node", END)
        app = graph.compile()

        result = app.invoke({"value": "test"})
        assert result["value"] == "test inferred"
