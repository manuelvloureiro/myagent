"""Import-swappable: streaming tests."""

from typing import TypedDict


class SimpleState(TypedDict):
    value: str


class TestStreamingCompat:
    def test_stream_values_mode(self, StateGraph, START, END):
        def node_a(state: SimpleState) -> dict:
            return {"value": state["value"] + " A"}

        graph = StateGraph(SimpleState)
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)
        app = graph.compile()

        chunks = list(app.stream({"value": "hi"}, stream_mode="values"))
        assert len(chunks) >= 1
        assert chunks[-1]["value"] == "hi A"

    def test_stream_updates_mode(self, StateGraph, START, END):
        def node_a(state: SimpleState) -> dict:
            return {"value": "from_a"}

        graph = StateGraph(SimpleState)
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)
        app = graph.compile()

        chunks = list(app.stream({"value": "hi"}, stream_mode="updates"))
        assert len(chunks) >= 1
        found = any("a" in chunk for chunk in chunks)
        assert found
