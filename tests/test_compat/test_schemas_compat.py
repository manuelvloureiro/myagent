"""Import-swappable: input_schema / output_schema tests.

Verifies that output_schema filters the returned state and that
nodes still see the full internal state regardless of schemas.
"""

from typing import TypedDict


class FullState(TypedDict):
    query: str
    intermediate: str
    answer: str


class OutputOnly(TypedDict):
    answer: str


class TestSchemasCompat:
    def test_output_schema_filters_result(self, StateGraph, START, END):
        """Only fields in output_schema appear in the invoke result."""

        def process(state: FullState) -> dict:
            return {
                "intermediate": "working",
                "answer": f"answer to {state['query']}",
            }

        graph = StateGraph(FullState, output_schema=OutputOnly)
        graph.add_node("process", process)
        graph.add_edge(START, "process")
        graph.add_edge("process", END)
        app = graph.compile()

        result = app.invoke({"query": "test", "intermediate": "", "answer": ""})
        assert "answer" in result
        assert "intermediate" not in result
        assert "test" in result["answer"]

    def test_nodes_see_full_state_despite_output_schema(self, StateGraph, START, END):
        """Nodes can read all state fields even when output_schema is set."""

        def step1(state: FullState) -> dict:
            return {"intermediate": state["query"] + " processed"}

        def step2(state: FullState) -> dict:
            return {"answer": state["intermediate"] + " done"}

        graph = StateGraph(FullState, output_schema=OutputOnly)
        graph.add_node("step1", step1)
        graph.add_node("step2", step2)
        graph.add_edge(START, "step1")
        graph.add_edge("step1", "step2")
        graph.add_edge("step2", END)
        app = graph.compile()

        result = app.invoke({"query": "hello", "intermediate": "", "answer": ""})
        assert "answer" in result
        assert "done" in result["answer"]
