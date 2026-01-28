"""Tests for input_schema and output_schema parameters."""

from typing import TypedDict

from myagent.graph import END, START, StateGraph


class FullState(TypedDict):
    query: str
    intermediate: str
    answer: str


class InputSchema(TypedDict):
    query: str


class OutputSchema(TypedDict):
    answer: str


class TestInputOutputSchema:
    def test_output_schema_filters(self):
        """Only output_schema fields should appear in result."""

        def process(state: FullState) -> dict:
            return {
                "intermediate": "working...",
                "answer": f"answer to {state['query']}",
            }

        graph = StateGraph(FullState, output_schema=OutputSchema)
        graph.add_node("process", process)
        graph.add_edge(START, "process")
        graph.add_edge("process", END)
        app = graph.compile()

        result = app.invoke({"query": "test", "intermediate": "", "answer": ""})
        assert "answer" in result
        # output_schema limits what's returned
        assert "intermediate" not in result

    def test_input_and_output_schema(self):
        """Both input and output schemas together."""

        def step1(state: FullState) -> dict:
            return {"intermediate": state["query"] + " processed"}

        def step2(state: FullState) -> dict:
            return {"answer": state["intermediate"] + " done"}

        graph = StateGraph(FullState, input_schema=InputSchema, output_schema=OutputSchema)
        graph.add_node("step1", step1)
        graph.add_node("step2", step2)
        graph.add_edge(START, "step1")
        graph.add_edge("step1", "step2")
        graph.add_edge("step2", END)
        app = graph.compile()

        result = app.invoke({"query": "hello", "intermediate": "", "answer": ""})
        assert "answer" in result
        assert "done" in result["answer"]

    def test_no_schema_returns_all(self):
        """Without schemas, all state fields are returned."""

        def process(state: FullState) -> dict:
            return {"intermediate": "mid", "answer": "ans"}

        graph = StateGraph(FullState)
        graph.add_node("process", process)
        graph.add_edge(START, "process")
        graph.add_edge("process", END)
        app = graph.compile()

        result = app.invoke({"query": "q", "intermediate": "", "answer": ""})
        assert "query" in result
        assert "intermediate" in result
        assert "answer" in result
