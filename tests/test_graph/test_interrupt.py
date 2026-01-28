"""Tests for interrupt_before and interrupt_after."""

from typing import TypedDict

from myagent.checkpoint.memory import InMemorySaver
from myagent.graph import END, START, StateGraph
from myagent_core.runnable.config import RunnableConfig


class InterruptState(TypedDict):
    value: str


class TestInterruptBefore:
    def test_interrupt_before_stops_execution(self):
        """Graph should stop before the interrupted node runs."""

        def node_a(state: InterruptState) -> dict:
            return {"value": state["value"] + " A"}

        def node_b(state: InterruptState) -> dict:
            return {"value": state["value"] + " B"}

        graph = StateGraph(InterruptState)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        saver = InMemorySaver()
        app = graph.compile(checkpointer=saver, interrupt_before=["b"])

        config: RunnableConfig = {"configurable": {"thread_id": "t1"}}
        result = app.invoke({"value": "start"}, config)

        # Should have executed A but stopped before B
        assert result["value"] == "start A"

        # State should be checkpointed
        state = app.get_state(config)
        assert state.values["value"] == "start A"
        assert state.next == ("b",)
        assert state.metadata["step"] == 1
        assert state.metadata["next"] == ("b",)

    def test_interrupt_before_first_node(self):
        """Interrupting before the first node should preserve input."""

        def node_a(state: InterruptState) -> dict:
            return {"value": state["value"] + " A"}

        graph = StateGraph(InterruptState)
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)
        saver = InMemorySaver()
        app = graph.compile(checkpointer=saver, interrupt_before=["a"])

        config: RunnableConfig = {"configurable": {"thread_id": "t1"}}
        result = app.invoke({"value": "input"}, config)

        # A should not have run
        assert result["value"] == "input"

        state = app.get_state(config)
        assert state.next == ("a",)
        assert state.metadata["step"] == 0


class TestInterruptAfter:
    def test_interrupt_after_stops_after_node(self):
        """Graph should stop after the interrupted node completes."""

        def node_a(state: InterruptState) -> dict:
            return {"value": state["value"] + " A"}

        def node_b(state: InterruptState) -> dict:
            return {"value": state["value"] + " B"}

        graph = StateGraph(InterruptState)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        saver = InMemorySaver()
        app = graph.compile(checkpointer=saver, interrupt_after=["a"])

        config: RunnableConfig = {"configurable": {"thread_id": "t1"}}
        result = app.invoke({"value": "start"}, config)

        # A should have run, but B should not
        assert result["value"] == "start A"

        state = app.get_state(config)
        assert state.next == ("b",)
        assert state.metadata["step"] == 0
