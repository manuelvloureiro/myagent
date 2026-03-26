"""Import-swappable: interrupt_before / interrupt_after tests.

Verifies that graph execution pauses at the correct point when
interrupt_before or interrupt_after is set during compile.
"""

from typing import TypedDict


class InterruptState(TypedDict):
    value: str


class TestInterruptCompat:
    def test_interrupt_before_stops_before_node(self, StateGraph, START, END, InMemorySaver):
        """Graph stops before the interrupted node executes."""

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

        config = {"configurable": {"thread_id": "t-interrupt-before"}}
        result = app.invoke({"value": "start"}, config)

        # A ran, B did not
        assert "A" in result["value"]
        assert "B" not in result["value"]

        state = app.get_state(config)
        assert state.next == ("b",)
        assert state.metadata["step"] == 1

    def test_interrupt_after_stops_after_node(self, StateGraph, START, END, InMemorySaver):
        """Graph stops after the interrupted node completes."""

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

        config = {"configurable": {"thread_id": "t-interrupt-after"}}
        result = app.invoke({"value": "start"}, config)

        # A ran, B did not
        assert "A" in result["value"]
        assert "B" not in result["value"]

        state = app.get_state(config)
        assert state.next == ("b",)
        assert state.metadata["step"] == 1

    def test_interrupt_before_checkpoints_state(self, StateGraph, START, END, InMemorySaver):
        """State is checkpointed at the interrupt point."""

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

        config = {"configurable": {"thread_id": "t-interrupt-cp"}}
        app.invoke({"value": "start"}, config)

        state = app.get_state(config)
        assert "A" in state.values["value"]
        assert state.next == ("b",)
        assert "next" not in state.metadata
