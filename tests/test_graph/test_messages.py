"""Tests for add_messages reducer and MessagesState."""

from myagent.graph import END, START, MessagesState, StateGraph, add_messages
from myagent_core.runnable.config import RunnableConfig


class TestAddMessages:
    def test_append_new(self):
        left = [{"id": "1", "content": "hello"}]
        right = [{"id": "2", "content": "world"}]
        result = add_messages(left, right)
        assert len(result) == 2
        assert result[0]["content"] == "hello"
        assert result[1]["content"] == "world"

    def test_replace_by_id(self):
        left = [{"id": "1", "content": "old"}]
        right = [{"id": "1", "content": "new"}]
        result = add_messages(left, right)
        assert len(result) == 1
        assert result[0]["content"] == "new"

    def test_single_dict_right(self):
        left = [{"id": "1", "content": "hi"}]
        right = {"id": "2", "content": "there"}
        result = add_messages(left, right)
        assert len(result) == 2

    def test_auto_id_generation(self):
        left = [{"content": "no id"}]
        right = [{"content": "also no id"}]
        result = add_messages(left, right)
        assert len(result) == 2
        assert "id" in result[0]
        assert "id" in result[1]
        assert result[0]["id"] != result[1]["id"]

    def test_mixed_replace_and_append(self):
        left = [
            {"id": "1", "content": "first"},
            {"id": "2", "content": "second"},
        ]
        right = [
            {"id": "2", "content": "updated_second"},
            {"id": "3", "content": "third"},
        ]
        result = add_messages(left, right)
        assert len(result) == 3
        assert result[0]["content"] == "first"
        assert result[1]["content"] == "updated_second"
        assert result[2]["content"] == "third"

    def test_empty_left(self):
        result = add_messages([], [{"id": "1", "content": "hi"}])
        assert len(result) == 1

    def test_empty_right(self):
        left = [{"id": "1", "content": "hi"}]
        result = add_messages(left, [])
        assert len(result) == 1


class TestMessagesStateGraph:
    def test_messages_state_graph(self):
        def chatbot(state):
            last = state["messages"][-1]
            return {"messages": [{"role": "assistant", "content": f"reply to: {last['content']}"}]}

        graph = StateGraph(MessagesState)
        graph.add_node("chatbot", chatbot)
        graph.add_edge(START, "chatbot")
        graph.add_edge("chatbot", END)
        app = graph.compile()

        result = app.invoke({"messages": [{"role": "user", "content": "hello"}]})
        assert len(result["messages"]) == 2
        assert "reply to: hello" in result["messages"][-1]["content"]

    def test_messages_state_multi_turn(self):
        from myagent.checkpoint.memory import InMemorySaver

        def chatbot(state):
            last = state["messages"][-1]
            return {"messages": [{"role": "assistant", "content": f"echo: {last['content']}"}]}

        graph = StateGraph(MessagesState)
        graph.add_node("chatbot", chatbot)
        graph.add_edge(START, "chatbot")
        graph.add_edge("chatbot", END)
        saver = InMemorySaver()
        app = graph.compile(checkpointer=saver)

        config: RunnableConfig = {"configurable": {"thread_id": "chat1"}}

        r1 = app.invoke({"messages": [{"role": "user", "content": "hi"}]}, config)
        assert len(r1["messages"]) == 2

        r2 = app.invoke({"messages": [{"role": "user", "content": "how are you"}]}, config)
        # Should have accumulated: hi, echo:hi, how are you, echo:how are you
        assert len(r2["messages"]) > len(r1["messages"])
        assert any("echo: how are you" in m["content"] for m in r2["messages"] if "content" in m)
