"""Import-swappable: MessagesState tests."""


class TestMessagesCompat:
    def test_messages_graph(self, StateGraph, START, END, MessagesState):
        def chatbot(state):
            last = state["messages"][-1]
            content = last["content"] if isinstance(last, dict) else last.content
            return {"messages": [{"role": "assistant", "content": f"reply to: {content}"}]}

        graph = StateGraph(MessagesState)
        graph.add_node("chatbot", chatbot)
        graph.add_edge(START, "chatbot")
        graph.add_edge("chatbot", END)
        app = graph.compile()

        result = app.invoke({"messages": [{"role": "user", "content": "hello"}]})
        messages = result["messages"]
        assert len(messages) >= 2
        last = messages[-1]
        content = last["content"] if isinstance(last, dict) else last.content
        assert "reply to: hello" in content
