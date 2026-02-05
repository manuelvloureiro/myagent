"""Import-swappable: add_messages reducer tests.

Verifies ID-based deduplication, replacement, and append semantics
of the add_messages reducer. Messages must include a 'role' key
as required by langgraph's message coercion.
"""


class TestAddMessagesCompat:
    def test_append_new_messages(self, add_messages):
        """New messages (different IDs) are appended."""
        left = [{"id": "1", "role": "user", "content": "hello"}]
        right = [{"id": "2", "role": "assistant", "content": "world"}]
        result = add_messages(left, right)
        assert len(result) == 2

    def test_replace_by_id(self, add_messages):
        """Messages with matching IDs are replaced, not duplicated."""
        left = [{"id": "1", "role": "user", "content": "old"}]
        right = [{"id": "1", "role": "user", "content": "new"}]
        result = add_messages(left, right)
        assert len(result) == 1
        content = result[0]["content"] if isinstance(result[0], dict) else result[0].content
        assert content == "new"

    def test_mixed_replace_and_append(self, add_messages):
        """Some messages replace, others append."""
        left = [
            {"id": "1", "role": "user", "content": "first"},
            {"id": "2", "role": "assistant", "content": "second"},
        ]
        right = [
            {"id": "2", "role": "assistant", "content": "updated"},
            {"id": "3", "role": "user", "content": "third"},
        ]
        result = add_messages(left, right)
        assert len(result) == 3

    def test_single_dict_right(self, add_messages):
        """Right side can be a single dict instead of a list."""
        left = [{"id": "1", "role": "user", "content": "hi"}]
        right = {"id": "2", "role": "assistant", "content": "there"}
        result = add_messages(left, right)
        assert len(result) == 2

    def test_empty_left(self, add_messages):
        """Empty left list just returns right messages."""
        result = add_messages([], [{"id": "1", "role": "user", "content": "hi"}])
        assert len(result) == 1
