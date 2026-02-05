"""Import-swappable: retry policy tests.

Verifies that nodes configured with RetryPolicy recover from
transient errors identically in both implementations.
"""

from typing import TypedDict


class RetryState(TypedDict):
    value: str


class TransientError(Exception):
    pass


class TestRetryCompat:
    def test_retry_recovers_from_transient_error(self, StateGraph, START, END, RetryPolicy):
        """A node that fails once then succeeds should produce a result."""
        call_count = 0

        def flaky_node(state: RetryState) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TransientError("transient failure")
            return {"value": state["value"] + " recovered"}

        policy = RetryPolicy(
            initial_interval=0.01,
            backoff_factor=1.0,
            max_interval=0.1,
            max_attempts=3,
            jitter=False,
            retry_on=TransientError,
        )

        graph = StateGraph(RetryState)
        graph.add_node("flaky", flaky_node, retry_policy=policy)
        graph.add_edge(START, "flaky")
        graph.add_edge("flaky", END)
        app = graph.compile()

        result = app.invoke({"value": "start"})
        assert result["value"] == "start recovered"
        assert call_count == 2

    def test_retry_exhausted_raises(self, StateGraph, START, END, RetryPolicy):
        """A node that always fails should raise after max_attempts."""
        import pytest

        def always_fails(state: RetryState) -> dict:
            raise TransientError("always fails")

        policy = RetryPolicy(
            initial_interval=0.01,
            backoff_factor=1.0,
            max_interval=0.1,
            max_attempts=2,
            jitter=False,
            retry_on=TransientError,
        )

        graph = StateGraph(RetryState)
        graph.add_node("broken", always_fails, retry_policy=policy)
        graph.add_edge(START, "broken")
        graph.add_edge("broken", END)
        app = graph.compile()

        with pytest.raises(TransientError):
            app.invoke({"value": "start"})
