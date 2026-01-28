"""Tests for retry on transient errors."""

import pytest
from myagent.pregel.retry import arun_with_retry, run_with_retry
from myagent.types import RetryPolicy


class TransientError(Exception):
    pass


class TestRetry:
    def test_succeeds_first_try(self):
        calls = []

        def fn():
            calls.append(1)
            return "ok"

        result = run_with_retry(fn, RetryPolicy(max_attempts=3))
        assert result == "ok"
        assert len(calls) == 1

    def test_retries_on_failure(self):
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise TransientError("transient")
            return "recovered"

        policy = RetryPolicy(
            initial_interval=0.01,
            max_attempts=5,
            jitter=False,
            retry_on=TransientError,
        )
        result = run_with_retry(fn, policy)
        assert result == "recovered"
        assert len(calls) == 3

    def test_exhausts_retries(self):
        calls = []

        def fn():
            calls.append(1)
            raise TransientError("always fails")

        policy = RetryPolicy(
            initial_interval=0.01,
            max_attempts=3,
            jitter=False,
            retry_on=TransientError,
        )
        with pytest.raises(TransientError):
            run_with_retry(fn, policy)
        assert len(calls) == 3

    def test_no_retry_on_unmatched_exception(self):
        calls = []

        def fn():
            calls.append(1)
            raise ValueError("not transient")

        policy = RetryPolicy(
            initial_interval=0.01,
            max_attempts=3,
            retry_on=TransientError,
        )
        with pytest.raises(ValueError):
            run_with_retry(fn, policy)
        assert len(calls) == 1

    def test_no_policy_runs_once(self):
        calls = []

        def fn():
            calls.append(1)
            return "done"

        result = run_with_retry(fn, None)
        assert result == "done"
        assert len(calls) == 1


class TestAsyncRetry:
    @pytest.mark.asyncio
    async def test_async_retries(self):
        calls = []

        async def fn():
            calls.append(1)
            if len(calls) < 2:
                raise TransientError("transient")
            return "ok"

        policy = RetryPolicy(
            initial_interval=0.01,
            max_attempts=3,
            jitter=False,
            retry_on=TransientError,
        )
        result = await arun_with_retry(fn, policy)
        assert result == "ok"
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_async_exhausts_retries(self):
        calls = []

        async def fn():
            calls.append(1)
            raise TransientError("fail")

        policy = RetryPolicy(
            initial_interval=0.01,
            max_attempts=2,
            jitter=False,
            retry_on=TransientError,
        )
        with pytest.raises(TransientError):
            await arun_with_retry(fn, policy)
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_async_no_policy(self):
        async def fn():
            return 42

        result = await arun_with_retry(fn, None)
        assert result == 42
