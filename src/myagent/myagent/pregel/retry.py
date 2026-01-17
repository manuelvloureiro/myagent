from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Callable, Optional

from myagent.types import RetryPolicy


def run_with_retry(
    fn: Callable[[], Any],
    retry_policy: Optional[RetryPolicy] = None,
) -> Any:
    """Run a function with retry logic."""
    if retry_policy is None:
        return fn()

    last_exc: Optional[Exception] = None
    interval = retry_policy.initial_interval
    for attempt in range(retry_policy.max_attempts):
        try:
            return fn()
        except retry_policy.retry_on as exc:
            last_exc = exc
            if attempt == retry_policy.max_attempts - 1:
                raise
            sleep_time = min(interval, retry_policy.max_interval)
            if retry_policy.jitter:
                sleep_time *= random.uniform(0.5, 1.5)
            time.sleep(sleep_time)
            interval *= retry_policy.backoff_factor
    if last_exc is not None:
        raise last_exc


async def arun_with_retry(
    fn: Callable[[], Any],
    retry_policy: Optional[RetryPolicy] = None,
) -> Any:
    """Run an async function with retry logic."""
    if retry_policy is None:
        return await fn()

    last_exc: Optional[Exception] = None
    interval = retry_policy.initial_interval
    for attempt in range(retry_policy.max_attempts):
        try:
            return await fn()
        except retry_policy.retry_on as exc:
            last_exc = exc
            if attempt == retry_policy.max_attempts - 1:
                raise
            sleep_time = min(interval, retry_policy.max_interval)
            if retry_policy.jitter:
                sleep_time *= random.uniform(0.5, 1.5)
            await asyncio.sleep(sleep_time)
            interval *= retry_policy.backoff_factor
    if last_exc is not None:
        raise last_exc
