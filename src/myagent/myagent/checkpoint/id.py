from __future__ import annotations

import time
import uuid


def create_checkpoint_id() -> str:
    """Create a lexicographically sortable checkpoint ID."""
    return f"{time.time_ns():020d}-{uuid.uuid4()}"
