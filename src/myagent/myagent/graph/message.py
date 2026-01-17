from __future__ import annotations

import uuid
from typing import Annotated, Any, TypedDict, Union


def add_messages(
    left: list[dict[str, Any]],
    right: Union[dict[str, Any], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Reducer that merges messages by ID. Matching IDs are replaced, new ones appended."""
    if isinstance(right, dict):
        right = [right]

    # Ensure all messages have IDs
    for msg in left:
        if "id" not in msg:
            msg["id"] = str(uuid.uuid4())
    for msg in right:
        if "id" not in msg:
            msg["id"] = str(uuid.uuid4())

    # Build index of existing messages by ID
    merged = list(left)
    id_to_idx = {msg["id"]: i for i, msg in enumerate(merged)}

    for msg in right:
        if msg["id"] in id_to_idx:
            merged[id_to_idx[msg["id"]]] = msg
        else:
            id_to_idx[msg["id"]] = len(merged)
            merged.append(msg)

    return merged


class MessagesState(TypedDict):
    messages: Annotated[list[dict[str, Any]], add_messages]
