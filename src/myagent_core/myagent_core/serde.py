from __future__ import annotations

import json
import uuid
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any


class JsonPlusSerializer:
    """JSON serializer with support for additional Python types."""

    def dumps(self, obj: Any) -> bytes:
        return json.dumps(obj, default=self._default).encode("utf-8")

    def loads(self, data: bytes) -> Any:
        return json.loads(data, object_hook=self._object_hook)

    def _default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return {"__type__": "datetime", "value": obj.isoformat()}
        if isinstance(obj, date):
            return {"__type__": "date", "value": obj.isoformat()}
        if isinstance(obj, time):
            return {"__type__": "time", "value": obj.isoformat()}
        if isinstance(obj, timedelta):
            return {"__type__": "timedelta", "value": obj.total_seconds()}
        if isinstance(obj, uuid.UUID):
            return {"__type__": "uuid", "value": str(obj)}
        if isinstance(obj, Decimal):
            return {"__type__": "decimal", "value": str(obj)}
        if isinstance(obj, bytes):
            return {"__type__": "bytes", "value": obj.decode("latin-1")}
        if isinstance(obj, set):
            return {"__type__": "set", "value": list(obj)}
        if isinstance(obj, frozenset):
            return {"__type__": "frozenset", "value": list(obj)}
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def _object_hook(self, obj: dict) -> Any:
        type_tag = obj.get("__type__")
        if type_tag is None:
            return obj
        value = obj["value"]
        if type_tag == "datetime":
            return datetime.fromisoformat(value)
        if type_tag == "date":
            return date.fromisoformat(value)
        if type_tag == "time":
            return time.fromisoformat(value)
        if type_tag == "timedelta":
            return timedelta(seconds=value)
        if type_tag == "uuid":
            return uuid.UUID(value)
        if type_tag == "decimal":
            return Decimal(value)
        if type_tag == "bytes":
            return value.encode("latin-1")
        if type_tag == "set":
            return set(value)
        if type_tag == "frozenset":
            return frozenset(value)
        return obj
