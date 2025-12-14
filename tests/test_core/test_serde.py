"""Extended tests for JsonPlusSerializer."""

import uuid
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

from myagent_core.serde import JsonPlusSerializer


class TestSerdeEdgeCases:
    def setup_method(self):
        self.serde = JsonPlusSerializer()

    def test_nested_special_types(self):
        data = {
            "id": uuid.uuid4(),
            "created": datetime(2024, 6, 15, tzinfo=timezone.utc),
            "tags": {"a", "b"},
        }
        result = self.serde.loads(self.serde.dumps(data))
        assert result["id"] == data["id"]
        assert result["created"] == data["created"]
        assert result["tags"] == data["tags"]

    def test_date(self):
        d = date(2024, 1, 15)
        assert self.serde.loads(self.serde.dumps(d)) == d

    def test_time(self):
        t = time(14, 30, 0)
        assert self.serde.loads(self.serde.dumps(t)) == t

    def test_timedelta(self):
        td = timedelta(hours=2, minutes=30)
        result = self.serde.loads(self.serde.dumps(td))
        assert result == td

    def test_frozenset(self):
        fs = frozenset([1, 2, 3])
        result = self.serde.loads(self.serde.dumps(fs))
        assert result == fs

    def test_bytes_with_high_bytes(self):
        b = bytes(range(256))
        result = self.serde.loads(self.serde.dumps(b))
        assert result == b

    def test_none(self):
        assert self.serde.loads(self.serde.dumps(None)) is None

    def test_list(self):
        data = [1, "two", 3.0]
        assert self.serde.loads(self.serde.dumps(data)) == data

    def test_nested_dict(self):
        data = {"a": {"b": {"c": 42}}}
        assert self.serde.loads(self.serde.dumps(data)) == data

    def test_decimal_precision(self):
        d = Decimal("3.14159265358979323846")
        result = self.serde.loads(self.serde.dumps(d))
        assert result == d
