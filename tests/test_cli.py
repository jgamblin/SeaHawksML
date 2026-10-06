from argparse import Namespace
from datetime import UTC, datetime

from seahawks_ml.cli import _now


def test_now_naive_is_utc():
    assert _now(Namespace(now="2026-10-11T12:00:00")) == datetime(2026, 10, 11, 12, tzinfo=UTC)


def test_now_aware_converted_to_utc():
    got = _now(Namespace(now="2026-10-11T12:00:00-07:00"))
    assert got == datetime(2026, 10, 11, 19, tzinfo=UTC) and got.utcoffset().total_seconds() == 0


def test_now_default_is_aware_utc():
    assert _now(Namespace(now=None)).tzinfo is not None
