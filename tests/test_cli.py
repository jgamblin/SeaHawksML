from argparse import Namespace
from datetime import UTC, datetime

import pytest

from seahawks_ml import cli
from seahawks_ml.cli import _now


def test_now_naive_is_utc():
    assert _now(Namespace(now="2026-10-11T12:00:00")) == datetime(2026, 10, 11, 12, tzinfo=UTC)


def test_now_aware_converted_to_utc():
    got = _now(Namespace(now="2026-10-11T12:00:00-07:00"))
    assert got == datetime(2026, 10, 11, 19, tzinfo=UTC) and got.utcoffset().total_seconds() == 0


def test_now_default_is_aware_utc():
    assert _now(Namespace(now=None)).tzinfo is not None


def _run_predict(monkeypatch, tmp_path, body):
    out = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setattr(cli, "_predict", body)
    cli.cmd_predict(Namespace())
    return out.read_text()


def test_changed_output_written_once_false(monkeypatch, tmp_path):
    assert _run_predict(monkeypatch, tmp_path, lambda args, changed: None) == "changed=false\n"


def test_changed_output_written_once_true(monkeypatch, tmp_path):
    assert _run_predict(monkeypatch, tmp_path, lambda args, changed: changed.append(True)) == "changed=true\n"


def test_changed_output_survives_failure(monkeypatch, tmp_path):
    def boom(args, changed):
        changed.append(True)
        raise RuntimeError("late failure")

    out = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setattr(cli, "_predict", boom)
    with pytest.raises(RuntimeError):
        cli.cmd_predict(Namespace())
    assert out.read_text() == "changed=true\n"
