from argparse import Namespace
from datetime import UTC, datetime, timedelta

import polars as pl
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


# ---- _predict end to end (offline) ----

@pytest.fixture
def predict_env(monkeypatch, tmp_path):
    import polars as pl

    from seahawks_ml.features.build import build_features
    from seahawks_ml.ingest import nflverse, weather
    from seahawks_ml.models import store
    from seahawks_ml.models.pipeline import ModelConfig, fit_model
    from seahawks_ml.stadiums import load_stadiums
    from tests.synthetic import make_raw

    raw = make_raw(seasons=(2009, 2010, 2011, 2012, 2013, 2014), unplayed_last_week=True)
    stadiums = load_stadiums()
    frame = build_features(raw, stadiums)
    model = fit_model(frame, ModelConfig(inner_folds=3), [2009, 2010, 2011, 2012, 2013])
    env = type("Env", (), {})()
    env.raw, env.forecast_calls, env.loads = raw, [], []
    env.hist, env.league = tmp_path / "h.jsonl", tmp_path / "l.jsonl"
    env.week6 = raw.games.filter((pl.col("season") == 2014) & (pl.col("week") == 6))
    env.kick = env.week6["kickoff_utc"][0]

    def fake_forecast(games, st, client=None):
        env.forecast_calls.append(set(games["game_id"].to_list()))
        return pl.DataFrame(schema=weather.WEATHER_SCHEMA)

    def fake_load(now, games=None):
        env.loads.append(now)
        return stadiums, raw

    monkeypatch.setattr(cli, "PREDICTIONS_PATH", env.hist)
    monkeypatch.setattr(cli, "LEAGUE_PATH", env.league)
    monkeypatch.setattr(nflverse, "load_schedules", lambda: raw.games)
    monkeypatch.setattr("seahawks_ml.features.base.prepare_games", lambda s, st: s)
    monkeypatch.setattr(weather, "forecast_for_games", fake_forecast)
    monkeypatch.setattr(cli, "_load", fake_load)
    monkeypatch.setattr(cli, "_build", lambda r, st, c: build_features(r, st))
    monkeypatch.setattr(store, "load_config", lambda: store.ProjectConfig())
    monkeypatch.setattr(store, "load_model", lambda: model)
    monkeypatch.setattr(store, "check_model_matches", lambda m, c: None)
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"model_version": "vtest"}')
    monkeypatch.setattr(store, "METRICS_PATH", metrics)
    return env


def _args(now, run_type=None):
    return Namespace(now=now.isoformat(), run_type=run_type)


def test_predict_logs_seahawks_and_league(predict_env):
    from seahawks_ml.pipeline.history import read_history
    env, changed = predict_env, []
    cli._predict(_args(env.kick - timedelta(hours=12)), changed)
    assert changed
    assert [r["run_type"] for r in read_history(env.hist)] == ["final_injury"]
    league = read_history(env.league)
    assert sorted(r["game_id"] for r in league) == sorted(env.week6["game_id"].to_list())
    assert env.forecast_calls == [set(env.week6["game_id"].to_list())]
    assert len(env.loads) == 1


def test_predict_league_only_when_seahawks_already_run(predict_env):
    from seahawks_ml.pipeline.history import read_history
    env, changed = predict_env, []
    now = env.kick - timedelta(hours=12)
    cli._predict(_args(now), [])
    env.league.unlink()  # pretend only the league log is missing
    cli._predict(_args(now + timedelta(hours=1)), changed)
    assert changed and len(read_history(env.league)) == 2
    assert [r["run_type"] for r in read_history(env.hist)] == ["final_injury"]  # no second Seahawks run


def test_predict_nothing_due_loads_nothing(predict_env):
    env, changed = predict_env, []
    cli._predict(_args(env.kick - timedelta(hours=36)), changed)
    assert not changed and env.loads == []


def test_predict_records_league_results(predict_env, monkeypatch):
    from seahawks_ml.pipeline.history import read_history
    env = predict_env
    cli._predict(_args(env.kick - timedelta(hours=12)), [])
    # pretend the week-6 games finished: mark margins in the schedule
    done = env.raw.games.with_columns(
        pl.when(pl.col("margin").is_null()).then(3).otherwise(pl.col("margin")).alias("margin"),
        pl.when(pl.col("home_score").is_null()).then(23).otherwise(pl.col("home_score")).alias("home_score"),
        pl.when(pl.col("away_score").is_null()).then(20).otherwise(pl.col("away_score")).alias("away_score"))
    from seahawks_ml.ingest import nflverse
    monkeypatch.setattr(nflverse, "load_schedules", lambda: done)
    changed = []
    cli._predict(_args(env.kick + timedelta(days=1)), changed)
    types = [r["type"] for r in read_history(env.league)]
    assert types.count("league_result") == 2 and changed
