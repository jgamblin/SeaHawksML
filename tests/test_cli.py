from argparse import Namespace
from dataclasses import replace
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
    # one outdoor league game in week 6 (the other two are indoors) so forecast handling is exercised
    raw = replace(raw, games=raw.games.with_columns(
        pl.when(pl.col("game_id") == "2014_06_SF_LA").then(pl.lit("outdoors")).otherwise(pl.col("roof")).alias("roof"),
        pl.when(pl.col("game_id") == "2014_06_SF_LA").then(pl.lit("SFO01")).otherwise(pl.col("stadium_id"))
        .alias("stadium_id")))
    stadiums = load_stadiums()
    frame = build_features(raw, stadiums)
    model = fit_model(frame, ModelConfig(inner_folds=3), [2009, 2010, 2011, 2012, 2013])
    env = type("Env", (), {})()
    env.raw, env.forecast_calls, env.loads = raw, [], []
    env.hist, env.league = tmp_path / "h.jsonl", tmp_path / "l.jsonl"
    env.week6 = raw.games.filter((pl.col("season") == 2014) & (pl.col("week") == 6))
    env.kick = env.week6["kickoff_utc"][0]

    env.forecast_hook = None  # tests may set a callable(games) run before the forecast is returned
    env.forecast_empty = False

    def fake_forecast(games, st, client=None):
        env.forecast_calls.append(set(games["game_id"].to_list()))
        if env.forecast_hook:
            env.forecast_hook(games)
        if env.forecast_empty:
            return pl.DataFrame(schema=weather.WEATHER_SCHEMA)
        hours = weather.game_hours(games)
        return hours.with_columns(pl.lit(60.0).alias("temp_f"), pl.lit(5.0).alias("wind_mph"),
                                  pl.lit(0.0).alias("precip_in"), pl.lit("forecast").alias("source"))

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


def test_league_failure_keeps_seahawks_record(predict_env, monkeypatch, capsys):
    from seahawks_ml.pipeline import league
    from seahawks_ml.pipeline.history import read_history

    def boom(*a, **k):
        raise RuntimeError("league exploded")

    monkeypatch.setattr(league, "make_league_predictions", boom)
    env, changed = predict_env, []
    cli._predict(_args(env.kick - timedelta(hours=12)), changed)  # must not raise
    assert changed
    assert [r["run_type"] for r in read_history(env.hist)] == ["final_injury"]
    assert read_history(env.league) == []
    out = capsys.readouterr().out
    assert "WARNING" in out and "league exploded" in out


def test_forecast_failure_retries_seahawks_alone(predict_env):
    from seahawks_ml.pipeline.history import read_history
    env, changed = predict_env, []

    def fail_if_combined(games):
        if games.height > 1:
            raise RuntimeError("forecast api down")

    env.forecast_hook = fail_if_combined
    cli._predict(_args(env.kick - timedelta(hours=12)), changed)
    assert [r["run_type"] for r in read_history(env.hist)] == ["final_injury"]
    assert len(env.forecast_calls) == 2 and len(env.forecast_calls[1]) == 1
    # league games have no forecast this hour: outdoor ones are skipped, to retry next hour
    outdoor = set(env.week6.filter(~pl.col("roof").is_in(["dome", "closed"]))["game_id"].to_list())
    logged = {r["game_id"] for r in read_history(env.league)}
    assert outdoor and not (logged & outdoor)


def test_league_games_without_forecast_are_skipped(predict_env, capsys):
    from seahawks_ml.pipeline.history import read_history
    env = predict_env
    env.forecast_empty = True
    cli._predict(_args(env.kick - timedelta(hours=12)), [])
    outdoor = set(env.week6.filter(~pl.col("roof").is_in(["dome", "closed"]))["game_id"].to_list())
    logged = {r["game_id"] for r in read_history(env.league)}
    assert outdoor and not (logged & outdoor)
    assert "no forecast" in capsys.readouterr().out


def test_league_games_without_feature_row_are_reported(predict_env, monkeypatch, capsys):
    from seahawks_ml.features.build import build_features
    from seahawks_ml.pipeline.history import read_history
    env = predict_env
    monkeypatch.setattr(cli, "_build", lambda r, st, c: build_features(r, st).filter(
        pl.col("game_id") != "2014_06_SF_LA"))
    cli._predict(_args(env.kick - timedelta(hours=12)), [])
    assert "no feature row" in capsys.readouterr().out
    assert "2014_06_SF_LA" not in {r["game_id"] for r in read_history(env.league)}


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


def test_rating_grid_covers_toggles():
    from seahawks_ml.features.ratings import RatingParams

    grid = [RatingParams(*g) for g in cli.RATING_GRID]
    assert len(grid) == len(set(grid)) == 7
    for adj in (False, True):
        for extra in (False, True):
            assert RatingParams(0.6, 6.0, 2.0, adj, extra) in grid
    for reg, k, k_new in [(0.5, 4.0, 2.0), (0.7, 6.0, 3.0), (0.6, 4.0, 2.0)]:
        assert RatingParams(reg, k, k_new, True, True) in grid


def test_backtest_stage0_picks_best_rating_params(monkeypatch, tmp_path, capsys):
    import seahawks_ml.models.backtest as bt
    import seahawks_ml.models.store as store
    from seahawks_ml.features.ratings import RatingParams

    winner = RatingParams(0.6, 4.0, 2.0, True, True)
    written = {}
    monkeypatch.setattr(cli, "_load", lambda now: (None, None))
    monkeypatch.setattr(cli, "_build", lambda raw, stadiums, config: config)
    monkeypatch.setattr(bt, "walk_forward_log_loss",
                        lambda cfg, model, seasons: 0.6 if cfg.rating == winner else 0.7)
    monkeypatch.setattr(bt, "tune", lambda frame, seasons: (frame.model, []))
    monkeypatch.setattr(bt, "walk_forward", lambda frame, model, seasons: None)
    monkeypatch.setattr(bt, "score", lambda preds: {})
    monkeypatch.setattr(bt, "write_json", lambda path, d: written.update(d))
    monkeypatch.setattr(store, "save_config", lambda cfg: written.update(saved=cfg))
    cli.cmd_backtest(Namespace(now=None, tune_features=True))
    assert written["saved"].rating == winner
    assert any(t["rating"] == {**winner.__dict__} and t["log_loss"] == 0.6 for t in written["feature_trials"])
    assert f"best rating: {winner}" in capsys.readouterr().out
