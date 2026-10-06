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

    env.sim = tmp_path / "sim.jsonl"
    monkeypatch.setattr(cli, "PREDICTIONS_PATH", env.hist)
    monkeypatch.setattr(cli, "LEAGUE_PATH", env.league)
    monkeypatch.setattr(cli, "SEASON_SIM_PATH", env.sim)
    teams = pl.DataFrame({"team": ["SEA", "SF", "LA", "ARI"], "conf": ["NFC"] * 4, "division": ["NFC West"] * 4})
    monkeypatch.setattr(nflverse, "load_teams", lambda: teams)
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


def test_league_games_wait_for_previous_game_data(predict_env, monkeypatch, capsys):
    from seahawks_ml.pipeline.history import read_history
    env = predict_env
    prev = env.raw.games.filter((pl.col("season") == 2014) & (pl.col("week") == 5)
                                & ((pl.col("home_team") == "ARI") | (pl.col("away_team") == "ARI")))["game_id"][0]
    team_epa = env.raw.team_epa.filter(~((pl.col("game_id") == prev) & (pl.col("team") == "ARI")))
    monkeypatch.setattr(cli, "_load", lambda now, games=None: (
        __import__("seahawks_ml.stadiums", fromlist=["x"]).load_stadiums(), replace(env.raw, team_epa=team_epa)))
    cli._predict(_args(env.kick - timedelta(hours=12)), [])
    out = capsys.readouterr().out
    held = set(env.week6.filter((pl.col("home_team") == "ARI") | (pl.col("away_team") == "ARI"))["game_id"].to_list())
    logged = {r["game_id"] for r in read_history(env.league)}
    assert held and not (logged & held) and logged
    assert "retry next hour" in out and sorted(held)[0] in out


def test_predict_league_only_when_seahawks_already_run(predict_env):
    from seahawks_ml.pipeline.history import read_history
    env, changed = predict_env, []
    now = env.kick - timedelta(hours=12)
    cli._predict(_args(now), [])
    env.league.unlink()  # pretend only the league log is missing
    cli._predict(_args(now + timedelta(hours=1)), changed)
    assert changed and len(read_history(env.league)) == 2
    assert [r["run_type"] for r in read_history(env.hist)] == ["final_injury"]  # no second Seahawks run


def _sim_snapshot(now):
    return {"as_of": now.isoformat(), "season": 2014, "team": "SEA", "n_sims": 1, "wins_mean": 3.0,
            "wins_p10": 3.0, "wins_p50": 3.0, "wins_p90": 3.0, "win_dist": [{"wins": 3, "prob": 1.0}],
            "p_playoffs": 1.0, "p_division": 1.0, "p_top_seed": 1.0, "record_now": "3-2-0",
            "model_version": "vtest"}


def test_predict_nothing_due_loads_nothing(predict_env):
    from seahawks_ml.pipeline.season_sim import append_snapshot
    env, changed = predict_env, []
    now = env.kick - timedelta(hours=36)
    append_snapshot(_sim_snapshot(now.replace(hour=0)), env.sim)  # today's simulation already done
    cli._predict(_args(now), changed)
    assert not changed and env.loads == []


def test_predict_runs_daily_season_sim_when_nothing_else_due(predict_env):
    from seahawks_ml.pipeline.history import read_history
    env, changed = predict_env, []
    now = (env.kick - timedelta(hours=36)).replace(hour=14)  # a SIM_ONLY_HOURS slot
    cli._predict(_args(now), changed)
    assert changed and len(env.loads) == 1
    assert read_history(env.hist) == [] and read_history(env.league) == []
    [snap] = read_history(env.sim)
    assert snap["season"] == 2014 and snap["team"] == "SEA" and snap["model_version"] == "vtest"
    assert snap["n_sims"] == 10_000 and snap["as_of"] == now.isoformat()
    assert snap["record_now"].count("-") == 2 and 0 <= snap["p_playoffs"] <= 1
    changed = []
    cli._predict(_args(now + timedelta(hours=1)), changed)  # same UTC day: quick exit
    assert not changed and len(env.loads) == 1


def test_sim_only_run_waits_for_sim_hours(predict_env):
    env, changed = predict_env, []
    base = (env.kick - timedelta(hours=36)).replace(hour=0)
    for hour in (9, 11, 13):
        cli._predict(_args(base.replace(hour=hour)), changed)
    assert not changed and env.loads == [] and not env.sim.exists()
    cli._predict(_args(base.replace(hour=10)), changed)
    assert changed and len(env.loads) == 1


def _previous_game_unplayed(env, monkeypatch):
    """Make SEA's previous game (week 5) look unplayed in the loaded schedule."""
    from seahawks_ml.stadiums import load_stadiums
    games = env.raw.games.with_columns(
        pl.when(pl.col("game_id").str.starts_with("2014_05_") & pl.col("game_id").str.contains("SEA"))
        .then(None).otherwise(pl.col("margin")).alias("margin"))
    monkeypatch.setattr(cli, "_load", lambda now, games_=None: (load_stadiums(), replace(env.raw, games=games)))


def test_sim_skipped_when_previous_game_has_no_result(predict_env, monkeypatch, capsys):
    env, changed = predict_env, []
    _previous_game_unplayed(env, monkeypatch)
    cli._predict(_args((env.kick - timedelta(hours=36)).replace(hour=14)), changed)  # sim-only path
    assert not changed and not env.sim.exists() and "no final result" in capsys.readouterr().out
    # forced prediction run (skips previous_game_ready) must not simulate either
    cli._predict(_args(env.kick - timedelta(hours=12), run_type="final_injury"), changed)
    assert changed and not env.sim.exists() and "no final result" in capsys.readouterr().out


def test_predict_skips_season_sim_outside_season(predict_env):
    env, changed = predict_env, []
    cli._predict(_args(env.kick + timedelta(days=30)), changed)
    assert not changed and env.loads == [] and not env.sim.exists()


def test_season_sim_runs_with_seahawks_prediction(predict_env):
    from seahawks_ml.pipeline.history import read_history
    env = predict_env
    cli._predict(_args(env.kick - timedelta(hours=12)), [])
    assert len(read_history(env.sim)) == 1 and len(env.loads) == 1


def test_season_sim_failure_keeps_seahawks_record(predict_env, monkeypatch, capsys):
    from seahawks_ml.pipeline import season_sim
    from seahawks_ml.pipeline.history import read_history

    def boom(*a, **k):
        raise RuntimeError("sim exploded")

    monkeypatch.setattr(season_sim, "simulate_season", boom)
    env, changed = predict_env, []
    cli._predict(_args(env.kick - timedelta(hours=12)), changed)
    assert changed and [r["run_type"] for r in read_history(env.hist)] == ["final_injury"]
    assert not env.sim.exists()
    out = capsys.readouterr().out
    assert "WARNING" in out and "sim exploded" in out


def test_simulate_command_appends_once_per_day(predict_env, capsys):
    from seahawks_ml.pipeline.history import read_history
    env = predict_env
    now = env.kick - timedelta(hours=36)
    cli.cmd_simulate(_args(now))
    cli.cmd_simulate(_args(now + timedelta(hours=2)))
    assert len(read_history(env.sim)) == 1 and len(env.loads) == 2
    assert "already" in capsys.readouterr().out


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


def test_sim_only_run_survives_data_load_failure(predict_env, monkeypatch, capsys):
    env, changed = predict_env, []

    def boom(now, games=None):
        raise RuntimeError("nflverse down")

    monkeypatch.setattr(cli, "_load", boom)
    cli._predict(_args((env.kick - timedelta(hours=36)).replace(hour=14)), changed)  # must not raise
    assert not changed and not env.sim.exists()
    assert "nflverse down" in capsys.readouterr().out


def _holdout_env(monkeypatch, tmp_path, existing=None):
    import json

    from seahawks_ml.models import backtest, store
    path = tmp_path / "holdout.json"
    if existing is not None:
        path.write_text(json.dumps(existing))
    cfg = store.ProjectConfig()
    monkeypatch.setattr(store, "HOLDOUT_PATH", path)
    monkeypatch.setattr(store, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "_load", lambda now: (None, None))
    monkeypatch.setattr(cli, "_build", lambda raw, st, c: None)
    monkeypatch.setattr(backtest, "walk_forward", lambda frame, mc, seasons: None)
    monkeypatch.setattr(backtest, "score", lambda preds: {"n": 0})
    return path, cfg


def test_holdout_first_run_marks_not_previously_viewed(monkeypatch, tmp_path):
    import json
    path, _ = _holdout_env(monkeypatch, tmp_path)
    cli.cmd_holdout(Namespace(force=False, now=None))
    out = json.loads(path.read_text())
    assert out["previously_viewed"] is False
    assert "previous_config_fingerprint" not in out
    assert datetime.fromisoformat(out["evaluated_at"]).utcoffset().total_seconds() == 0


def test_holdout_force_rerun_records_previous_view(monkeypatch, tmp_path):
    import json

    from seahawks_ml.models.store import ProjectConfig
    old = replace(ProjectConfig(), rating=replace(ProjectConfig().rating, prior_games=9.0))
    path, cfg = _holdout_env(monkeypatch, tmp_path, existing={"seasons": [2024], "config": old.to_dict(),
                                                              "score": {}})
    cli.cmd_holdout(Namespace(force=True, now=None))
    out = json.loads(path.read_text())
    assert out["previously_viewed"] is True
    assert out["previous_config_fingerprint"] == old.fingerprint() != cfg.fingerprint()
    assert out["evaluated_at"]


def test_holdout_refuses_overwrite_without_force(monkeypatch, tmp_path):
    path, _ = _holdout_env(monkeypatch, tmp_path, existing={"config": {}})
    with pytest.raises(SystemExit):
        cli.cmd_holdout(Namespace(force=False, now=None))


def test_build_site_falls_back_when_schedule_download_fails(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace

    from seahawks_ml.ingest import nflverse
    from seahawks_ml.site import build

    def boom():
        raise OSError("offline")

    seen = {}
    monkeypatch.setattr(nflverse, "load_schedules", boom)
    monkeypatch.setattr(build, "build_site", lambda now, schedule=None: seen.update(schedule=schedule) or tmp_path)
    cli.cmd_build_site(SimpleNamespace(now="2026-10-10T00:00:00"))
    assert seen["schedule"] is None
    assert "warning: schedule unavailable" in capsys.readouterr().out
