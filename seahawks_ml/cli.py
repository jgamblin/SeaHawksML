"""Command-line entry point: python -m seahawks_ml.cli <command>.

Local (offseason / manual):  features, backtest, holdout, simulate
CI (in season):              retrain, predict, build-site
"""

import argparse
import json
import os
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta

import polars as pl

from seahawks_ml.config import (
    BACKTEST_SEASONS,
    FEATURES_PATH,
    HOLDOUT_SEASONS,
    LEAGUE_PATH,
    PREDICTIONS_PATH,
    SEASON_SIM_PATH,
    TEAM,
)

# RatingParams fields: (prior_regression, prior_games, prior_games_new_coach, opponent_adjust, extra_stats).
# The previous best shrinkage with every toggle combination, plus other shrinkage with both on.
RATING_GRID = [
    *[(0.6, 6.0, 2.0, adj, extra) for adj in (False, True) for extra in (False, True)],
    (0.5, 4.0, 2.0, True, True), (0.7, 6.0, 3.0, True, True), (0.6, 4.0, 2.0, True, True),
]
QB_PRIOR_GRID = [150.0, 250.0, 400.0]
FORECAST_HORIZON = timedelta(days=16)


def _now(args) -> datetime:
    """Current time in UTC. A naive --now is taken as UTC; an aware one is converted."""
    raw = getattr(args, "now", None)
    if not raw:
        return datetime.now(UTC)
    dt = datetime.fromisoformat(raw)
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _set_output(key: str, value: str) -> None:
    """Expose a value to later GitHub Actions steps (no-op locally)."""
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            f.write(f"{key}={value}\n")


def _load(now: datetime, games: pl.DataFrame | None = None):
    from seahawks_ml.data import load_raw
    from seahawks_ml.stadiums import load_stadiums

    stadiums = load_stadiums()
    return stadiums, load_raw(stadiums, now, games)


def _build(raw, stadiums, config) -> pl.DataFrame:
    from seahawks_ml.features.build import build_features

    return build_features(raw, stadiums, config.rating, config.qb)


def cmd_features(args) -> None:
    from seahawks_ml.models.store import ProjectConfig, load_config

    stadiums, raw = _load(_now(args))
    try:
        config = load_config()
    except FileNotFoundError:
        config = ProjectConfig()
    frame = _build(raw, stadiums, config)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(FEATURES_PATH)
    print(f"wrote {frame.height} games to {FEATURES_PATH}")


def cmd_backtest(args) -> None:
    """Local only: tune feature params + model config by walk-forward, lock models/config.json."""
    from seahawks_ml.features.qb import QBParams
    from seahawks_ml.features.ratings import RatingParams
    from seahawks_ml.models.backtest import score, tune, walk_forward, walk_forward_log_loss, write_json
    from seahawks_ml.models.pipeline import ModelConfig
    from seahawks_ml.models.store import BACKTEST_PATH, ProjectConfig, save_config

    stadiums, raw = _load(_now(args))
    seasons = list(BACKTEST_SEASONS)
    config = ProjectConfig()
    feature_trials = []
    if args.tune_features:
        print("stage 0a: team ratings (shrinkage, opponent adjustment, extra stats)")
        best = None
        for fields in RATING_GRID:
            cand = replace(config, rating=RatingParams(*fields))
            ll = walk_forward_log_loss(_build(raw, stadiums, cand), ModelConfig(), seasons)
            print(f"  {ll:.5f}  {cand.rating}")
            feature_trials.append({"stage": "rating", "rating": asdict(cand.rating), "log_loss": ll})
            if best is None or ll < best[1]:
                best = (cand, ll)
        config = best[0]
        print(f"  best rating: {config.rating}")
        print("stage 0b: QB prior strength")
        for prior in QB_PRIOR_GRID:
            cand = replace(config, qb=QBParams(prior_dropbacks=prior))
            ll = walk_forward_log_loss(_build(raw, stadiums, cand), ModelConfig(), seasons)
            print(f"  {ll:.5f}  {cand.qb}")
            feature_trials.append({"stage": "qb", "qb": asdict(cand.qb), "log_loss": ll})
            if ll < best[1]:
                best = (cand, ll)
        config = best[0]
    frame = _build(raw, stadiums, config)
    model_config, trials = tune(frame, seasons)
    config = replace(config, model=model_config)
    save_config(config)
    preds = walk_forward(frame, model_config, seasons)
    write_json(BACKTEST_PATH, {"seasons": seasons, "config": config.to_dict(),
                               "score": score(preds), "trials": trials,
                               "feature_trials": feature_trials})
    print(f"locked config {config.fingerprint()} -> models/config.json; report -> {BACKTEST_PATH}")


def cmd_holdout(args) -> None:
    """Local only, run once after tuning: score the locked config on 2024-2025."""
    from seahawks_ml.models.backtest import score, walk_forward, write_json
    from seahawks_ml.models.store import HOLDOUT_PATH, load_config

    if HOLDOUT_PATH.exists() and not args.force:
        raise SystemExit(f"{HOLDOUT_PATH} exists - the holdout is evaluated once. Use --force to overwrite.")
    config = load_config()
    stadiums, raw = _load(_now(args))
    frame = _build(raw, stadiums, config)
    preds = walk_forward(frame, config.model, list(HOLDOUT_SEASONS))
    write_json(HOLDOUT_PATH, {"seasons": list(HOLDOUT_SEASONS), "config": config.to_dict(), "score": score(preds)})
    print(f"holdout written to {HOLDOUT_PATH}")


def cmd_retrain(args) -> None:
    """CI: refit final weights with the locked config. No tuning."""
    from seahawks_ml.models.backtest import write_json
    from seahawks_ml.models.store import METRICS_PATH, load_config, save_model
    from seahawks_ml.models.train import train_production

    now = _now(args)
    config = load_config()
    stadiums, raw = _load(now)
    frame = _build(raw, stadiums, config)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(FEATURES_PATH)
    model, meta = train_production(frame, config, now)
    save_model(model)
    write_json(METRICS_PATH, meta)
    print(f"trained {meta['model_version']} on {meta['n_games']} games")


def _warn_league(what: str) -> None:
    """Report a league-step failure without aborting the run (the Seahawks record is already kept)."""
    import traceback

    print(f"WARNING: league {what} failed; Seahawks prediction is unaffected. Will retry next hour.")
    print("".join(traceback.format_exc(limit=-3)).rstrip())


def _warn_sim() -> None:
    """Report a season-simulation failure without aborting the run."""
    import traceback

    print("WARNING: season simulation failed; predictions are unaffected. Will retry next hour.")
    print("".join(traceback.format_exc(limit=-3)).rstrip())


def _run_season_sim(games: pl.DataFrame, frame: pl.DataFrame, model, now: datetime, version: str) -> bool:
    """Simulate the current season with the current model and log today's snapshot. True if appended."""
    from seahawks_ml.data import current_season
    from seahawks_ml.ingest import nflverse
    from seahawks_ml.pipeline import season_sim

    season = current_season(games, now)
    probs = season_sim.remaining_game_probs(frame, model, season)
    rec = season_sim.simulate_season(games, probs, nflverse.load_teams(), season, now=now)
    rec["model_version"] = version
    if not season_sim.append_snapshot(rec, SEASON_SIM_PATH):
        print("season simulation for today already logged; skipped")
        return False
    print(f"season sim {season}: {rec['wins_mean']:.1f} wins, P(playoffs) {rec['p_playoffs']:.0%}, "
          f"P(division) {rec['p_division']:.0%}, P(#1 seed) {rec['p_top_seed']:.0%}")
    return True


def _fetch_forecast(targets: pl.DataFrame, stadiums, seahawks_id: str | None) -> pl.DataFrame:
    """Forecast for the target games; if the combined fetch fails, retry for the Seahawks game alone
    (league games then have no forecast this hour and are skipped by the caller)."""
    from seahawks_ml.ingest.weather import WEATHER_SCHEMA, forecast_for_games

    try:
        return forecast_for_games(targets, stadiums)
    except Exception as exc:
        print(f"WARNING: forecast fetch failed ({type(exc).__name__}: {exc})")
        if seahawks_id is None:
            return pl.DataFrame(schema=WEATHER_SCHEMA)
        print(f"retrying forecast for {seahawks_id} alone")
        return forecast_for_games(targets.filter(pl.col("game_id") == seahawks_id), stadiums)


def _without_forecast(games: pl.DataFrame, weather: pl.DataFrame) -> set[str]:
    """Outdoor games with any game-window hour lacking weather data."""
    from seahawks_ml.ingest.weather import game_hours

    have = weather.filter(pl.col("temp_f").is_not_null()).select("stadium_id", "time_utc")
    missing = set()
    for g in games.filter(~pl.col("roof").is_in(["dome", "closed"])).iter_rows(named=True):
        hours = game_hours(games.filter(pl.col("game_id") == g["game_id"]))
        if hours.join(have, on=["stadium_id", "time_utc"], how="anti").height:
            missing.add(g["game_id"])
    return missing


def cmd_predict(args) -> None:
    """CI: if a run is due for the next Seahawks game, predict and log it."""
    changed: list[bool] = []  # non-empty once anything was appended to the history
    try:
        _predict(args, changed)
    finally:
        _set_output("changed", "true" if changed else "false")


def _predict(args, changed: list[bool]) -> None:
    from seahawks_ml.features.base import prepare_games
    from seahawks_ml.ingest import nflverse
    from seahawks_ml.models.store import METRICS_PATH, check_model_matches, load_config, load_model
    from seahawks_ml.pipeline.gate import due_run, next_game, previous_game_ready, previous_kickoff
    from seahawks_ml.pipeline.history import append_record, read_history, runs_done
    from seahawks_ml.pipeline.league import (
        due_league_games,
        make_league_predictions,
        new_league_results,
        validate_league,
    )
    from seahawks_ml.pipeline.predict import latest_injury_week, make_prediction, single_game_row
    from seahawks_ml.pipeline.results import new_results
    from seahawks_ml.pipeline.season_sim import has_snapshot_for, season_in_progress
    from seahawks_ml.stadiums import load_stadiums

    now = _now(args)
    stadiums = load_stadiums()
    history = read_history(PREDICTIONS_PATH)
    league_log = read_history(LEAGUE_PATH)
    games = prepare_games(nflverse.load_schedules(), stadiums)

    for rec in new_results(history, games, now):
        append_record(rec, PREDICTIONS_PATH)
        changed.append(True)
        print(f"result recorded: {rec['game_id']} {rec['margin_seahawks']:+d}")
    history = read_history(PREDICTIONS_PATH)
    try:
        for rec in new_league_results(league_log, games, now):
            append_record(rec, LEAGUE_PATH, validate_league)
            changed.append(True)
            print(f"league result recorded: {rec['game_id']} {rec['margin_home']:+d}")
    except Exception:
        _warn_league("result recording")
    league_log = read_history(LEAGUE_PATH)

    game = next_game(games, TEAM, now)
    run_type = None
    if game is None:
        print("no upcoming Seahawks game")
    else:
        run_type = args.run_type or due_run(now, game["kickoff_utc"], runs_done(history, game["game_id"]),
                                            previous_kickoff(games, TEAM, now))
        if run_type is None:
            print(f"no run due for {game['game_id']} (kickoff {game['kickoff_utc']:%Y-%m-%d %H:%M} UTC)")
    logged = {r["game_id"] for r in league_log if r["type"] == "league_prediction"}
    league_ids = due_league_games(games, now, logged)
    # The season simulation runs on any run that loads data, and at least once per UTC day in season.
    sim_due = season_in_progress(games, now) and not has_snapshot_for(read_history(SEASON_SIM_PATH), now)
    if run_type is None and not league_ids:
        if sim_due:  # nothing else to do: a failure here (e.g. a data download) must not fail the job
            print("season simulation due (none logged today)")
            try:
                if _simulate(now, games):
                    changed.append(True)
            except Exception:
                _warn_sim()
        return

    if run_type:
        print(f"{run_type} run for {game['game_id']}")
    if league_ids:
        print(f"league run for {len(league_ids)} games")
    config = load_config()
    stadiums, raw = _load(now, games)
    if run_type and not args.run_type and not previous_game_ready(raw.games, raw.team_epa, TEAM, now):
        print("previous game data not in yet; will retry next hour")
        run_type = None
        sim_due = False  # its result would be simulated as if unplayed
        if not league_ids:
            return
    target_ids = list(league_ids) + ([game["game_id"]] if run_type else [])
    targets = raw.games.filter(pl.col("game_id").is_in(target_ids) & (pl.col("kickoff_utc") - now <= FORECAST_HORIZON))
    if targets.height:
        forecast = _fetch_forecast(targets, stadiums, game["game_id"] if run_type else None)
        raw = replace(raw, weather=pl.concat([raw.weather, forecast]).unique(["stadium_id", "time_utc"], keep="last"))
    if league_ids:
        no_forecast = _without_forecast(raw.games.filter(pl.col("game_id").is_in(league_ids)), raw.weather)
        if no_forecast:
            print(f"no forecast yet for league games {sorted(no_forecast)}; skipping them this hour (retry next hour)")
            league_ids = [g for g in league_ids if g not in no_forecast]
    frame = _build(raw, stadiums, config)
    model = load_model()
    metrics = json.loads(METRICS_PATH.read_text())
    check_model_matches(metrics, config)
    version = metrics["model_version"]

    if run_type:
        row = single_game_row(frame, game["game_id"])
        record = make_prediction(row, model, run_type, now, version,
                                 latest_injury_week(raw.injuries, game["season"]))
        append_record(record, PREDICTIONS_PATH)
        changed.append(True)
        print(f"SEA win probability {record['p_seahawks']:.1%}, margin {record['margin_seahawks']:+.1f}")
    if league_ids:
        try:
            rows = frame.filter(pl.col("game_id").is_in(league_ids))
            no_row = [g for g in league_ids if g not in set(rows["game_id"].to_list())]
            if no_row:
                print(f"league games with no feature row (retry next hour): {no_row}")
            records = make_league_predictions(rows, model, now, version)
            for rec in records:
                append_record(rec, LEAGUE_PATH, validate_league)
                changed.append(True)
            print(f"logged {len(records)} league predictions")
        except Exception:
            _warn_league("predictions")
    if sim_due:
        print("season simulation due (none logged today)")
        try:
            if _run_season_sim(raw.games, frame, model, now, version):
                changed.append(True)
        except Exception:
            _warn_sim()


def _simulate(now: datetime, games: pl.DataFrame | None = None) -> bool:
    """Load data, build features and run the season simulation. True if a snapshot was appended."""
    from seahawks_ml.models.store import METRICS_PATH, check_model_matches, load_config, load_model

    config = load_config()
    stadiums, raw = _load(now, games)
    frame = _build(raw, stadiums, config)
    model = load_model()
    metrics = json.loads(METRICS_PATH.read_text())
    check_model_matches(metrics, config)
    return _run_season_sim(raw.games, frame, model, now, metrics["model_version"])


def cmd_simulate(args) -> None:
    """Simulate the rest of the season with the current model; log at most one snapshot per UTC day."""
    _simulate(_now(args))


def cmd_build_site(args) -> None:
    from seahawks_ml.site.build import build_site

    print(f"site written to {build_site(_now(args))}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="seahawks_ml")
    parser.add_argument("--now", help="override current time (ISO 8601, for testing)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("features", help="refresh data and rebuild the feature table").set_defaults(func=cmd_features)
    bt = sub.add_parser("backtest", help="LOCAL: tune by walk-forward and lock models/config.json")
    bt.add_argument("--tune-features", action="store_true", help="also tune rating/QB shrinkage (slow)")
    bt.set_defaults(func=cmd_backtest)
    ho = sub.add_parser("holdout", help="LOCAL: evaluate locked config on 2024-2025 (once)")
    ho.add_argument("--force", action="store_true")
    ho.set_defaults(func=cmd_holdout)
    sub.add_parser("retrain", help="CI: refit with locked config").set_defaults(func=cmd_retrain)
    pr = sub.add_parser("predict", help="CI: predict the next Seahawks game if a run is due")
    pr.add_argument("--run-type", choices=["midweek", "final_injury", "gameday"], help="force a run type")
    pr.set_defaults(func=cmd_predict)
    sub.add_parser("simulate", help="simulate the rest of the season; log today's snapshot").set_defaults(
        func=cmd_simulate)
    sub.add_parser("build-site", help="render public/").set_defaults(func=cmd_build_site)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
