"""Command-line entry point: python -m seahawks_ml.cli <command>.

Local (offseason / manual):  features, backtest, holdout
CI (in season):              retrain, predict, build-site
"""

import argparse
import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import polars as pl

from seahawks_ml.config import BACKTEST_SEASONS, FEATURES_PATH, HOLDOUT_SEASONS, TEAM

RATING_GRID = [(0.5, 4.0, 2.0), (0.6, 4.0, 2.0), (0.7, 6.0, 3.0), (0.6, 6.0, 2.0)]
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


def _load(now: datetime):
    from seahawks_ml.data import load_raw
    from seahawks_ml.stadiums import load_stadiums

    stadiums = load_stadiums()
    return stadiums, load_raw(stadiums, now)


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
    if args.tune_features:
        print("stage 0a: team-rating shrinkage")
        best = None
        for reg, k, k_new in RATING_GRID:
            cand = replace(config, rating=RatingParams(reg, k, k_new))
            ll = walk_forward_log_loss(_build(raw, stadiums, cand), ModelConfig(), seasons)
            print(f"  {ll:.5f}  {cand.rating}")
            if best is None or ll < best[1]:
                best = (cand, ll)
        config = best[0]
        print("stage 0b: QB prior strength")
        for prior in QB_PRIOR_GRID:
            cand = replace(config, qb=QBParams(prior_dropbacks=prior))
            ll = walk_forward_log_loss(_build(raw, stadiums, cand), ModelConfig(), seasons)
            print(f"  {ll:.5f}  {cand.qb}")
            if ll < best[1]:
                best = (cand, ll)
        config = best[0]
    frame = _build(raw, stadiums, config)
    model_config, trials = tune(frame, seasons)
    config = replace(config, model=model_config)
    save_config(config)
    preds = walk_forward(frame, model_config, seasons)
    write_json(BACKTEST_PATH, {"seasons": seasons, "config": config.to_dict(),
                               "score": score(preds), "trials": trials})
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


def cmd_predict(args) -> None:
    """CI: if a run is due for the next Seahawks game, predict and log it."""
    from seahawks_ml.features.base import prepare_games
    from seahawks_ml.ingest import nflverse
    from seahawks_ml.ingest.weather import forecast_for_games
    from seahawks_ml.models.store import METRICS_PATH, check_model_matches, load_config, load_model
    from seahawks_ml.pipeline.gate import due_run, next_game, previous_game_ready, previous_kickoff
    from seahawks_ml.pipeline.history import append_record, read_history, runs_done
    from seahawks_ml.pipeline.predict import latest_injury_week, make_prediction
    from seahawks_ml.pipeline.results import new_results
    from seahawks_ml.stadiums import load_stadiums

    now = _now(args)
    stadiums = load_stadiums()
    history = read_history()
    games = prepare_games(nflverse.load_schedules(), stadiums)

    results = new_results(history, games, now)
    for rec in results:
        append_record(rec)
        print(f"result recorded: {rec['game_id']} {rec['margin_seahawks']:+d}")
    history = read_history()
    _set_output("changed", "true" if results else "false")

    game = next_game(games, TEAM, now)
    if game is None:
        print("no upcoming Seahawks game")
        return
    run_type = args.run_type or due_run(now, game["kickoff_utc"], runs_done(history, game["game_id"]),
                                        previous_kickoff(games, TEAM, now))
    if run_type is None:
        print(f"no run due for {game['game_id']} (kickoff {game['kickoff_utc']:%Y-%m-%d %H:%M} UTC)")
        return

    print(f"{run_type} run for {game['game_id']}")
    config = load_config()
    stadiums, raw = _load(now)
    if not args.run_type and not previous_game_ready(raw.games, raw.team_epa, TEAM, now):
        print("previous game data not in yet; will retry next hour")
        return
    target = raw.games.filter(pl.col("game_id") == game["game_id"])
    if game["kickoff_utc"] - now <= FORECAST_HORIZON:
        forecast = forecast_for_games(target, stadiums)
        raw = replace(raw, weather=pl.concat([raw.weather, forecast]).unique(["stadium_id", "time_utc"], keep="last"))
    frame = _build(raw, stadiums, config)
    row = frame.filter(pl.col("game_id") == game["game_id"])
    model = load_model()
    metrics = json.loads(METRICS_PATH.read_text())
    check_model_matches(metrics, config)
    version = metrics["model_version"]
    record = make_prediction(row, model, run_type, now, version,
                             latest_injury_week(raw.injuries, game["season"]))
    append_record(record)
    print(f"SEA win probability {record['p_seahawks']:.1%}, margin {record['margin_seahawks']:+.1f}")
    _set_output("changed", "true")


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
    sub.add_parser("build-site", help="render public/").set_defaults(func=cmd_build_site)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
