"""Command-line entry point: python -m seahawks_ml.cli <command>.

Local (offseason / manual):  features, backtest, holdout
CI (in season):              retrain
"""

import argparse
from dataclasses import replace
from datetime import UTC, datetime

import polars as pl

from seahawks_ml.config import BACKTEST_SEASONS, FEATURES_PATH, HOLDOUT_SEASONS

RATING_GRID = [(0.5, 4.0, 2.0), (0.6, 4.0, 2.0), (0.7, 6.0, 3.0), (0.6, 6.0, 2.0)]
QB_PRIOR_GRID = [150.0, 250.0, 400.0]


def _now(args) -> datetime:
    return datetime.fromisoformat(args.now) if getattr(args, "now", None) else datetime.now(UTC)


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
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
