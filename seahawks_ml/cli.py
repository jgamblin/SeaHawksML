"""Command-line entry point: python -m seahawks_ml.cli <command>."""

import argparse
from datetime import UTC, datetime

from seahawks_ml.config import FEATURES_PATH


def _now(args) -> datetime:
    return datetime.fromisoformat(args.now) if getattr(args, "now", None) else datetime.now(UTC)


def cmd_features(args) -> None:
    from seahawks_ml.data import load_raw
    from seahawks_ml.features.build import build_features
    from seahawks_ml.stadiums import load_stadiums

    stadiums = load_stadiums()
    raw = load_raw(stadiums, _now(args))
    frame = build_features(raw, stadiums)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(FEATURES_PATH)
    print(f"wrote {frame.height} games to {FEATURES_PATH}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="seahawks_ml")
    parser.add_argument("--now", help="override current time (ISO 8601, for testing)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("features", help="refresh data and rebuild the feature table").set_defaults(func=cmd_features)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
