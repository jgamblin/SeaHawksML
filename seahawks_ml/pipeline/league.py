"""League-wide live predictions (predictions/league.jsonl): one append-only prediction per game,
made in the same window as the Seahawks final-injury run, plus result records and a scorecard."""

import math
from datetime import datetime

import numpy as np
import polars as pl

from seahawks_ml.config import LEAGUE_PATH  # noqa: F401  (re-exported for callers)
from seahawks_ml.features.columns import FEATURE_COLUMNS
from seahawks_ml.features.elo import elo_win_prob
from seahawks_ml.models.metrics import log_loss, outcome, summarize
from seahawks_ml.models.pipeline import FittedModel
from seahawks_ml.pipeline.gate import RUN_WINDOWS

PREDICTION_KEYS = {
    "type", "game_id", "season", "week", "predicted_at", "kickoff_utc", "home_team", "away_team",
    "p_home", "margin_home", "p_vegas_home", "p_elo_home", "model_version",
}
RESULT_KEYS = {"type", "game_id", "recorded_at", "home_score", "away_score", "margin_home"}


def _finite(x) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)


def _prob(x) -> bool:
    return _finite(x) and 0 < x < 1


def validate_league(record: dict) -> None:
    expected = {"league_prediction": PREDICTION_KEYS, "league_result": RESULT_KEYS}.get(record.get("type"))
    if expected is None:
        raise ValueError(f"unknown league record type {record.get('type')!r}")
    if set(record) != expected:
        raise ValueError(f"record keys mismatch: missing {expected - set(record)}, extra {set(record) - expected}")
    if record["type"] == "league_prediction":
        if not _prob(record["p_home"]):
            raise ValueError(f"p_home must be in (0, 1), got {record['p_home']!r}")
        if not (_finite(record["margin_home"]) and _finite(record["p_elo_home"])):
            raise ValueError("margin_home and p_elo_home must be finite numbers")
        if record["p_vegas_home"] is not None and not _prob(record["p_vegas_home"]):
            raise ValueError(f"p_vegas_home must be None or in (0, 1), got {record['p_vegas_home']!r}")


def due_league_games(games: pl.DataFrame, now: datetime, logged_ids: set[str]) -> list[str]:
    opens, closes = RUN_WINDOWS["final_injury"]
    window = games.filter((pl.col("kickoff_utc") - opens <= now) & (now < pl.col("kickoff_utc") - closes)
                          & ~pl.col("game_id").is_in(list(logged_ids)))
    return window.sort("kickoff_utc", "game_id")["game_id"].to_list()


def make_league_predictions(frame_rows: pl.DataFrame, model: FittedModel, now: datetime,
                            model_version: str) -> list[dict]:
    if not frame_rows.height:
        return []
    cols = [pl.col(c).cast(pl.Float64) for c in FEATURE_COLUMNS]
    bad_row = frame_rows.select(pl.any_horizontal(*[c.is_null() | c.is_nan() | c.is_infinite() for c in cols]))
    unusable = set(frame_rows.filter(bad_row.to_series())["game_id"].to_list())
    if unusable:
        print(f"league: skipping games with missing/non-finite features (retry next hour): {sorted(unusable)}")
        frame_rows = frame_rows.filter(~pl.col("game_id").is_in(list(unusable)))
        if not frame_rows.height:
            return []
    out = model.predict(frame_rows)
    spreads = frame_rows["spread_line"].to_list()
    known = [s is not None and math.isfinite(s) for s in spreads]
    vegas = np.full(len(spreads), np.nan)
    if any(known):
        vegas[known] = model.vegas_prob(np.array([s for s, k in zip(spreads, known) if k], dtype=float))
    recs = []
    for i, g in enumerate(frame_rows.iter_rows(named=True)):
        if g["elo_home_pre"] is None or g["elo_away_pre"] is None:
            print(f"league: skipping {g['game_id']}: missing Elo (retry next hour)")
            continue
        elo = elo_win_prob(g["elo_home_pre"] - g["elo_away_pre"], g["neutral"])
        values = [out["p_win"][i], out["margin"][i], elo] + ([vegas[i]] if known[i] else [])
        if not all(math.isfinite(float(v)) for v in values):
            print(f"league: skipping {g['game_id']}: non-finite model output (retry next hour)")
            continue
        rec = {
            "type": "league_prediction", "game_id": g["game_id"], "season": g["season"], "week": g["week"],
            "predicted_at": now.isoformat(), "kickoff_utc": g["kickoff_utc"].isoformat(),
            "home_team": g["home_team"], "away_team": g["away_team"],
            "p_home": round(float(out["p_win"][i]), 4), "margin_home": round(float(out["margin"][i]), 2),
            "p_vegas_home": round(float(vegas[i]), 4) if known[i] else None,
            "p_elo_home": round(float(elo), 4), "model_version": model_version,
        }
        try:
            validate_league(rec)  # e.g. a probability that rounds to exactly 0 or 1
        except ValueError as exc:
            print(f"league: skipping {g['game_id']}: {exc} (retry next hour)")
            continue
        recs.append(rec)
    return recs


def new_league_results(league_log: list[dict], games: pl.DataFrame, now: datetime) -> list[dict]:
    predicted = {r["game_id"] for r in league_log if r["type"] == "league_prediction"}
    recorded = {r["game_id"] for r in league_log if r["type"] == "league_result"}
    finished = games.filter(pl.col("game_id").is_in(list(predicted - recorded)) & pl.col("margin").is_not_null())
    return [{"type": "league_result", "game_id": g["game_id"], "recorded_at": now.isoformat(),
             "home_score": g["home_score"], "away_score": g["away_score"], "margin_home": g["margin"]}
            for g in finished.sort("game_id").iter_rows(named=True)]


def league_scorecard(league_log: list[dict], season: int | None) -> dict:
    """Model vs Vegas vs Elo on completed games of `season`. Compared on the common set of games that
    have a Vegas line (all games if none do); `n_all` counts every completed game."""
    results = {r["game_id"]: r for r in league_log if r["type"] == "league_result"}
    done = sorted((p for p in league_log if p["type"] == "league_prediction"
                   and p["season"] == season and p["game_id"] in results),
                  key=lambda p: (p["week"], p["kickoff_utc"], p["game_id"]))
    common = [p for p in done if p["p_vegas_home"] is not None] or done
    has_vegas = any(p["p_vegas_home"] is not None for p in common)
    empty = {"season": season, "n": 0, "n_all": len(done), "model": None, "vegas": None, "elo": None,
             "by_week": {"weeks": [], "model": [], "vegas": [], "elo": []}}
    if not common:
        return empty
    margins = np.array([results[p["game_id"]]["margin_home"] for p in common], dtype=float)
    cols = {"model": "p_home", "vegas": "p_vegas_home", "elo": "p_elo_home"}
    probs = {k: np.array([np.nan if p[c] is None else p[c] for p in common], dtype=float) for k, c in cols.items()}
    pred_margin = np.array([p["margin_home"] for p in common], dtype=float)
    card = {"season": season, "n": len(common), "n_all": len(done),
            "model": summarize(probs["model"], pred_margin, margins),
            "vegas": summarize(probs["vegas"], None, margins) if has_vegas else None,
            "elo": summarize(probs["elo"], None, margins)}
    y, weeks = outcome(margins), np.array([p["week"] for p in common])
    by_week = {"weeks": sorted({int(w) for w in weeks}), "model": [], "vegas": [], "elo": []}
    for w in by_week["weeks"]:
        upto = weeks <= w
        for k in ("model", "vegas", "elo"):
            by_week[k].append(round(log_loss(probs[k][upto], y[upto]), 4) if k != "vegas" or has_vegas else None)
    card["by_week"] = by_week
    return card
