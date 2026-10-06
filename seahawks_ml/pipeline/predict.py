"""Turn one feature row + the fitted model into a Seahawks-perspective prediction record."""

from datetime import datetime

import numpy as np
import polars as pl

from seahawks_ml.config import TEAM
from seahawks_ml.features.columns import FEATURE_COLUMNS
from seahawks_ml.models.pipeline import FittedModel

TOP_FACTORS = 6


def top_factors(model: FittedModel, row: pl.DataFrame, seahawks_home: bool, k: int = TOP_FACTORS) -> list[dict]:
    """Largest per-feature contributions in points, signed for the Seahawks."""
    contrib = model.contributions(row)[0] * (1 if seahawks_home else -1)
    order = np.argsort(-np.abs(contrib))[:k]
    return [{"feature": FEATURE_COLUMNS[i], "points": round(float(contrib[i]), 2)} for i in order]


def latest_injury_week(injuries: pl.DataFrame, season: int, team: str = TEAM) -> int | None:
    weeks = injuries.filter((pl.col("season") == season) & (pl.col("team") == team))["week"]
    return int(weeks.max()) if weeks.len() else None


def single_game_row(frame: pl.DataFrame, game_id: str) -> pl.DataFrame:
    """The one feature row for `game_id`; SystemExit with a clear message otherwise."""
    row = frame.filter(pl.col("game_id") == game_id)
    if row.height != 1:
        raise SystemExit(f"expected exactly one feature row for {game_id}, got {row.height}")
    return row


def make_prediction(
    row: pl.DataFrame,
    model: FittedModel,
    run_type: str,
    now: datetime,
    model_version: str,
    injury_week: int | None,
) -> dict:
    g = row.row(0, named=True)
    home = g["home_team"] == TEAM
    out = model.predict(row)
    p_home, m = float(out["p_win"][0]), float(out["margin"][0])
    lo, hi = float(out["margin_lo"][0]), float(out["margin_hi"][0])
    spread = g["spread_line"]
    p_vegas_home = float(model.vegas_prob(np.array([spread]))[0]) if spread is not None else None
    return {
        "type": "prediction",
        "game_id": g["game_id"],
        "run_type": run_type,
        "predicted_at": now.isoformat(),
        "kickoff_utc": g["kickoff_utc"].isoformat(),
        "is_final_injury_report": run_type in ("final_injury", "gameday") and injury_week == g["week"],
        "opponent": g["away_team"] if home else g["home_team"],
        "seahawks_home": home and not g["neutral"],
        "p_seahawks": round(p_home if home else 1 - p_home, 4),
        "margin_seahawks": round(m if home else -m, 2),
        "margin_lo": round(lo if home else -hi, 1),
        "margin_hi": round(hi if home else -lo, 1),
        "p_vegas_seahawks": None if p_vegas_home is None else round(p_vegas_home if home else 1 - p_vegas_home, 4),
        "weather": {
            "source": g["weather_source"],
            "temp_f": round(g["temp_f"], 1),
            "wind_mph": round(g["wind_mph"], 1),
            "precip_in": round(g["precip_in"], 2),
        },
        "latest_injury_week": injury_week,
        "top_factors": top_factors(model, row, home),
        "model_version": model_version,
    }
