"""Append result records for predicted Seahawks games that have finished."""

from datetime import datetime

import polars as pl

from seahawks_ml.config import TEAM


def new_results(history: list[dict], games: pl.DataFrame, now: datetime, team: str = TEAM) -> list[dict]:
    predicted = {r["game_id"] for r in history if r["type"] == "prediction"}
    recorded = {r["game_id"] for r in history if r["type"] == "result"}
    pending = predicted - recorded
    out = []
    finished = games.filter(pl.col("game_id").is_in(list(pending)) & pl.col("margin").is_not_null())
    for g in finished.iter_rows(named=True):
        home = g["home_team"] == team
        us, them = (g["home_score"], g["away_score"]) if home else (g["away_score"], g["home_score"])
        out.append({"type": "result", "game_id": g["game_id"], "recorded_at": now.isoformat(),
                    "seahawks_score": us, "opponent_score": them, "margin_seahawks": us - them})
    return out
