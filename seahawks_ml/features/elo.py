"""In-house Elo with margin-of-victory multiplier and preseason regression."""

import math
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0
    hfa: float = 48.0  # Elo points (~1.7 game points)
    revert: float = 1 / 3  # share pulled back to the mean each offseason
    base: float = 1500.0


def elo_win_prob(elo_diff: float, neutral: bool, params: EloParams = EloParams()) -> float:
    """Home win probability from home-minus-away Elo."""
    diff = elo_diff + (0.0 if neutral else params.hfa)
    return 1.0 / (1.0 + 10 ** (-diff / 400))


def _mov_multiplier(margin: int, winner_elo_diff: float) -> float:
    if margin == 0:
        return 1.0
    return math.log(abs(margin) + 1) * 2.2 / (winner_elo_diff * 0.001 + 2.2)


def compute_elo(games: pl.DataFrame, params: EloParams = EloParams()) -> pl.DataFrame:
    """Pre-game Elo for both teams. Games without a result don't update ratings."""
    ratings: dict[str, float] = {}
    rating_season: dict[str, int] = {}
    out = []
    for g in games.sort("kickoff_utc", "game_id").iter_rows(named=True):
        for team in (g["home_team"], g["away_team"]):
            if team not in ratings:
                ratings[team], rating_season[team] = params.base, g["season"]
            elif rating_season[team] != g["season"]:
                ratings[team] = params.base + (ratings[team] - params.base) * (1 - params.revert)
                rating_season[team] = g["season"]
        home, away = ratings[g["home_team"]], ratings[g["away_team"]]
        out.append({"game_id": g["game_id"], "elo_home_pre": home, "elo_away_pre": away})
        margin = g["margin"]
        if margin is None:
            continue
        diff = home - away + (0.0 if g["neutral"] else params.hfa)
        expected = 1.0 / (1.0 + 10 ** (-diff / 400))
        actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        winner_diff = diff if margin > 0 else -diff
        delta = params.k * _mov_multiplier(margin, winner_diff) * (actual - expected)
        ratings[g["home_team"]] += delta
        ratings[g["away_team"]] -= delta
    return pl.DataFrame(out, schema={"game_id": pl.Utf8, "elo_home_pre": pl.Float64, "elo_away_pre": pl.Float64})
