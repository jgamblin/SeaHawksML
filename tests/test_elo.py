from datetime import UTC, datetime

import polars as pl
import pytest

from seahawks_ml.features.base import GAMES_SCHEMA
from seahawks_ml.features.elo import EloParams, compute_elo, elo_win_prob


def _game(gid, season, day, home, away, margin, neutral=False):
    return {"game_id": gid, "season": season, "week": 1, "game_type": "REG",
            "kickoff_utc": datetime(season, 9, day, 17, tzinfo=UTC), "home_team": home,
            "away_team": away, "home_score": None, "away_score": None, "margin": margin,
            "neutral": neutral, "roof": "outdoors", "stadium_id": "SEA00", "home_rest": 7,
            "away_rest": 7, "div_game": False, "home_qb_id": None, "away_qb_id": None,
            "home_coach": None, "away_coach": None, "spread_line": None}


def test_elo_win_prob_includes_home_field():
    assert elo_win_prob(0.0, neutral=True) == pytest.approx(0.5)
    assert elo_win_prob(0.0, neutral=False) > 0.5


def test_compute_elo_updates_after_result_and_is_zero_sum():
    games = pl.DataFrame([
        _game("g1", 2020, 1, "SEA", "SF", 14),
        _game("g2", 2020, 8, "SF", "SEA", None),
    ], schema=GAMES_SCHEMA)
    out = {r["game_id"]: r for r in compute_elo(games).iter_rows(named=True)}
    assert out["g1"]["elo_home_pre"] == 1500 and out["g1"]["elo_away_pre"] == 1500
    sea_after, sf_after = out["g2"]["elo_away_pre"], out["g2"]["elo_home_pre"]
    assert sea_after > 1500 > sf_after
    assert sea_after + sf_after == pytest.approx(3000)


def test_compute_elo_regresses_toward_mean_between_seasons():
    games = pl.DataFrame([
        _game("g1", 2020, 1, "SEA", "SF", 30),
        _game("g2", 2021, 1, "SEA", "SF", None),
    ], schema=GAMES_SCHEMA)
    kept = compute_elo(games, EloParams(revert=0.0)).row(1, named=True)["elo_home_pre"]
    reverted = compute_elo(games, EloParams(revert=1 / 3)).row(1, named=True)["elo_home_pre"]
    assert reverted - 1500 == pytest.approx((kept - 1500) * (2 / 3))
