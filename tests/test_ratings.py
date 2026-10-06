from datetime import UTC, datetime

import polars as pl
import pytest

from seahawks_ml.features.base import TEAM_GAMES_SCHEMA
from seahawks_ml.features.ratings import RatingParams, compute_team_ratings


def _tg(game_id, season, day, team, opp):
    return {"game_id": game_id, "season": season, "week": 1,
            "kickoff_utc": datetime(season, 9, day, 17, tzinfo=UTC), "team": team, "opponent": opp,
            "is_home": True, "points_for": 0, "points_against": 0, "coach": "c", "qb_id": None}


def _setup():
    tg = pl.DataFrame([
        _tg("a", 2020, 1, "SEA", "SF"), _tg("a", 2020, 1, "SF", "SEA"),
        _tg("b", 2021, 1, "SEA", "SF"), _tg("b", 2021, 1, "SF", "SEA"),
        _tg("c", 2021, 8, "SEA", "SF"), _tg("c", 2021, 8, "SF", "SEA"),
    ], schema=TEAM_GAMES_SCHEMA)
    epa = pl.DataFrame({
        "game_id": ["a", "a", "b", "b"], "season": [2020, 2020, 2021, 2021],
        "team": ["SEA", "SF", "SEA", "SF"], "opponent": ["SF", "SEA", "SF", "SEA"],
        "epa_sum": [12.0, -6.0, 30.0, 0.0], "plays": [60, 60, 60, 60],
    })
    return tg, epa


def _flags(tg, sf_new=0):
    return tg.select("game_id", "team").with_columns(
        pl.when((pl.col("team") == "SF") & (pl.col("game_id") != "a")).then(sf_new).otherwise(0)
        .alias("new_head_coach"))


def test_ratings_use_shrunk_prior_then_blend_current_season():
    tg, epa = _setup()
    out = compute_team_ratings(tg, epa, _flags(tg), RatingParams(0.5, 4.0, 2.0))
    r = {(x["game_id"], x["team"]): x for x in out.iter_rows(named=True)}
    # 2020 has no prior season -> zero
    assert r[("a", "SEA")]["off_rating"] == 0.0
    # 2020: SEA off 0.2, SF off -0.1, league 0.05. SEA prior off = 0.5*(0.2-0.05) = 0.075
    assert r[("b", "SEA")]["off_rating"] == pytest.approx(0.075)
    assert r[("b", "SEA")]["def_rating"] == pytest.approx(0.5 * (-0.1 - 0.05))
    # game c blends one 2021 game: SEA off 0.5, centered on 2020 league 0.05 -> 0.45
    assert r[("c", "SEA")]["off_rating"] == pytest.approx((4 * 0.075 + 0.45) / 5)


def test_new_head_coach_discounts_prior_faster():
    tg, epa = _setup()
    params = RatingParams(0.5, 4.0, 1.0)
    same = compute_team_ratings(tg, epa, _flags(tg, 0), params)
    new = compute_team_ratings(tg, epa, _flags(tg, 1), params)

    def get(df):
        return df.filter((pl.col("game_id") == "c") & (pl.col("team") == "SF"))["off_rating"][0]

    # SF prior off = 0.5*(-0.1-0.05) = -0.075; 2021 game dev = 0.0-0.05 = -0.05
    assert get(same) == pytest.approx((4 * -0.075 - 0.05) / 5)
    assert get(new) == pytest.approx((1 * -0.075 - 0.05) / 2)
