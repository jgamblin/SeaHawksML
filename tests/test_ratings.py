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


EXTRA = ["pass_off_rating", "pass_def_rating", "rush_off_rating", "rush_def_rating",
         "success_off_rating", "success_def_rating"]


def _with_extras(epa):
    return epa.with_columns(
        pass_epa_sum=pl.Series([6.0, -12.0, 20.0, 4.0]), pass_plays=pl.Series([30, 30, 40, 20]),
        rush_epa_sum=pl.Series([6.0, 6.0, 10.0, -4.0]), rush_plays=pl.Series([30, 30, 20, 40]),
        success_sum=pl.Series([30.0, 24.0, 33.0, 27.0]),
    )


def _by_key(out):
    return {(x["game_id"], x["team"]): x for x in out.iter_rows(named=True)}


def test_extra_stats_use_same_shrinkage_scheme():
    tg, epa = _setup()
    r = _by_key(compute_team_ratings(tg, _with_extras(epa), _flags(tg), RatingParams(0.5, 4.0, 2.0)))
    # 2020 pass EPA/play: SEA 0.2, SF -0.4 -> league -0.1. SEA prior = 0.5*(0.2+0.1) = 0.15
    assert r[("b", "SEA")]["pass_off_rating"] == pytest.approx(0.15)
    assert r[("b", "SEA")]["pass_def_rating"] == pytest.approx(0.5 * (-0.4 + 0.1))
    # 2021 SEA pass 0.5 centered on -0.1 -> 0.6
    assert r[("c", "SEA")]["pass_off_rating"] == pytest.approx((4 * 0.15 + 0.6) / 5)
    # rush 2020: SEA 0.2, SF 0.2 -> league 0.2, SEA prior 0; 2021 SEA rush 0.5 -> dev 0.3
    assert r[("c", "SEA")]["rush_off_rating"] == pytest.approx(0.3 / 5)
    # success 2020: SEA 0.5, SF 0.4 -> league 0.45; SF prior = 0.5*(0.4-0.45) = -0.025
    # 2021 SF success 0.45 -> dev 0.0
    assert r[("c", "SF")]["success_off_rating"] == pytest.approx(4 * -0.025 / 5)
    assert r[("c", "SF")]["success_def_rating"] == pytest.approx((4 * 0.025 + (0.55 - 0.45)) / 5)


def test_extra_stats_off_or_missing_columns_give_zero():
    tg, epa = _setup()
    base = compute_team_ratings(tg, epa, _flags(tg), RatingParams(0.5, 4.0, 2.0))
    off = compute_team_ratings(tg, _with_extras(epa), _flags(tg), RatingParams(0.5, 4.0, 2.0, extra_stats=False))
    for df in (base, off):
        for c in EXTRA:
            assert (df[c] == 0.0).all(), c
    assert base["off_rating"].to_list() == off["off_rating"].to_list()


def test_opponent_adjust_uses_opponents_pregame_rating():
    tg, epa = _setup()
    params = RatingParams(0.5, 4.0, 2.0, opponent_adjust=True)
    r = _by_key(compute_team_ratings(tg, epa, _flags(tg), params))
    # pre-game b: SEA off 0.075 / def -0.075; SF off -0.075 / def 0.075 (priors unchanged)
    assert r[("b", "SEA")]["off_rating"] == pytest.approx(0.075)
    # SEA off dev at b = 0.45 minus SF pre-game def 0.075
    assert r[("c", "SEA")]["off_rating"] == pytest.approx((4 * 0.075 + 0.375) / 5)
    # SEA def dev at b = -0.05 minus SF pre-game off -0.075
    assert r[("c", "SEA")]["def_rating"] == pytest.approx((4 * -0.075 + 0.025) / 5)
    # SF off dev at b = -0.05 minus SEA pre-game def -0.075 (not SEA's post-game rating)
    assert r[("c", "SF")]["off_rating"] == pytest.approx((4 * -0.075 + 0.025) / 5)
    assert r[("c", "SF")]["def_rating"] == pytest.approx((4 * 0.075 + 0.375) / 5)


def test_opponent_adjust_independent_of_row_order():
    tg, epa = _setup()
    params = RatingParams(0.5, 4.0, 2.0, opponent_adjust=True)
    a = compute_team_ratings(tg, _with_extras(epa), _flags(tg), params).sort("game_id", "team")
    b = compute_team_ratings(tg.reverse(), _with_extras(epa), _flags(tg), params).sort("game_id", "team")
    assert a.equals(b)
