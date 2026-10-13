from datetime import UTC, datetime

import polars as pl
import pytest

from seahawks_ml.features.base import GAMES_SCHEMA
from seahawks_ml.features.situational import (
    SituationalParams,
    compute_situational,
    haversine_miles,
    team_home_stadiums,
)
from seahawks_ml.stadiums import load_stadiums


def _game(gid, home, away, stadium, ko, neutral=False, home_rest=7, away_rest=7, season=2024):
    return {"game_id": gid, "season": season, "week": 1, "game_type": "REG", "kickoff_utc": ko,
            "home_team": home, "away_team": away, "home_score": None, "away_score": None,
            "margin": None, "neutral": neutral, "roof": "outdoors", "stadium_id": stadium,
            "home_rest": home_rest, "away_rest": away_rest, "div_game": True, "home_qb_id": None,
            "away_qb_id": None, "home_coach": None, "away_coach": None, "spread_line": None}


def _games():
    return pl.DataFrame([
        # Seattle at Miami, 1pm ET (10am PT body clock for SEA)
        _game("mia", "MIA", "SEA", "MIA00", datetime(2024, 10, 6, 17, 0, tzinfo=UTC), away_rest=14),
        _game("sea", "SEA", "MIA", "SEA00", datetime(2024, 10, 13, 20, 5, tzinfo=UTC), home_rest=4),
        # London neutral-site game, SEA listed as home
        _game("lon", "SEA", "MIA", "LON02", datetime(2024, 10, 20, 13, 30, tzinfo=UTC), neutral=True),
        # Sunday night game in Seattle
        _game("snf", "SEA", "MIA", "SEA00", datetime(2024, 10, 28, 0, 20, tzinfo=UTC)),
    ], schema=GAMES_SCHEMA)


def test_haversine_seattle_to_miami():
    assert haversine_miles(47.595, -122.332, 25.958, -80.239) == pytest.approx(2720, rel=0.02)


def test_team_home_stadiums_ignores_neutral_sites():
    home = team_home_stadiums(_games())
    assert home[("SEA", 2024)] == "SEA00" and home[("MIA", 2024)] == "MIA00"


def test_compute_situational_values():
    out = {r["game_id"]: r for r in compute_situational(_games(), load_stadiums()).iter_rows(named=True)}
    mia = out["mia"]
    assert mia["away_body_clock"] == pytest.approx(10.0)  # 1pm ET = 10am PT
    assert mia["home_body_clock"] == pytest.approx(13.0)
    assert mia["away_tz_shift"] == pytest.approx(3.0) and mia["home_tz_shift"] == 0.0
    assert mia["away_post_bye"] == 1 and mia["rest_diff"] == -7
    assert mia["travel_diff"] == pytest.approx(-2720, rel=0.02)
    assert out["sea"]["home_short_week"] == 1
    lon = out["lon"]
    assert lon["home_field"] == 0 and lon["hfa_trend"] == 0.0
    assert lon["home_tz_shift"] == pytest.approx(8.0)  # BST vs PDT
    assert out["snf"]["primetime"] == 1 and out["sea"]["primetime"] == 0
    assert out["sea"]["hfa_trend"] == pytest.approx(0.9)


def _altitude_games():
    ko = datetime(2024, 11, 3, 21, 5, tzinfo=UTC)
    return pl.concat([_games(), pl.DataFrame([
        _game("den", "DEN", "SEA", "DEN00", ko),
        _game("sea2", "SEA", "DEN", "SEA00", ko),
        _game("mex", "ARI", "SF", "MEX00", ko, neutral=True),
        _game("ari", "ARI", "DEN", "PHO00", ko),
        _game("sf", "SF", "ARI", "SFO01", ko),
    ], schema=GAMES_SCHEMA)])


def test_away_altitude_gain_km():
    stadiums = load_stadiums()
    on = {r["game_id"]: r for r in
          compute_situational(_altitude_games(), stadiums, SituationalParams(altitude=True)).iter_rows(named=True)}
    # Seattle visiting Denver climbs 1609 - 5 m
    assert on["den"]["away_altitude_gain_km"] == pytest.approx((1609 - stadiums["SEA00"].elevation_m) / 1000)
    assert on["sea2"]["away_altitude_gain_km"] == 0.0  # Denver descending
    assert on["ari"]["away_altitude_gain_km"] == 0.0
    # neutral site: SF (home Levi's) climbs to Mexico City
    assert on["mex"]["away_altitude_gain_km"] == pytest.approx((2240 - stadiums["SFO01"].elevation_m) / 1000)
    assert on["sf"]["away_altitude_gain_km"] == pytest.approx(0.0)  # ARI at sea level from 331 m
    off = compute_situational(_altitude_games(), stadiums)
    assert (off["away_altitude_gain_km"] == 0.0).all()
