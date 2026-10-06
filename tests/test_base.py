from datetime import UTC, datetime

import polars as pl

from seahawks_ml.features.base import kickoff_utc, prepare_games, resolve_roof, team_games
from seahawks_ml.stadiums import load_stadiums


def _schedule_row(**overrides):
    row = {
        "game_id": "2024_01_DEN_SEA", "season": 2024, "week": 1, "game_type": "REG",
        "gameday": "2024-09-08", "gametime": "16:05", "away_team": "DEN", "home_team": "SEA",
        "away_score": 20, "home_score": 26, "result": 6, "location": "Home", "roof": "outdoors",
        "stadium_id": "SEA00", "home_rest": 7, "away_rest": 7, "div_game": 0,
        "home_qb_id": "QB-SEA", "away_qb_id": "QB-DEN", "home_coach": "Mike Macdonald",
        "away_coach": "Sean Payton", "spread_line": 6.5,
    }
    row.update(overrides)
    return row


def test_kickoff_utc_converts_eastern_time():
    # 16:05 ET on 2024-09-08 is EDT (UTC-4) -> 20:05 UTC
    assert kickoff_utc("2024-09-08", "16:05") == datetime(2024, 9, 8, 20, 5, tzinfo=UTC)
    # December is EST (UTC-5)
    assert kickoff_utc("2024-12-15", "13:00") == datetime(2024, 12, 15, 18, 0, tzinfo=UTC)


def test_resolve_roof_falls_back_to_stadium_default():
    stadiums = load_stadiums()
    assert resolve_roof("open", stadiums["PHO00"]) == "open"
    assert resolve_roof(None, stadiums["PHO00"]) == "closed"
    assert resolve_roof(None, stadiums["LAX01"]) == "dome"
    assert resolve_roof(None, stadiums["SEA00"]) == "outdoors"


def test_prepare_games_builds_canonical_table():
    sched = pl.DataFrame([_schedule_row(), _schedule_row(
        game_id="2024_09_SEA_LON", stadium_id="LON02", location="Neutral", roof=None,
        result=None, home_score=None, away_score=None, gameday="2024-11-03", gametime="09:30",
    )])
    games = prepare_games(sched, load_stadiums())
    assert games.height == 2
    first = games.row(0, named=True)
    assert first["margin"] == 6 and first["neutral"] is False and first["div_game"] is False
    second = games.row(1, named=True)
    assert second["neutral"] is True and second["roof"] == "outdoors" and second["margin"] is None


def test_team_games_has_two_rows_per_game():
    games = prepare_games(pl.DataFrame([_schedule_row()]), load_stadiums())
    tg = team_games(games)
    assert tg.height == 2
    sea = tg.filter(pl.col("team") == "SEA").row(0, named=True)
    assert sea["is_home"] and sea["points_for"] == 26 and sea["opponent"] == "DEN"
    den = tg.filter(pl.col("team") == "DEN").row(0, named=True)
    assert not den["is_home"] and den["points_for"] == 20 and den["qb_id"] == "QB-DEN"
