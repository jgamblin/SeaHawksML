from datetime import UTC, datetime

import polars as pl
import pytest

from seahawks_ml.features.base import GAMES_SCHEMA
from seahawks_ml.features.qb import QBParams, compute_qb_features, draft_bucket
from seahawks_ml.ingest.nflverse import PLAYERS_SCHEMA, QB_GAMES_SCHEMA


def test_draft_bucket():
    assert draft_bucket(1) == "round_1"
    assert draft_bucket(2) == draft_bucket(3) == "day_2"
    assert draft_bucket(5) == draft_bucket(None) == "day_3_udfa"


def _game(gid, season, day, home_qb, away_qb):
    return {"game_id": gid, "season": season, "week": 1, "game_type": "REG",
            "kickoff_utc": datetime(season, 9, day, 17, tzinfo=UTC), "home_team": "SEA",
            "away_team": "SF", "home_score": 0, "away_score": 0, "margin": 0, "neutral": False,
            "roof": "outdoors", "stadium_id": "SEA00", "home_rest": 7, "away_rest": 7,
            "div_game": True, "home_qb_id": home_qb, "away_qb_id": away_qb, "home_coach": None,
            "away_coach": None, "spread_line": None}


def _setup():
    games = pl.DataFrame([
        _game("g1", 2020, 1, "VET", "ROOK1"),
        _game("g2", 2021, 1, "VET", "ROOK2"),
        _game("g3", 2021, 8, "VET", "ROOK2"),
    ], schema=GAMES_SCHEMA)
    qb = pl.DataFrame([
        {"game_id": "g1", "season": 2020, "team": "SEA", "qb_id": "VET", "dropbacks": 40,
         "qb_epa_sum": 8.0, "cpoe_sum": 40.0, "cpoe_n": 30},
        # 600 early-career round-1 dropbacks at +0.1/db in 2020 sets the 2021 round_1 prior
        {"game_id": "g1", "season": 2020, "team": "SF", "qb_id": "ROOK1", "dropbacks": 600,
         "qb_epa_sum": 60.0, "cpoe_sum": 0.0, "cpoe_n": 500},
        {"game_id": "g2", "season": 2021, "team": "SF", "qb_id": "ROOK2", "dropbacks": 50,
         "qb_epa_sum": -10.0, "cpoe_sum": 0.0, "cpoe_n": 40},
    ], schema=QB_GAMES_SCHEMA)
    players = pl.DataFrame([
        {"gsis_id": "VET", "pfr_id": None, "position": "QB", "draft_round": 2, "rookie_season": 2010},
        {"gsis_id": "ROOK1", "pfr_id": None, "position": "QB", "draft_round": 1, "rookie_season": 2020},
        {"gsis_id": "ROOK2", "pfr_id": None, "position": "QB", "draft_round": 1, "rookie_season": 2021},
    ], schema=PLAYERS_SCHEMA)
    return games, qb, players


def test_rookie_rating_starts_at_bucket_prior_then_updates():
    games, qb, players = _setup()
    out = {r["game_id"]: r for r in compute_qb_features(games, qb, players, QBParams(250, 3)).iter_rows(named=True)}
    assert out["g2"]["away_qb_bucket"] == "round_1"
    assert out["g2"]["away_qb_epa"] == pytest.approx(0.1)  # pure prior: no dropbacks yet
    assert out["g3"]["away_qb_epa"] == pytest.approx((250 * 0.1 - 10.0) / 300)


def test_rating_uses_only_games_before_kickoff():
    games, qb, players = _setup()
    out = {r["game_id"]: r for r in compute_qb_features(games, qb, players, QBParams(250, 3)).iter_rows(named=True)}
    # in 2020 nothing is known: prior defaults to 0.0 and no history before g1
    assert out["g1"]["home_qb_epa"] == 0.0
    # VET's 2021 games see the 40 dropbacks from 2020, shrunk to the day_2 prior (pooled 0.1)
    assert out["g2"]["home_qb_epa"] == pytest.approx((250 * 0.1 + 8.0) / 290)
    assert out["g2"]["home_qb_bucket"] == "day_2"


def test_null_expected_starter_falls_back_to_teams_latest_starter():
    games, qb, players = _setup()
    extra = pl.DataFrame([
        _game("g4", 2021, 15, None, None),  # unknown starters
        _game("g5", 2021, 15, "VET", "ROOK2"),  # same day, known starters
        _game("g6", 2021, 15, None, None) | {"away_team": "KC"},  # KC has no history
    ], schema=GAMES_SCHEMA)
    games = pl.concat([games, extra])
    out = {r["game_id"]: r for r in compute_qb_features(games, qb, players, QBParams(250, 3)).iter_rows(named=True)}
    # SEA's latest starter before g4 is VET (g1), SF's is ROOK2 (g2)
    for col in ("home_qb_epa", "home_qb_cpoe", "home_qb_bucket", "away_qb_epa", "away_qb_bucket"):
        assert out["g4"][col] == out["g5"][col]
    assert out["g4"]["home_qb_bucket"] == "day_2"
    assert out["g6"]["away_qb_bucket"] == "day_3_udfa"  # no history: unchanged behavior
