from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from seahawks_ml.pipeline.gate import due_run, next_game, previous_kickoff

SUN_1PM_ET = datetime(2026, 10, 11, 17, 0, tzinfo=UTC)
THU_820_ET = datetime(2026, 10, 16, 0, 15, tzinfo=UTC)
LONDON_930_ET = datetime(2026, 10, 25, 13, 30, tzinfo=UTC)
MON_815_ET = datetime(2026, 10, 20, 0, 15, tzinfo=UTC)
SAT_430_ET = datetime(2026, 12, 19, 21, 30, tzinfo=UTC)


@pytest.mark.parametrize("kickoff", [SUN_1PM_ET, THU_820_ET, LONDON_930_ET, MON_815_ET, SAT_430_ET])
@pytest.mark.parametrize("before,expected", [
    (timedelta(days=5), None),
    (timedelta(days=4), "midweek"),
    (timedelta(days=2, hours=1), "midweek"),
    (timedelta(days=1, hours=12), None),
    (timedelta(hours=25), None),
    (timedelta(hours=24), "final_injury"),
    (timedelta(hours=7), "final_injury"),
    (timedelta(hours=5), None),
    (timedelta(hours=3), "gameday"),
    (timedelta(minutes=60), None),
])
def test_due_run_windows(kickoff, before, expected):
    assert due_run(kickoff - before, kickoff, done=set()) == expected


def test_due_run_skips_runs_already_done():
    assert due_run(SUN_1PM_ET - timedelta(hours=3), SUN_1PM_ET, done={"gameday"}) is None


def test_due_run_waits_for_previous_game_result():
    prev = THU_820_ET - timedelta(days=4)  # Sunday game before a Thursday game
    now = prev + timedelta(hours=2)
    assert due_run(now, THU_820_ET, set(), prev_kickoff=prev) is None
    assert due_run(prev + timedelta(hours=7), THU_820_ET, set(), prev_kickoff=prev) == "midweek"


def test_next_and_previous_game():
    games = pl.DataFrame({
        "game_id": ["a", "b", "c"],
        "home_team": ["SEA", "SF", "SEA"],
        "away_team": ["LA", "SEA", "KC"],
        "kickoff_utc": [SUN_1PM_ET - timedelta(days=7), SUN_1PM_ET, SUN_1PM_ET + timedelta(days=7)],
    }, schema_overrides={"kickoff_utc": pl.Datetime("us", "UTC")})
    now = SUN_1PM_ET - timedelta(days=1)
    assert next_game(games, "SEA", now)["game_id"] == "b"
    assert previous_kickoff(games, "SEA", now) == SUN_1PM_ET - timedelta(days=7)
    assert next_game(games, "SEA", SUN_1PM_ET + timedelta(days=8)) is None
