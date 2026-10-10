from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from seahawks_ml.pipeline.gate import due_run, gameday_wait, next_game, previous_game_ready, previous_kickoff

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
    (timedelta(hours=6), "gameday"),
    (timedelta(hours=5), "gameday"),
    (timedelta(hours=3), "gameday"),
    (timedelta(minutes=80), "gameday"),
    (timedelta(minutes=75), None),
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


def _ready_frames(margin, with_epa=True):
    games = pl.DataFrame({
        "game_id": ["a", "b"],
        "home_team": ["SEA", "SF"],
        "away_team": ["LA", "SEA"],
        "kickoff_utc": [SUN_1PM_ET - timedelta(days=7), SUN_1PM_ET],
        "margin": [margin, None],
    }, schema_overrides={"kickoff_utc": pl.Datetime("us", "UTC"), "margin": pl.Int64})
    epa = pl.DataFrame({"game_id": ["a"] if with_epa else [], "team": ["SEA"] if with_epa else []},
                       schema={"game_id": pl.Utf8, "team": pl.Utf8})
    return games, epa


def test_previous_game_ready():
    now = SUN_1PM_ET - timedelta(days=1)
    assert previous_game_ready(*_ready_frames(3), "SEA", now)
    assert not previous_game_ready(*_ready_frames(None), "SEA", now)
    assert not previous_game_ready(*_ready_frames(3, with_epa=False), "SEA", now)
    # no previous game (season opener): nothing to wait for
    assert previous_game_ready(*_ready_frames(None), "SEA", SUN_1PM_ET - timedelta(days=30))


MAX_WAIT = timedelta(hours=5, minutes=30)


def test_gameday_wait_sleeps_until_window_opens():
    # window opens 6h before kickoff; wait one extra minute so the gate sees it open
    assert gameday_wait(SUN_1PM_ET - timedelta(hours=9), SUN_1PM_ET, set(), MAX_WAIT) == timedelta(hours=3, minutes=1)


@pytest.mark.parametrize("before,done", [
    (timedelta(hours=12), set()),          # too far ahead: a later run will handle it
    (timedelta(hours=5), set()),           # window already open: the normal gate runs it
    (timedelta(hours=9), {"gameday"}),     # already done
    (timedelta(hours=-1), set()),          # after kickoff
])
def test_gameday_wait_returns_none(before, done):
    assert gameday_wait(SUN_1PM_ET - before, SUN_1PM_ET, done, MAX_WAIT) is None
