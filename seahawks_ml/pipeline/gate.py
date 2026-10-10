"""Decide whether a prediction run is due. Cheap: needs only the schedule and history.

Windows are relative to kickoff, so Thursday/Monday/Saturday and international games
work like Sunday games. Scheduled runs land inside these windows; the first one in a window does
the run, later ones see it in history and skip. GitHub fires scheduled runs irregularly (every few
hours in practice), so a run that lands shortly before the short gameday window waits for it
(`gameday_wait`) instead of relying on a later scheduled run.
"""

from datetime import datetime, timedelta

import polars as pl

# run_type -> (opens this long before kickoff, closes this long before kickoff)
RUN_WINDOWS = {
    "midweek": (timedelta(days=4, hours=12), timedelta(days=2)),
    "final_injury": (timedelta(hours=24), timedelta(hours=6)),
    "gameday": (timedelta(hours=6), timedelta(minutes=75)),
}
RESULT_GRACE = timedelta(hours=6)  # wait for the previous game's result before predicting
WAIT_MARGIN = timedelta(minutes=1)  # sleep slightly past the window opening


def team_schedule(games: pl.DataFrame, team: str) -> pl.DataFrame:
    return games.filter((pl.col("home_team") == team) | (pl.col("away_team") == team)).sort("kickoff_utc")


def next_game(games: pl.DataFrame, team: str, now: datetime) -> dict | None:
    upcoming = team_schedule(games, team).filter(pl.col("kickoff_utc") > now)
    return upcoming.row(0, named=True) if upcoming.height else None


def previous_kickoff(games: pl.DataFrame, team: str, now: datetime) -> datetime | None:
    past = team_schedule(games, team).filter(pl.col("kickoff_utc") <= now)
    return past["kickoff_utc"][-1] if past.height else None


def due_run(now: datetime, kickoff: datetime, done: set[str], prev_kickoff: datetime | None = None) -> str | None:
    if prev_kickoff is not None and now - prev_kickoff < RESULT_GRACE:
        return None
    for run_type, (opens, closes) in RUN_WINDOWS.items():
        if kickoff - opens <= now < kickoff - closes:
            return None if run_type in done else run_type
    return None


def previous_game_ready(games: pl.DataFrame, team_epa: pl.DataFrame, team: str, now: datetime) -> bool:
    """True when the team's previous game has both its final margin and its team EPA rows."""
    past = team_schedule(games, team).filter(pl.col("kickoff_utc") < now)
    if not past.height:
        return True
    prev = past.row(-1, named=True)
    if prev["margin"] is None:
        return False
    return team_epa.filter((pl.col("game_id") == prev["game_id"]) & (pl.col("team") == team)).height > 0


def gameday_wait(now: datetime, kickoff: datetime, done: set[str], max_wait: timedelta) -> timedelta | None:
    """How long to sleep so a run can catch the gameday window, or None if it shouldn't wait.

    Only waits when the window hasn't opened yet, opens within `max_wait`, and the gameday run
    hasn't been logged.
    """
    opens = kickoff - RUN_WINDOWS["gameday"][0]
    if "gameday" in done or now >= opens or opens - now > max_wait:
        return None
    return opens - now + WAIT_MARGIN
