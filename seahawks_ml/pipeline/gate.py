"""Decide whether a prediction run is due. Cheap: needs only the schedule and history.

Windows are relative to kickoff, so Thursday/Monday/Saturday and international games
work like Sunday games. Hourly cron runs land inside these windows; the first one in a
window does the run, later ones see it in history and skip.
"""

from datetime import datetime, timedelta

import polars as pl

# run_type -> (opens this long before kickoff, closes this long before kickoff)
RUN_WINDOWS = {
    "midweek": (timedelta(days=4, hours=12), timedelta(days=2)),
    "final_injury": (timedelta(hours=30), timedelta(hours=6)),
    "gameday": (timedelta(hours=4), timedelta(minutes=90)),
}
RESULT_GRACE = timedelta(hours=6)  # wait for the previous game's result before predicting


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
