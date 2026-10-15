"""Expected starting QBs for games not yet played.

nflverse's schedule fills `home_qb_id`/`away_qb_id` for upcoming games with a projected
starter, but that projection can go stale (e.g. it kept listing a backup for weeks after
the starter returned from injury). For unplayed games still in the future, prefer the
latest depth-chart QB1 published before `now`. Played games keep the schedule's starter.
"""

from datetime import datetime

import polars as pl

DEPTH_QB_SCHEMA = {"team": pl.Utf8, "dt": pl.Datetime("us", "UTC"), "qb_id": pl.Utf8}


def apply_expected_starters(games: pl.DataFrame, depth_qbs: pl.DataFrame, now: datetime) -> pl.DataFrame:
    """Replace home/away QB ids of future, unplayed games with each team's latest depth-chart QB1."""
    known = depth_qbs.filter(pl.col("dt") <= now)
    if known.height == 0:
        return games
    latest = dict(known.sort("dt").group_by("team").agg(pl.col("qb_id").last()).iter_rows())
    upcoming = pl.col("margin").is_null() & (pl.col("kickoff_utc") > now)
    return games.with_columns(
        pl.when(upcoming & pl.col(f"{side}_team").is_in(list(latest)))
        .then(pl.col(f"{side}_team").replace_strict(latest, default=None))
        .otherwise(pl.col(f"{side}_qb_id"))
        .alias(f"{side}_qb_id")
        for side in ("home", "away")
    )
