"""RawData: every input table the feature builder needs, plus as-of truncation."""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

import polars as pl

from seahawks_ml.config import FIRST_RATING_SEASON
from seahawks_ml.features.base import prepare_games
from seahawks_ml.stadiums import Stadium


@dataclass(frozen=True)
class RawData:
    games: pl.DataFrame  # canonical games table (features.base.GAMES_SCHEMA)
    team_epa: pl.DataFrame  # ingest.nflverse.TEAM_EPA_SCHEMA
    qb_games: pl.DataFrame  # ingest.nflverse.QB_GAMES_SCHEMA
    injuries: pl.DataFrame  # ingest.nflverse.INJURIES_SCHEMA
    snaps: pl.DataFrame  # ingest.nflverse.SNAPS_SCHEMA
    players: pl.DataFrame  # ingest.nflverse.PLAYERS_SCHEMA
    weather: pl.DataFrame  # ingest.weather.WEATHER_SCHEMA

    def as_of(self, cutoff: datetime) -> "RawData":
        """Drop everything not knowable before `cutoff`.

        Results of games kicking off at/after the cutoff are nulled; per-game stats for
        those games are removed; injury reports are kept for every (season, week) whose
        earliest kickoff is before cutoff + 1 day, so a game's own final report survives.
        That can include other teams' later reports for the same week; per-game features
        only read their own game's week, so this does not leak into the target game. Weather is
        left alone because a forecast stands in for it at prediction time.
        """
        later = pl.col("kickoff_utc") >= cutoff
        games = self.games.with_columns(
            *[pl.when(later).then(None).otherwise(pl.col(c)).alias(c)
              for c in ("home_score", "away_score", "margin")]
        )
        played = set(self.games.filter(~later)["game_id"].to_list())
        week_start = self.games.group_by("season", "week").agg(pl.col("kickoff_utc").min().alias("start"))
        open_weeks = week_start.filter(pl.col("start") < cutoff + timedelta(days=1)).select("season", "week")
        return replace(
            self,
            games=games,
            team_epa=self.team_epa.filter(pl.col("game_id").is_in(list(played))),
            qb_games=self.qb_games.filter(pl.col("game_id").is_in(list(played))),
            snaps=self.snaps.filter(pl.col("game_id").is_in(list(played))),
            injuries=self.injuries.join(open_weeks, on=["season", "week"], how="semi"),
        )


def current_season(games: pl.DataFrame, now: datetime) -> int:
    """Latest season with a game kicking off by now + 7 days (so an opener's week counts)."""
    return int(games.filter(pl.col("kickoff_utc") <= now + timedelta(days=7))["season"].max())


def load_raw(stadiums: dict[str, Stadium], now: datetime, games: pl.DataFrame | None = None) -> RawData:
    """Download/refresh all inputs. Completed seasons come from data/cache.

    Pass an already-prepared `games` table to avoid downloading the schedule again.
    """
    from seahawks_ml.ingest import nflverse, weather

    if games is None:
        games = prepare_games(nflverse.load_schedules(), stadiums)
    current = current_season(games, now)
    seasons = list(range(FIRST_RATING_SEASON, current + 1))
    # Refetch last season's cache once after its final game (+7d) so late playoff
    # data and stat corrections are picked up.
    last_kickoff = games.filter(pl.col("season") == current - 1)["kickoff_utc"].max()
    stale_before = {current - 1: last_kickoff + timedelta(days=7)} if last_kickoff is not None else None
    tables = nflverse.load_season_tables(
        seasons, current_season=current, stale_before=stale_before
    )
    return RawData(
        games=games,
        team_epa=tables["team_epa"],
        qb_games=tables["qb_games"],
        injuries=tables["injuries"],
        snaps=tables["snaps"],
        players=nflverse.load_players(),
        weather=weather.update_archive_cache(games, stadiums, now),
    )
