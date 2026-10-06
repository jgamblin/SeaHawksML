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
        those games are removed; injury reports are kept only for weeks that have
        started by cutoff + 1 day (so a game's own final report survives). Weather is
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


def load_raw(stadiums: dict[str, Stadium], now: datetime) -> RawData:
    """Download/refresh all inputs. Completed seasons come from data/cache."""
    from seahawks_ml.ingest import nflverse, weather

    games = prepare_games(nflverse.load_schedules(), stadiums)
    current = int(games.filter(pl.col("kickoff_utc") <= now)["season"].max())
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
