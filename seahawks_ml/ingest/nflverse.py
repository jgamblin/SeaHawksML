"""Download nflverse tables and reduce them to compact, per-season cached parquet files.

Completed seasons are cached in data/cache and committed; the current season is
refreshed on every run. Play-by-play is aggregated immediately so the full
372-column table never has to be kept around.
"""

from collections.abc import Callable
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from seahawks_ml.config import CACHE_DIR, FIRST_SNAP_SEASON, FIRST_TRAIN_SEASON
from seahawks_ml.teams import normalize_team

TEAM_EPA_SCHEMA = {
    "game_id": pl.Utf8, "season": pl.Int64, "team": pl.Utf8, "opponent": pl.Utf8,
    "epa_sum": pl.Float64, "plays": pl.Int64,
}
QB_GAMES_SCHEMA = {
    "game_id": pl.Utf8, "season": pl.Int64, "team": pl.Utf8, "qb_id": pl.Utf8,
    "dropbacks": pl.Int64, "qb_epa_sum": pl.Float64, "cpoe_sum": pl.Float64, "cpoe_n": pl.Int64,
}
INJURIES_SCHEMA = {
    "season": pl.Int64, "week": pl.Int64, "team": pl.Utf8, "gsis_id": pl.Utf8,
    "position": pl.Utf8, "report_status": pl.Utf8,
}
SNAPS_SCHEMA = {
    "game_id": pl.Utf8, "season": pl.Int64, "week": pl.Int64, "team": pl.Utf8,
    "pfr_player_id": pl.Utf8, "position": pl.Utf8, "offense_pct": pl.Float64,
    "defense_pct": pl.Float64,
}
PLAYERS_SCHEMA = {
    "gsis_id": pl.Utf8, "pfr_id": pl.Utf8, "position": pl.Utf8,
    "draft_round": pl.Int64, "rookie_season": pl.Int64,
}


def aggregate_team_epa(pbp: pl.DataFrame) -> pl.DataFrame:
    """Offensive EPA per team-game from scrimmage plays (pass + run)."""
    plays = pbp.filter(
        pl.col("play_type").is_in(["pass", "run"])
        & pl.col("epa").is_not_null()
        & pl.col("posteam").is_not_null()
        & (pl.col("two_point_attempt").fill_null(0) != 1)
    )
    return (
        plays.group_by("game_id", "season", "posteam", "defteam")
        .agg(pl.col("epa").sum().alias("epa_sum"), pl.len().alias("plays"))
        .rename({"posteam": "team", "defteam": "opponent"})
        .with_columns(normalize_team("team"), normalize_team("opponent"))
        .cast(TEAM_EPA_SCHEMA)
        .select(list(TEAM_EPA_SCHEMA))
        .sort("game_id", "team")
    )


def aggregate_qb_games(pbp: pl.DataFrame) -> pl.DataFrame:
    """Dropback totals per passer per game."""
    drops = pbp.filter(
        (pl.col("qb_dropback") == 1)
        & pl.col("passer_id").is_not_null()  # includes scrambles, unlike passer_player_id
        & pl.col("qb_epa").is_not_null()
    )
    return (
        drops.group_by("game_id", "season", "posteam", "passer_id")
        .agg(
            pl.len().alias("dropbacks"),
            pl.col("qb_epa").sum().alias("qb_epa_sum"),
            pl.col("cpoe").sum().alias("cpoe_sum"),
            pl.col("cpoe").count().alias("cpoe_n"),
        )
        .rename({"posteam": "team", "passer_id": "qb_id"})
        .with_columns(normalize_team("team"))
        .cast(QB_GAMES_SCHEMA)
        .select(list(QB_GAMES_SCHEMA))
        .sort("game_id", "qb_id")
    )


def _cached(
    path: Path, fetch: Callable[[], pl.DataFrame], refresh: bool, empty_on_error: dict | None = None
) -> pl.DataFrame:
    """Read-through cache. With `empty_on_error` (a schema) a failed fetch yields an
    empty table that is NOT cached; without it the exception propagates."""
    if path.exists() and not refresh:
        return pl.read_parquet(path)
    try:
        df = fetch()
    except Exception as exc:  # noqa: BLE001 - nflreadpy raises several error types
        if empty_on_error is None:
            raise
        print(f"warning: nflverse fetch returned no data ({exc}); using empty table")
        return pl.DataFrame(schema=empty_on_error)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


def load_schedules() -> pl.DataFrame:
    return nfl.load_schedules(True).with_columns(
        normalize_team("home_team"), normalize_team("away_team")
    )


def load_players() -> pl.DataFrame:
    return (
        nfl.load_players()
        .select(list(PLAYERS_SCHEMA))
        .cast(PLAYERS_SCHEMA)
        .filter(pl.col("gsis_id").is_not_null())
    )


def _fetch_pbp_tables(season: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    pbp = nfl.load_pbp(season)
    return aggregate_team_epa(pbp), aggregate_qb_games(pbp)


def _fetch_injuries(season: int) -> pl.DataFrame:
    return (
        nfl.load_injuries(season)
        .with_columns(normalize_team("team"))
        .select(list(INJURIES_SCHEMA))
        .cast(INJURIES_SCHEMA)
    )


def _fetch_snaps(season: int) -> pl.DataFrame:
    return (
        nfl.load_snap_counts(season)
        .with_columns(normalize_team("team"))
        .select(list(SNAPS_SCHEMA))
        .cast(SNAPS_SCHEMA)
    )


def load_season_tables(
    seasons: list[int], current_season: int, cache_dir: Path = CACHE_DIR
) -> dict[str, pl.DataFrame]:
    """Return team_epa, qb_games, injuries and snaps for the given seasons."""
    parts: dict[str, list[pl.DataFrame]] = {k: [] for k in ("team_epa", "qb_games", "injuries", "snaps")}
    for season in seasons:
        refresh = season == current_season
        tolerate = refresh  # only the current season may legitimately lack data
        team_path = cache_dir / f"team_epa_{season}.parquet"
        qb_path = cache_dir / f"qb_games_{season}.parquet"
        if refresh or not (team_path.exists() and qb_path.exists()):
            try:
                team_epa, qb_games = _fetch_pbp_tables(season)
            except Exception as exc:  # noqa: BLE001 - nflreadpy raises several error types
                if not tolerate:
                    raise
                print(f"warning: no play-by-play for {season} yet ({exc})")
                team_epa = pl.DataFrame(schema=TEAM_EPA_SCHEMA)
                qb_games = pl.DataFrame(schema=QB_GAMES_SCHEMA)
            else:
                cache_dir.mkdir(parents=True, exist_ok=True)
                team_epa.write_parquet(team_path)
                qb_games.write_parquet(qb_path)
        else:
            team_epa, qb_games = pl.read_parquet(team_path), pl.read_parquet(qb_path)
        parts["team_epa"].append(team_epa)
        parts["qb_games"].append(qb_games)
        if season >= FIRST_TRAIN_SEASON:
            parts["injuries"].append(_cached(
                cache_dir / f"injuries_{season}.parquet",
                lambda s=season: _fetch_injuries(s),
                refresh,
                INJURIES_SCHEMA if tolerate else None,
            ))
        if season >= FIRST_SNAP_SEASON:
            parts["snaps"].append(_cached(
                cache_dir / f"snaps_{season}.parquet",
                lambda s=season: _fetch_snaps(s),
                refresh,
                SNAPS_SCHEMA if tolerate else None,
            ))
    schemas = {"team_epa": TEAM_EPA_SCHEMA, "qb_games": QB_GAMES_SCHEMA,
               "injuries": INJURIES_SCHEMA, "snaps": SNAPS_SCHEMA}
    return {
        k: pl.concat([p.cast(schemas[k]) for p in v]) if v else pl.DataFrame(schema=schemas[k])
        for k, v in parts.items()
    }
