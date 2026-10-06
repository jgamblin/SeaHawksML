"""Download nflverse tables and reduce them to compact, per-season cached parquet files.

Completed seasons are cached in data/cache and committed; the current season is
refreshed on every run. Play-by-play is aggregated immediately so the full
372-column table never has to be kept around.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from seahawks_ml.config import CACHE_DIR, FIRST_SNAP_SEASON, FIRST_TRAIN_SEASON
from seahawks_ml.teams import normalize_team

TEAM_EPA_SCHEMA = {
    "game_id": pl.Utf8, "season": pl.Int64, "team": pl.Utf8, "opponent": pl.Utf8,
    "epa_sum": pl.Float64, "plays": pl.Int64,
    "pass_epa_sum": pl.Float64, "pass_plays": pl.Int64,
    "rush_epa_sum": pl.Float64, "rush_plays": pl.Int64,
    "success_sum": pl.Float64,
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
TEAMS_SCHEMA = {"team": pl.Utf8, "conf": pl.Utf8, "division": pl.Utf8}
# Fallback if nflverse's teams table can't be fetched (current abbreviations, as in schedules).
TEAM_DIVISIONS = {
    **{t: ("AFC", "AFC East") for t in ("BUF", "MIA", "NE", "NYJ")},
    **{t: ("AFC", "AFC North") for t in ("BAL", "CIN", "CLE", "PIT")},
    **{t: ("AFC", "AFC South") for t in ("HOU", "IND", "JAX", "TEN")},
    **{t: ("AFC", "AFC West") for t in ("DEN", "KC", "LAC", "LV")},
    **{t: ("NFC", "NFC East") for t in ("DAL", "NYG", "PHI", "WAS")},
    **{t: ("NFC", "NFC North") for t in ("CHI", "DET", "GB", "MIN")},
    **{t: ("NFC", "NFC South") for t in ("ATL", "CAR", "NO", "TB")},
    **{t: ("NFC", "NFC West") for t in ("ARI", "LA", "SEA", "SF")},
}
PLAYERS_SCHEMA = {
    "gsis_id": pl.Utf8, "pfr_id": pl.Utf8, "position": pl.Utf8,
    "draft_round": pl.Int64, "rookie_season": pl.Int64,
}


def aggregate_team_epa(pbp: pl.DataFrame) -> pl.DataFrame:
    """Offensive totals per team-game from scrimmage plays (pass + run, no two-point tries).

    `epa_sum`/`plays` cover all scrimmage plays; `pass_*` uses nflverse `pass == 1`
    (dropbacks, so scrambles and sacks count as passes) and `rush_*` uses `rush == 1`
    (designed runs). `success_sum` counts nflverse `success` over the same plays as `plays`.

    Fumble luck: on plays with `fumble == 1` the EPA is replaced by this frame's (one
    season's) mean EPA over fumble plays, so who happens to recover doesn't move ratings.
    """
    plays = pbp.filter(
        pl.col("play_type").is_in(["pass", "run"])
        & pl.col("epa").is_not_null()
        & pl.col("posteam").is_not_null()
        & (pl.col("two_point_attempt").fill_null(0) != 1)
    )
    fumble = pl.col("fumble").fill_null(0) == 1
    fumble_mean = plays.filter(fumble)["epa"].mean()
    if fumble_mean is not None:
        plays = plays.with_columns(pl.when(fumble).then(pl.lit(fumble_mean)).otherwise(pl.col("epa")).alias("epa"))
    is_pass = pl.col("pass").fill_null(0) == 1
    is_rush = pl.col("rush").fill_null(0) == 1
    return (
        plays.group_by("game_id", "season", "posteam", "defteam")
        .agg(
            pl.col("epa").sum().alias("epa_sum"),
            pl.len().alias("plays"),
            pl.col("epa").filter(is_pass).sum().alias("pass_epa_sum"),
            is_pass.sum().alias("pass_plays"),
            pl.col("epa").filter(is_rush).sum().alias("rush_epa_sum"),
            is_rush.sum().alias("rush_plays"),
            pl.col("success").fill_null(0).sum().alias("success_sum"),
        )
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


def _static_teams() -> pl.DataFrame:
    return pl.DataFrame([(t, c, d) for t, (c, d) in TEAM_DIVISIONS.items()], schema=TEAMS_SCHEMA, orient="row")


def load_teams() -> pl.DataFrame:
    """Conference and division of the 32 current teams (columns team, conf, division).

    Historical abbreviations are normalized and only the 32 current franchises kept. Falls
    back to TEAM_DIVISIONS when the fetch fails or doesn't yield all 32 teams.
    """
    try:
        df = (
            nfl.load_teams()
            .select(pl.col("team_abbr").alias("team"), pl.col("team_conf").alias("conf"),
                    pl.col("team_division").alias("division"))
            .with_columns(normalize_team("team"))
            .cast(TEAMS_SCHEMA)
            .filter(pl.col("team").is_in(list(TEAM_DIVISIONS)))
            .unique("team", keep="first", maintain_order=True)
            .sort("team")
        )
    except Exception as exc:  # noqa: BLE001 - nflreadpy raises several error types
        print(f"warning: nflverse teams fetch failed ({exc}); using built-in divisions")
        return _static_teams()
    if df.height != len(TEAM_DIVISIONS):
        print(f"warning: nflverse teams table has {df.height} current teams; using built-in divisions")
        return _static_teams()
    return df


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
    seasons: list[int],
    current_season: int,
    cache_dir: Path = CACHE_DIR,
    stale_before: dict[int, datetime] | None = None,
) -> dict[str, pl.DataFrame]:
    """Return team_epa, qb_games, injuries and snaps for the given seasons.

    The current season is always refetched. A completed season is also refetched when
    any of its cache files was written before `stale_before[season]`, and its
    play-by-play tables are refetched when the cached team_epa lacks a current column.
    """
    parts: dict[str, list[pl.DataFrame]] = {k: [] for k in ("team_epa", "qb_games", "injuries", "snaps")}
    for season in seasons:
        tolerate = season == current_season  # only the current season may lack data
        cached = list(cache_dir.glob(f"*_{season}.parquet"))
        cutoff = (stale_before or {}).get(season)
        stale = cutoff is not None and any(
            datetime.fromtimestamp(f.stat().st_mtime, tz=cutoff.tzinfo) < cutoff for f in cached
        )
        refresh = tolerate or stale
        team_path = cache_dir / f"team_epa_{season}.parquet"
        qb_path = cache_dir / f"qb_games_{season}.parquet"
        outdated = team_path.exists() and not set(TEAM_EPA_SCHEMA) <= set(pl.read_parquet_schema(team_path))
        if refresh or outdated or not (team_path.exists() and qb_path.exists()):
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
