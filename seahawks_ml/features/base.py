"""Canonical games table and the per-team long view used by most feature modules."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import polars as pl

from seahawks_ml.config import FIRST_RATING_SEASON
from seahawks_ml.stadiums import Stadium, validate_stadiums

ET = ZoneInfo("America/New_York")  # nflverse gametime is US Eastern

GAMES_SCHEMA = {
    "game_id": pl.Utf8,
    "season": pl.Int64,
    "week": pl.Int64,
    "game_type": pl.Utf8,
    "kickoff_utc": pl.Datetime("us", "UTC"),
    "home_team": pl.Utf8,
    "away_team": pl.Utf8,
    "home_score": pl.Int64,
    "away_score": pl.Int64,
    "margin": pl.Int64,  # home score minus away score
    "neutral": pl.Boolean,
    "roof": pl.Utf8,
    "stadium_id": pl.Utf8,
    "home_rest": pl.Int64,
    "away_rest": pl.Int64,
    "div_game": pl.Boolean,
    "home_qb_id": pl.Utf8,
    "away_qb_id": pl.Utf8,
    "home_coach": pl.Utf8,
    "away_coach": pl.Utf8,
    "spread_line": pl.Float64,  # positive = home favored (benchmark only)
}

TEAM_GAMES_SCHEMA = {
    "game_id": pl.Utf8,
    "season": pl.Int64,
    "week": pl.Int64,
    "kickoff_utc": pl.Datetime("us", "UTC"),
    "team": pl.Utf8,
    "opponent": pl.Utf8,
    "is_home": pl.Boolean,
    "points_for": pl.Int64,
    "points_against": pl.Int64,
    "coach": pl.Utf8,
    "qb_id": pl.Utf8,
}


def kickoff_utc(gameday: str, gametime: str) -> datetime:
    local = datetime.strptime(f"{gameday} {gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    return local.astimezone(UTC)


def resolve_roof(roof: str | None, stadium: Stadium) -> str:
    """Use the schedule's roof value; fall back to the stadium default (retractable -> closed)."""
    if roof:
        return roof
    return "closed" if stadium.roof_default == "retractable" else stadium.roof_default


def prepare_games(schedules: pl.DataFrame, stadiums: dict[str, Stadium]) -> pl.DataFrame:
    """Convert team-normalized nflverse schedules into the canonical games table."""
    sched = schedules.filter(pl.col("season") >= FIRST_RATING_SEASON)
    validate_stadiums(sched["stadium_id"].to_list(), stadiums)
    rows = []
    for r in sched.iter_rows(named=True):
        rows.append(
            {
                "game_id": r["game_id"],
                "season": r["season"],
                "week": r["week"],
                "game_type": r["game_type"],
                "kickoff_utc": kickoff_utc(r["gameday"], r["gametime"]),
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "home_score": r["home_score"],
                "away_score": r["away_score"],
                "margin": r["result"],
                "neutral": r["location"] == "Neutral",
                "roof": resolve_roof(r["roof"], stadiums[r["stadium_id"]]),
                "stadium_id": r["stadium_id"],
                "home_rest": r["home_rest"],
                "away_rest": r["away_rest"],
                "div_game": bool(r["div_game"]),
                "home_qb_id": r["home_qb_id"],
                "away_qb_id": r["away_qb_id"],
                "home_coach": r["home_coach"],
                "away_coach": r["away_coach"],
                "spread_line": r["spread_line"],
            }
        )
    return pl.DataFrame(rows, schema=GAMES_SCHEMA).sort("kickoff_utc", "game_id")


def team_games(games: pl.DataFrame) -> pl.DataFrame:
    """Two rows per game, one from each team's perspective."""
    rows = []
    for g in games.iter_rows(named=True):
        for side, other in (("home", "away"), ("away", "home")):
            rows.append(
                {
                    "game_id": g["game_id"],
                    "season": g["season"],
                    "week": g["week"],
                    "kickoff_utc": g["kickoff_utc"],
                    "team": g[f"{side}_team"],
                    "opponent": g[f"{other}_team"],
                    "is_home": side == "home",
                    "points_for": g[f"{side}_score"],
                    "points_against": g[f"{other}_score"],
                    "coach": g[f"{side}_coach"],
                    "qb_id": g[f"{side}_qb_id"],
                }
            )
    return pl.DataFrame(rows, schema=TEAM_GAMES_SCHEMA).sort("kickoff_utc", "game_id", "team")
