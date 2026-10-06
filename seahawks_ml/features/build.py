"""Join every feature module into one row per game."""

import polars as pl

from seahawks_ml.data import RawData
from seahawks_ml.features.availability import GROUPS, AvailabilityParams, compute_availability
from seahawks_ml.features.base import team_games
from seahawks_ml.features.coaching import new_head_coach
from seahawks_ml.features.columns import FEATURE_COLUMNS, ID_COLUMNS
from seahawks_ml.features.elo import compute_elo
from seahawks_ml.features.qb import QBParams, compute_qb_features
from seahawks_ml.features.ratings import RATING_COLUMNS, RatingParams, compute_team_ratings
from seahawks_ml.features.season_timing import compute_season_timing
from seahawks_ml.features.situational import compute_situational
from seahawks_ml.features.weather import compute_weather
from seahawks_ml.stadiums import Stadium


def _home_away(games: pl.DataFrame, per_team: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Turn a (game_id, team, cols...) frame into home_<col>/away_<col> per game."""
    out = games.select("game_id", "home_team", "away_team")
    for side in ("home", "away"):
        renamed = per_team.select(
            "game_id", pl.col("team").alias(f"{side}_team"), *[pl.col(c).alias(f"{side}_{c}") for c in cols]
        )
        out = out.join(renamed, on=["game_id", f"{side}_team"], how="left")
    return out.drop("home_team", "away_team")


def build_features(
    raw: RawData,
    stadiums: dict[str, Stadium],
    rating_params: RatingParams = RatingParams(),
    qb_params: QBParams = QBParams(),
    availability_params: AvailabilityParams = AvailabilityParams(),
) -> pl.DataFrame:
    games = raw.games
    tg = team_games(games)
    coach = new_head_coach(tg)
    ratings = compute_team_ratings(tg, raw.team_epa, coach, rating_params)

    frame = (
        games.select("game_id", "season", "week", "game_type", "kickoff_utc", "home_team",
                     "away_team", "neutral", "margin", "spread_line")
        .join(compute_elo(games), on="game_id")
        .join(_home_away(games, ratings, RATING_COLUMNS), on="game_id")
        .join(_home_away(games, coach, ["new_head_coach"]), on="game_id")
        .join(compute_qb_features(games, raw.qb_games, raw.players, qb_params), on="game_id")
        .join(compute_situational(games, stadiums), on="game_id")
        .join(compute_season_timing(games), on="game_id")
        .join(compute_availability(games, raw.snaps, raw.injuries, raw.players, raw.player_stats,
                                   availability_params), on="game_id")
        .join(compute_weather(games, raw.weather), on="game_id")
    )
    # Each availability mode feeds one representation; the other's columns are 0.0.
    avail = ["home_off_out", "home_def_out", "away_off_out", "away_def_out"]
    count_mode = availability_params.mode == "count"
    frame = frame.with_columns(
        (pl.col("elo_home_pre") - pl.col("elo_away_pre")).alias("elo_diff"),
        (pl.col("home_off_rating") - pl.col("away_off_rating")).alias("off_rating_diff"),
        (pl.col("home_def_rating") - pl.col("away_def_rating")).alias("def_rating_diff"),
        *[(pl.col(f"home_{p}_{s}_rating") - pl.col(f"away_{p}_{s}_rating")).alias(f"{p}_{s}_diff")
          for p in ("pass", "rush", "success") for s in ("off", "def")],
        (pl.col("home_qb_epa") - pl.col("away_qb_epa")).alias("qb_epa_diff"),
        (pl.col("home_qb_cpoe") - pl.col("away_qb_cpoe")).alias("qb_cpoe_diff"),
        *[(pl.col(f"{s}_qb_bucket") == b).cast(pl.Int64).alias(f"{s}_qb_{b}")
          for s in ("home", "away") for b in ("round_1", "day_2")],
        pl.col("home_new_head_coach").alias("home_new_coach"),
        pl.col("away_new_head_coach").alias("away_new_coach"),
        (pl.col("home_off_out").is_not_null() & pl.col("away_off_out").is_not_null())
        .cast(pl.Int64).alias("availability_known"),
        *[(pl.col(c).fill_null(0.0) if count_mode else pl.lit(0.0)).alias(c) for c in avail],
        *[(pl.lit(0.0) if count_mode else (pl.col(f"home_out_{g}") - pl.col(f"away_out_{g}")).fill_null(0.0))
          .alias(f"out_{g}_diff") for g in GROUPS],
    )
    return frame.select(ID_COLUMNS + FEATURE_COLUMNS).sort("kickoff_utc", "game_id")
