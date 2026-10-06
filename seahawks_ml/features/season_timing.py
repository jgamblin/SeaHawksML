"""Week number, final-regular-week and playoff flags."""

import polars as pl


def compute_season_timing(games: pl.DataFrame) -> pl.DataFrame:
    final_week = (
        games.filter(pl.col("game_type") == "REG")
        .group_by("season")
        .agg(pl.col("week").max().alias("final_week"))
    )
    return (
        games.join(final_week, on="season", how="left")
        .select(
            "game_id",
            pl.col("week").alias("week_number"),
            ((pl.col("game_type") == "REG") & (pl.col("week") == pl.col("final_week")))
            .cast(pl.Int64).alias("is_final_regular_week"),
            (pl.col("game_type") != "REG").cast(pl.Int64).alias("is_playoff"),
        )
    )
