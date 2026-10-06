"""Production refit with locked hyperparameters (what CI runs - no tuning)."""

from datetime import datetime

import polars as pl

from seahawks_ml.config import FIRST_TRAIN_SEASON
from seahawks_ml.features.columns import FEATURE_COLUMNS
from seahawks_ml.models.pipeline import FittedModel, fit_model
from seahawks_ml.models.store import ProjectConfig


def completed_seasons(frame: pl.DataFrame) -> list[int]:
    done = frame.filter(pl.col("margin").is_not_null() & (pl.col("season") >= FIRST_TRAIN_SEASON))
    return sorted(done["season"].unique().to_list())


def train_production(frame: pl.DataFrame, config: ProjectConfig, now: datetime) -> tuple[FittedModel, dict]:
    seasons = completed_seasons(frame)
    model = fit_model(frame, config.model, seasons)
    n_games = frame.filter(pl.col("season").is_in(seasons) & pl.col("margin").is_not_null()).height
    meta = {
        "model_version": f"{now:%Y%m%d}-{config.fingerprint()}",
        "trained_at": now.isoformat(),
        "train_seasons": seasons,
        "n_games": n_games,
        "config": config.to_dict(),
        "feature_columns": FEATURE_COLUMNS,
    }
    return model, meta
