"""Team abbreviation normalization."""

import polars as pl

from seahawks_ml.config import TEAM_ALIASES


def normalize_team(col: str) -> pl.Expr:
    """Map historical abbreviations (OAK, SD, STL) to the current franchise abbreviation."""
    return pl.col(col).replace(TEAM_ALIASES)
