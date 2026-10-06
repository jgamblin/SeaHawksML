"""Paths and season constants shared across the project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
STADIUMS_CSV = DATA_DIR / "stadiums.csv"
FEATURES_PATH = DATA_DIR / "features" / "games.parquet"
MODELS_DIR = ROOT / "models"
PREDICTIONS_PATH = ROOT / "predictions" / "history.jsonl"
LEAGUE_PATH = ROOT / "predictions" / "league.jsonl"
SEASON_SIM_PATH = ROOT / "predictions" / "season_sim.jsonl"
SITE_DIR = ROOT / "public"

FIRST_RATING_SEASON = 2002  # ratings warm up from here (32-team era)
FIRST_TRAIN_SEASON = 2009  # first season with injury reports
FIRST_SNAP_SEASON = 2013  # first season nflverse has snap counts
BACKTEST_SEASONS = tuple(range(2012, 2024))
HOLDOUT_SEASONS = (2024, 2025)

TEAM = "SEA"
# Relocated franchises are tracked under their current abbreviation.
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
