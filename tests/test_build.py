import polars as pl
import pytest

from seahawks_ml.features.build import build_features
from seahawks_ml.features.columns import FEATURE_COLUMNS, ID_COLUMNS
from seahawks_ml.stadiums import load_stadiums
from tests.synthetic import make_raw


@pytest.fixture(scope="module")
def stadiums():
    return load_stadiums()


def test_build_features_schema(stadiums):
    raw = make_raw()
    frame = build_features(raw, stadiums)
    assert frame.columns == ID_COLUMNS + FEATURE_COLUMNS
    assert frame.height == raw.games.height
    assert frame["game_id"].n_unique() == frame.height
    for col in FEATURE_COLUMNS:
        assert frame[col].null_count() == 0, col
        assert frame[col].dtype.is_numeric(), col


def test_build_features_rows_for_unplayed_games(stadiums):
    raw = make_raw(unplayed_last_week=True)
    frame = build_features(raw, stadiums)
    future = frame.filter(pl.col("margin").is_null())
    assert future.height == 2
    assert future["elo_diff"].null_count() == 0


@pytest.mark.parametrize("row_index", [10, 30, 45])
def test_no_leakage_features_match_as_of_kickoff(stadiums, row_index):
    """Features for a game must be identical whether or not later data exists."""
    raw = make_raw()
    target = raw.games.row(row_index, named=True)
    full = build_features(raw, stadiums).filter(pl.col("game_id") == target["game_id"])
    cut = build_features(raw.as_of(target["kickoff_utc"]), stadiums).filter(
        pl.col("game_id") == target["game_id"])
    for col in FEATURE_COLUMNS:
        assert full[col][0] == pytest.approx(cut[col][0]), col


def test_availability_known_requires_both_sides(stadiums, monkeypatch):
    import seahawks_ml.features.build as build

    raw = make_raw()
    real = build.compute_availability
    target = raw.games.filter(pl.col("season") == 2014).row(0, named=True)["game_id"]

    def patched(*args, **kwargs):
        out = real(*args, **kwargs)
        return out.with_columns(
            pl.when(pl.col("game_id") == target).then(None).otherwise(pl.col("away_off_out"))
            .alias("away_off_out"))

    monkeypatch.setattr(build, "compute_availability", patched)
    frame = build.build_features(raw, stadiums)
    assert frame.filter(pl.col("game_id") == target)["availability_known"][0] == 0
    assert frame["availability_known"].sum() > 0
