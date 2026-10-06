import polars as pl
import pytest

from seahawks_ml.features.availability import AvailabilityParams
from seahawks_ml.features.build import build_features
from seahawks_ml.features.columns import FEATURE_COLUMNS, ID_COLUMNS
from seahawks_ml.features.ratings import RatingParams
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


EXTRA_DIFFS = ["pass_off_diff", "pass_def_diff", "rush_off_diff", "rush_def_diff",
               "success_off_diff", "success_def_diff"]
RATING_VARIANTS = [
    RatingParams(),
    RatingParams(opponent_adjust=True),
    RatingParams(0.5, 4.0, 2.0, opponent_adjust=True, extra_stats=False),
]


def test_extra_stat_columns_toggle(stadiums):
    raw = make_raw()
    assert set(EXTRA_DIFFS) <= set(FEATURE_COLUMNS)
    on = build_features(raw, stadiums, RatingParams(extra_stats=True))
    off = build_features(raw, stadiums, RatingParams(extra_stats=False))
    assert off.columns == on.columns
    for c in EXTRA_DIFFS:
        assert (off[c] == 0.0).all(), c
        assert on[c].abs().sum() > 0, c
    assert on["off_rating_diff"].to_list() == off["off_rating_diff"].to_list()


OLD_AVAIL = ["home_off_out", "home_def_out", "away_off_out", "away_def_out"]
GROUP_DIFFS = ["out_ol_diff", "out_wrte_diff", "out_rb_diff", "out_dl_diff", "out_lb_diff", "out_db_diff"]
AVAIL_VARIANTS = [AvailabilityParams(mode=m) for m in ("count", "groups", "values")]


def test_availability_modes_use_one_representation(stadiums):
    raw = make_raw()
    assert set(OLD_AVAIL + GROUP_DIFFS) <= set(FEATURE_COLUMNS)
    count, groups, values = (build_features(raw, stadiums, availability_params=p) for p in AVAIL_VARIANTS)
    assert count.columns == groups.columns == values.columns
    for c in GROUP_DIFFS:
        assert (count[c] == 0.0).all(), c
    for c in OLD_AVAIL:
        assert (groups[c] == 0.0).all() and (values[c] == 0.0).all(), c
    assert sum(count[c].abs().sum() for c in OLD_AVAIL) > 0
    for c in ("out_wrte_diff", "out_lb_diff"):  # synthetic injuries never hit the RB
        assert groups[c].abs().sum() > 0, c
        assert values[c].to_list() != groups[c].to_list(), c
    # home minus away of the per-team group weights
    from seahawks_ml.features.availability import compute_availability

    av = compute_availability(raw.games, raw.snaps, raw.injuries, raw.players, raw.player_stats,
                              AVAIL_VARIANTS[1])
    joined = groups.join(av, on="game_id").filter(pl.col("home_out_wrte").is_not_null()
                                                   & pl.col("away_out_wrte").is_not_null())
    assert joined.height > 0
    assert (joined["out_wrte_diff"] - (joined["home_out_wrte"] - joined["away_out_wrte"])).abs().max() < 1e-12
    for frame in (count, groups, values):
        assert frame["availability_known"].to_list() == count["availability_known"].to_list()


@pytest.mark.parametrize("params,avail", [*[(p, AvailabilityParams()) for p in RATING_VARIANTS],
                                          *[(RatingParams(), a) for a in AVAIL_VARIANTS[1:]]])
@pytest.mark.parametrize("row_index", [10, 30, 40, 45])
def test_no_leakage_features_match_as_of_kickoff(stadiums, row_index, params, avail):
    """Features for a game must be identical whether or not later data exists."""
    raw = make_raw()
    target = raw.games.row(row_index, named=True)
    full = build_features(raw, stadiums, params, availability_params=avail).filter(
        pl.col("game_id") == target["game_id"])
    cut = build_features(raw.as_of(target["kickoff_utc"]), stadiums, params, availability_params=avail).filter(
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


def test_every_feature_column_has_a_label():
    from seahawks_ml.site.labels import FEATURE_LABELS

    assert [c for c in FEATURE_COLUMNS if c not in FEATURE_LABELS] == []
