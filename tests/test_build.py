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
POOLED_DIFFS = ["off_out_diff", "def_out_diff"]
AVAIL_VARIANTS = [AvailabilityParams(mode=m) for m in ("count", "groups", "values", "count_diff", "values_pooled")]


def test_availability_modes_use_one_representation(stadiums):
    raw = make_raw()
    assert set(OLD_AVAIL + GROUP_DIFFS) <= set(FEATURE_COLUMNS)
    count, groups, values = (build_features(raw, stadiums, availability_params=p) for p in AVAIL_VARIANTS[:3])
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


def test_pooled_modes_use_only_the_two_diffs_and_sum_group_diffs(stadiums):
    from seahawks_ml.features.availability import compute_availability

    raw = make_raw()
    assert set(POOLED_DIFFS) <= set(FEATURE_COLUMNS)
    frames = {p.mode: build_features(raw, stadiums, availability_params=p) for p in AVAIL_VARIANTS}
    unused = {"count": GROUP_DIFFS + POOLED_DIFFS, "groups": OLD_AVAIL + POOLED_DIFFS,
              "values": OLD_AVAIL + POOLED_DIFFS, "count_diff": OLD_AVAIL + GROUP_DIFFS,
              "values_pooled": OLD_AVAIL + GROUP_DIFFS}
    for mode, cols in unused.items():
        for c in cols:
            assert (frames[mode][c] == 0.0).all(), (mode, c)
    cd, vp = frames["count_diff"], frames["values_pooled"]
    for c in POOLED_DIFFS:
        assert cd[c].abs().sum() > 0 and vp[c].abs().sum() > 0, c
    # count_diff: home minus away of the snap-share counts
    cnt = compute_availability(raw.games, raw.snaps, raw.injuries, raw.players, raw.player_stats,
                               AvailabilityParams(mode="count_diff"))
    cnt = cnt.select("game_id", *[pl.col(c).alias(f"c_{c}") for c in OLD_AVAIL])
    j = cd.join(cnt, on="game_id").filter(pl.col("availability_known") == 1)
    assert j.height > 0
    assert (j["off_out_diff"] - (j["c_home_off_out"] - j["c_away_off_out"])).abs().max() < 1e-12
    assert (j["def_out_diff"] - (j["c_home_def_out"] - j["c_away_def_out"])).abs().max() < 1e-12
    # values_pooled: sums of the per-group value diffs (same weights as "values")
    grp = build_features(raw, stadiums, availability_params=AvailabilityParams(mode="values"))
    # (the frames share column names, so compare against a renamed copy)
    g = grp.select("game_id", *[pl.col(c).alias(f"g_{c}") for c in GROUP_DIFFS])
    j = vp.join(g, on="game_id").filter(pl.col("availability_known") == 1)
    off = j["g_out_ol_diff"] + j["g_out_wrte_diff"] + j["g_out_rb_diff"]
    de = j["g_out_dl_diff"] + j["g_out_lb_diff"] + j["g_out_db_diff"]
    assert (j["off_out_diff"] - off).abs().max() < 1e-12
    assert (j["def_out_diff"] - de).abs().max() < 1e-12


def _assert_no_leakage(raw, stadiums, target, params, avail):
    full = build_features(raw, stadiums, params, availability_params=avail).filter(
        pl.col("game_id") == target["game_id"])
    cut = build_features(raw.as_of(target["kickoff_utc"]), stadiums, params, availability_params=avail).filter(
        pl.col("game_id") == target["game_id"])
    for col in FEATURE_COLUMNS:
        assert full[col][0] == pytest.approx(cut[col][0]), col
    return full.row(0, named=True)


@pytest.mark.parametrize("params", RATING_VARIANTS)
@pytest.mark.parametrize("row_index", [10, 30, 40, 45])
def test_no_leakage_features_match_as_of_kickoff(stadiums, row_index, params):
    """Features for a game must be identical whether or not later data exists."""
    raw = make_raw()
    _assert_no_leakage(raw, stadiums, raw.games.row(row_index, named=True), params, AvailabilityParams())


@pytest.mark.parametrize("avail", AVAIL_VARIANTS[1:])  # every mode but "count"
@pytest.mark.parametrize("row_index", [30, 40, 45])  # snap era (2013+), so availability is known
def test_no_leakage_availability_modes(stadiums, row_index, avail):
    """Same check for the group / player-value features. The tested game's reports are replaced by a
    starting home WR listed Out (his value depends on his stats before kickoff), so the diff is not zero."""
    from dataclasses import replace

    from seahawks_ml.ingest.nflverse import INJURIES_SCHEMA

    raw = make_raw()
    target = raw.games.row(row_index, named=True)
    week = {"season": target["season"], "week": target["week"]}
    injected = pl.DataFrame([
        week | {"team": target["home_team"], "gsis_id": f"{target['home_team']}-P0", "position": "WR",
                "report_status": "Out"},
        week | {"team": target["away_team"], "gsis_id": "x", "position": "LB", "report_status": "Questionable"},
    ], schema=INJURIES_SCHEMA)
    teams = [target["home_team"], target["away_team"]]
    others = raw.injuries.filter(~((pl.col("season") == target["season"]) & (pl.col("week") == target["week"])
                                   & pl.col("team").is_in(teams)))
    raw = replace(raw, injuries=pl.concat([others, injected]))
    row = _assert_no_leakage(raw, stadiums, target, RatingParams(), avail)
    assert row["availability_known"] == 1
    if avail.mode in ("groups", "values"):
        assert row["out_wrte_diff"] != 0.0
    else:
        assert row["off_out_diff"] != 0.0


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
