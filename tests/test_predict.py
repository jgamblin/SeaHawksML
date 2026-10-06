import json
from datetime import timedelta

import polars as pl
import pytest

from seahawks_ml.features.build import build_features
from seahawks_ml.models.pipeline import ModelConfig, fit_model
from seahawks_ml.pipeline.history import (
    append_record,
    read_history,
    runs_done,
    scored_predictions,
    validate,
)
from seahawks_ml.pipeline.predict import latest_injury_week, make_prediction, single_game_row
from seahawks_ml.pipeline.results import new_results
from seahawks_ml.stadiums import load_stadiums
from tests.synthetic import make_raw


@pytest.fixture(scope="module")
def setup():
    raw = make_raw(seasons=(2009, 2010, 2011, 2012, 2013, 2014), unplayed_last_week=True)
    frame = build_features(raw, load_stadiums())
    model = fit_model(frame, ModelConfig(inner_folds=3), [2009, 2010, 2011, 2012, 2013])
    return raw, frame, model


def _sea_upcoming(frame):
    return frame.filter(pl.col("margin").is_null() & ((pl.col("home_team") == "SEA") | (pl.col("away_team") == "SEA")))


def test_make_prediction_is_seahawks_perspective(setup):
    raw, frame, model = setup
    row = _sea_upcoming(frame)
    now = row["kickoff_utc"][0] - timedelta(hours=3)
    rec = make_prediction(row, model, "gameday", now, "v1", latest_injury_week(raw.injuries, 2014))
    validate(rec)
    p_home = float(model.predict(row)["p_win"][0])
    home = row["home_team"][0] == "SEA"
    assert rec["p_seahawks"] == pytest.approx(p_home if home else 1 - p_home, abs=1e-4)
    assert rec["is_final_injury_report"] is True
    assert rec["margin_lo"] <= rec["margin_hi"]
    assert len(rec["top_factors"]) == 6
    assert rec["opponent"] != "SEA"
    json.dumps(rec)  # must be JSON-serializable


def test_results_recorded_once_and_scored(tmp_path, setup):
    raw, frame, model = setup
    path = tmp_path / "h.jsonl"
    played = frame.filter(pl.col("margin").is_not_null() & (pl.col("home_team") == "SEA")).tail(1)
    ko = played["kickoff_utc"][0]
    early = make_prediction(played, model, "midweek", ko - timedelta(days=4), "v1", None)
    late = make_prediction(played, model, "gameday", ko - timedelta(hours=3), "v1", None)
    for r in (early, late):
        append_record(r, path)
    history = read_history(path)
    assert runs_done(history, early["game_id"]) == {"midweek", "gameday"}
    results = new_results(history, raw.games, ko + timedelta(days=1))
    assert len(results) == 1
    append_record(results[0], path)
    history = read_history(path)
    assert new_results(history, raw.games, ko + timedelta(days=2)) == []
    scored = scored_predictions(history)
    assert len(scored) == 1 and scored[0]["prediction"]["run_type"] == "gameday"


def test_final_injury_flag_requires_matching_week(setup):
    raw, frame, model = setup
    row = _sea_upcoming(frame)
    week = int(row["week"][0])
    now = row["kickoff_utc"][0] - timedelta(hours=3)
    assert make_prediction(row, model, "gameday", now, "v1", week)["is_final_injury_report"] is True
    assert make_prediction(row, model, "gameday", now, "v1", week - 1)["is_final_injury_report"] is False
    assert make_prediction(row, model, "gameday", now, "v1", None)["is_final_injury_report"] is False
    assert make_prediction(row, model, "midweek", now, "v1", week)["is_final_injury_report"] is False


def test_single_game_row(setup):
    _, frame, _ = setup
    gid = frame["game_id"][0]
    assert single_game_row(frame, gid).height == 1
    with pytest.raises(SystemExit, match="exactly one feature row"):
        single_game_row(frame, "nope")
    with pytest.raises(SystemExit, match="exactly one feature row"):
        single_game_row(pl.concat([frame.head(1), frame.head(1)]), gid)


def _home_row(frame):
    return frame.filter((pl.col("home_team") == "SEA") & ~pl.col("neutral")).tail(1)


def test_away_perspective_flips_everything(setup):
    _, frame, model = setup
    home_row = _home_row(frame)
    away_row = home_row.with_columns(pl.lit("SF").alias("home_team"), pl.lit("SEA").alias("away_team"))
    now = home_row["kickoff_utc"][0] - timedelta(hours=3)
    h = make_prediction(home_row, model, "gameday", now, "v1", None)
    a = make_prediction(away_row, model, "gameday", now, "v1", None)
    assert h["seahawks_home"] is True and a["seahawks_home"] is False
    assert a["opponent"] == "SF" and h["opponent"] == home_row["away_team"][0]
    assert a["p_seahawks"] == pytest.approx(1 - h["p_seahawks"], abs=1e-3)
    assert a["margin_seahawks"] == pytest.approx(-h["margin_seahawks"], abs=1e-2)
    assert a["margin_lo"] == pytest.approx(-h["margin_hi"], abs=0.1)
    assert a["margin_hi"] == pytest.approx(-h["margin_lo"], abs=0.1)
    assert a["margin_lo"] <= a["margin_hi"]
    assert a["p_vegas_seahawks"] == pytest.approx(1 - h["p_vegas_seahawks"], abs=1e-3)
    assert [f["feature"] for f in a["top_factors"]] == [f["feature"] for f in h["top_factors"]]
    for fa, fh in zip(a["top_factors"], h["top_factors"], strict=True):
        assert fa["points"] == pytest.approx(-fh["points"], abs=0.01)


def test_neutral_site_is_not_seahawks_home(setup):
    _, frame, model = setup
    home_row = _home_row(frame)
    neutral_row = home_row.with_columns(pl.lit(True).alias("neutral"))
    now = home_row["kickoff_utc"][0] - timedelta(hours=3)
    rec = make_prediction(neutral_row, model, "midweek", now, "v1", None)
    base = make_prediction(home_row, model, "midweek", now, "v1", None)
    assert rec["seahawks_home"] is False and base["seahawks_home"] is True
    # SEA is still the nominal home team in the feature row, so the probability is not flipped
    assert rec["p_seahawks"] == pytest.approx(base["p_seahawks"], abs=1e-3)
    assert rec["margin_lo"] <= rec["margin_hi"]
