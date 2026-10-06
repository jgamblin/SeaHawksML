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
from seahawks_ml.pipeline.predict import latest_injury_week, make_prediction
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
