import pytest

from seahawks_ml.pipeline.history import append_record, read_history, runs_done, scored_predictions, validate


def _pred(game_id, run_type, predicted_at, kickoff="2026-10-11T20:25:00+00:00", p=0.5):
    return {"type": "prediction", "game_id": game_id, "run_type": run_type, "predicted_at": predicted_at,
            "kickoff_utc": kickoff, "is_final_injury_report": run_type != "midweek", "opponent": "SF",
            "seahawks_home": True, "p_seahawks": p, "margin_seahawks": 1.0, "margin_lo": -12.0,
            "margin_hi": 14.0, "p_vegas_seahawks": None,
            "weather": {"source": "indoor", "temp_f": 70.0, "wind_mph": 0.0, "precip_in": 0.0},
            "latest_injury_week": None, "top_factors": [], "model_version": "v1"}


def _result(game_id, margin):
    return {"type": "result", "game_id": game_id, "recorded_at": "2026-10-12T12:00:00+00:00",
            "seahawks_score": 20 + max(margin, 0), "opponent_score": 20 + max(-margin, 0), "margin_seahawks": margin}


def test_append_and_read_round_trip(tmp_path):
    path = tmp_path / "h.jsonl"
    assert read_history(path) == []
    append_record(_pred("g1", "midweek", "2026-10-07T12:00:00+00:00"), path)
    append_record(_pred("g1", "gameday", "2026-10-11T17:00:00+00:00"), path)
    history = read_history(path)
    assert len(history) == 2
    assert runs_done(history, "g1") == {"midweek", "gameday"}
    assert runs_done(history, "g2") == set()


def test_scored_predictions_use_last_pre_kickoff_run(tmp_path):
    history = [
        _pred("g1", "midweek", "2026-10-07T12:00:00+00:00", p=0.4),
        _pred("g1", "gameday", "2026-10-11T17:00:00+00:00", p=0.6),
        _pred("g1", "gameday", "2026-10-11T21:00:00+00:00", p=0.9),  # after kickoff: ignored
        _result("g1", 7),
        _pred("g2", "midweek", "2026-10-14T12:00:00+00:00", kickoff="2026-10-18T17:00:00+00:00"),
    ]
    scored = scored_predictions(history)
    assert len(scored) == 1  # g2 has no result yet
    assert scored[0]["prediction"]["p_seahawks"] == 0.6
    assert scored[0]["result"]["margin_seahawks"] == 7


def test_validate_rejects_bad_records():
    validate(_pred("g1", "midweek", "2026-10-07T12:00:00+00:00"))
    with pytest.raises(ValueError):
        validate({"type": "prediction"})
    with pytest.raises(ValueError):
        validate({"type": "nope"})
    with pytest.raises(ValueError):
        validate(_pred("g1", "sunday", "2026-10-07T12:00:00+00:00"))
