from seahawks_ml.site.changes import run_changes, trajectory_changes


def _run(run_type, p, margin, **overrides):
    rec = {
        "type": "prediction", "game_id": "g1", "run_type": run_type,
        "predicted_at": "2026-10-07T12:00:00+00:00", "kickoff_utc": "2026-10-11T20:25:00+00:00",
        "is_final_injury_report": False, "opponent": "SF", "seahawks_home": True,
        "p_seahawks": p, "margin_seahawks": margin, "margin_lo": -12.0, "margin_hi": 14.0,
        "p_vegas_seahawks": 0.59,
        "weather": {"source": "forecast", "temp_f": 58.0, "wind_mph": 6.0, "precip_in": 0.0},
        "latest_injury_week": 4,
        "top_factors": [{"feature": "qb_epa_diff", "points": -2.5}, {"feature": "elo_diff", "points": 0.8}],
        "model_version": "v1",
    }
    rec.update(overrides)
    return rec


def test_identical_runs_report_no_input_changes():
    a, b = _run("midweek", 0.48, -0.8), _run("final_injury", 0.48, -0.8)
    out = run_changes(a, b)
    assert out["from_run"] == "midweek" and out["to_run"] == "final_injury"
    assert out["p_delta"] == 0.0 and out["margin_delta"] == 0.0
    assert out["notes"] == ["No meaningful input changes since the last run."]


def test_deltas_are_in_percentage_points_and_points():
    out = run_changes(_run("midweek", 0.48, -0.8), _run("final_injury", 0.531, 0.9))
    assert out["p_delta"] == 5.1
    assert out["margin_delta"] == 1.7


def test_injury_report_and_final_report_notes():
    a = _run("midweek", 0.48, -0.8, latest_injury_week=4)
    b = _run("final_injury", 0.50, 0.0, latest_injury_week=5, is_final_injury_report=True)
    notes = run_changes(a, b)["notes"]
    assert "Final injury report for this week now included" in notes


def test_new_injury_week_without_final_report():
    a = _run("midweek", 0.48, -0.8, latest_injury_week=None)
    b = _run("midweek", 0.48, -0.8, latest_injury_week=5)
    assert "Week 5 injury report now included" in run_changes(a, b)["notes"]


def test_weather_source_and_large_moves_reported_small_moves_ignored():
    a = _run("final_injury", 0.5, 0.0, weather={"source": "climatology", "temp_f": 58.0, "wind_mph": 6.0,
                                                "precip_in": 0.0})
    b = _run("gameday", 0.5, 0.0, weather={"source": "forecast", "temp_f": 60.0, "wind_mph": 15.0,
                                           "precip_in": 0.2})
    notes = run_changes(a, b)["notes"]
    assert "Weather now from the forecast (was a monthly average)" in notes
    assert "Wind 6 → 15 mph" in notes
    assert "Precipitation 0.00 → 0.20 in" in notes
    assert not any(n.startswith("Temperature") for n in notes)  # 2°F is below threshold


def test_vegas_move_reported():
    a = _run("midweek", 0.48, -0.8, p_vegas_seahawks=0.59)
    b = _run("gameday", 0.48, -0.8, p_vegas_seahawks=0.64)
    assert "Vegas benchmark 59% → 64%" in run_changes(a, b)["notes"]


def test_factor_moves_new_and_dropped_factors_largest_first():
    a = _run("midweek", 0.48, -0.8, top_factors=[
        {"feature": "qb_epa_diff", "points": -2.5}, {"feature": "elo_diff", "points": 0.8},
        {"feature": "rest_diff", "points": 0.5}])
    b = _run("final_injury", 0.52, 0.6, top_factors=[
        {"feature": "qb_epa_diff", "points": -1.1}, {"feature": "elo_diff", "points": 0.9},
        {"feature": "away_off_out", "points": 1.2}])
    notes = run_changes(a, b)["notes"]
    assert "QB EPA gap −2.5 → −1.1 pts" in notes
    assert "Away offensive starters out is now a top factor (+1.2 pts)" in notes
    assert "Rest advantage dropped out of the top factors" in notes
    assert not any(n.startswith("Elo gap") for n in notes)  # 0.1 pt move is below threshold
    assert notes.index("QB EPA gap −2.5 → −1.1 pts") < notes.index(
        "Away offensive starters out is now a top factor (+1.2 pts)")


def test_model_retrain_noted():
    notes = run_changes(_run("midweek", 0.5, 0.0), _run("gameday", 0.5, 0.0, model_version="v2"))["notes"]
    assert "Model retrained (v2)" in notes


def test_trajectory_changes_pairs_consecutive_runs():
    runs = [_run("midweek", 0.48, -0.8), _run("final_injury", 0.5, 0.0), _run("gameday", 0.55, 1.5)]
    out = trajectory_changes(runs)
    assert [(c["from_run"], c["to_run"]) for c in out] == [("midweek", "final_injury"), ("final_injury", "gameday")]
    assert trajectory_changes(runs[:1]) == []
