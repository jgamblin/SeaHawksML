from datetime import UTC, datetime

from seahawks_ml.pipeline.history import append_record
from seahawks_ml.site.build import build_site, build_site_data


def _pred(game_id, run_type, predicted_at, kickoff, p):
    return {"type": "prediction", "game_id": game_id, "run_type": run_type, "predicted_at": predicted_at,
            "kickoff_utc": kickoff, "is_final_injury_report": run_type != "midweek", "opponent": "SF",
            "seahawks_home": True, "p_seahawks": p, "margin_seahawks": 2.5, "margin_lo": -10.0,
            "margin_hi": 15.0, "p_vegas_seahawks": 0.55,
            "weather": {"source": "forecast", "temp_f": 55.0, "wind_mph": 8.0, "precip_in": 0.1},
            "latest_injury_week": 5, "top_factors": [{"feature": "elo_diff", "points": 1.2}],
            "model_version": "v1"}


def _history(tmp_path):
    path = tmp_path / "h.jsonl"
    append_record(_pred("g1", "gameday", "2026-10-04T17:00:00+00:00", "2026-10-04T20:25:00+00:00", 0.6), path)
    append_record({"type": "result", "game_id": "g1", "recorded_at": "2026-10-05T12:00:00+00:00",
                   "seahawks_score": 24, "opponent_score": 17, "margin_seahawks": 7}, path)
    append_record(_pred("g2", "midweek", "2026-10-07T12:00:00+00:00", "2026-10-11T20:25:00+00:00", 0.52), path)
    append_record(_pred("g2", "final_injury", "2026-10-10T18:00:00+00:00", "2026-10-11T20:25:00+00:00", 0.48), path)
    return path


def test_site_data_picks_next_game_and_trajectory(tmp_path):
    from seahawks_ml.pipeline.history import read_history
    data = build_site_data(read_history(_history(tmp_path)), datetime(2026, 10, 10, 19, tzinfo=UTC))
    assert data["next_game"]["game_id"] == "g2"
    assert [t["run_type"] for t in data["next_game"]["trajectory"]] == ["midweek", "final_injury"]
    assert data["next_game"]["latest"]["top_factors"][0]["label"] == "Elo gap"
    assert data["record"] == {"games": 1, "correct": 1}


def test_build_site_writes_html(tmp_path):
    out = build_site(datetime(2026, 10, 10, 19, tzinfo=UTC), history_path=_history(tmp_path), out_dir=tmp_path / "site")
    html = out.read_text()
    assert "Seahawks win probability" in html and "48%" in html
    assert (tmp_path / "site" / "data.json").exists()


def test_build_site_with_empty_history(tmp_path):
    out = build_site(datetime(2026, 10, 10, tzinfo=UTC), history_path=tmp_path / "none.jsonl", out_dir=tmp_path / "s")
    assert "No upcoming prediction yet" in out.read_text()


def test_record_excludes_ties(tmp_path):
    from seahawks_ml.pipeline.history import read_history
    path = _history(tmp_path)
    append_record(_pred("g3", "gameday", "2026-10-10T17:00:00+00:00", "2026-10-10T20:25:00+00:00", 0.7), path)
    append_record({"type": "result", "game_id": "g3", "recorded_at": "2026-10-11T00:00:00+00:00",
                   "seahawks_score": 20, "opponent_score": 20, "margin_seahawks": 0}, path)
    data = build_site_data(read_history(path), datetime(2026, 10, 11, 1, tzinfo=UTC))
    assert data["record"] == {"games": 1, "correct": 1}


def test_next_game_ignores_games_already_kicked_off(tmp_path):
    from seahawks_ml.pipeline.history import read_history
    # g2 kicked off 10-11 20:25 and has no result yet: it is not "next"
    data = build_site_data(read_history(_history(tmp_path)), datetime(2026, 10, 12, tzinfo=UTC))
    assert data["next_game"] is None


def test_backtest_trials_dropped(tmp_path, monkeypatch):
    import json

    from seahawks_ml.pipeline.history import read_history
    from seahawks_ml.site import build
    bt = tmp_path / "backtest.json"
    bt.write_text(json.dumps({"seasons": [2020], "score": {"n": 1}, "trials": [{"x": 1}] * 5}))
    monkeypatch.setattr(build, "BACKTEST_PATH", bt)
    data = build_site_data(read_history(_history(tmp_path)), datetime(2026, 10, 10, 19, tzinfo=UTC))
    assert data["backtest"] == {"seasons": [2020], "score": {"n": 1}}


def test_site_data_uses_latest_kickoff(tmp_path):
    from seahawks_ml.pipeline.history import read_history
    path = tmp_path / "h.jsonl"
    append_record(_pred("g2", "midweek", "2026-10-07T12:00:00+00:00", "2026-10-11T20:25:00+00:00", 0.52), path)
    moved = _pred("g2", "final_injury", "2026-10-10T18:00:00+00:00", "2026-10-12T01:15:00+00:00", 0.48)
    moved["opponent"], moved["seahawks_home"] = "LA", False
    append_record(moved, path)
    data = build_site_data(read_history(path), datetime(2026, 10, 10, 19, tzinfo=UTC))
    g = data["season_log"][0]
    assert (g["kickoff_utc"], g["opponent"], g["seahawks_home"]) == ("2026-10-12T01:15:00+00:00", "LA", False)


def test_site_data_includes_changes_between_runs(tmp_path):
    from seahawks_ml.pipeline.history import read_history
    data = build_site_data(read_history(_history(tmp_path)), datetime(2026, 10, 10, 19, tzinfo=UTC))
    changes = data["next_game"]["changes"]
    assert len(changes) == 1
    assert (changes[0]["from_run"], changes[0]["to_run"]) == ("midweek", "final_injury")
    assert changes[0]["p_delta"] == -4.0
    assert "Final injury report for this week now included" in changes[0]["notes"]
    single = next(g for g in data["season_log"] if g["game_id"] == "g1")
    assert single["changes"] == []


def test_build_site_renders_change_notes(tmp_path):
    out = build_site(datetime(2026, 10, 10, 19, tzinfo=UTC), history_path=_history(tmp_path), out_dir=tmp_path / "s")
    html = out.read_text()
    assert "What changed" in html
    assert "Final injury report for this week now included" in html
