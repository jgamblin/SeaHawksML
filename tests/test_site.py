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
