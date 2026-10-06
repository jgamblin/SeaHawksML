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


def _league_log():
    from tests.test_league import _pred as lp
    from tests.test_league import _res
    return [lp("g0", 1, .8, .7, .6), _res("g0", 7), lp("g1", 2, .4, .55, .5), _res("g1", -3),
            lp("g2", 3, .6, .6, .6)]


def test_site_league_scorecard(tmp_path):
    data = build_site_data([], datetime(2026, 10, 12, tzinfo=UTC), _league_log())
    assert data["league"]["season"] == 2026 and data["league"]["n"] == 2
    assert build_site_data([], datetime(2026, 10, 12, tzinfo=UTC), [])["league"] is None


def test_build_site_renders_league_card(tmp_path):
    from seahawks_ml.pipeline.league import validate_league
    lpath = tmp_path / "l.jsonl"
    for r in _league_log():
        append_record(r, lpath, validate_league)
    html = build_site(datetime(2026, 10, 12, tzinfo=UTC), history_path=tmp_path / "none.jsonl",
                      out_dir=tmp_path / "s", league_path=lpath).read_text()
    assert "Live this season" in html and 'id="leaguechart"' in html and "never edited" in html


def test_build_site_league_empty_state(tmp_path):
    html = build_site(datetime(2026, 10, 12, tzinfo=UTC), history_path=tmp_path / "none.jsonl",
                      out_dir=tmp_path / "s", league_path=tmp_path / "none2.jsonl").read_text()
    assert "No completed games yet" in html and 'id="leaguechart"' not in html


def test_build_site_reads_league_path_at_call_time(tmp_path, monkeypatch):
    from seahawks_ml.pipeline.league import validate_league
    from seahawks_ml.site import build
    lpath = tmp_path / "l.jsonl"
    for r in _league_log():
        append_record(r, lpath, validate_league)
    monkeypatch.setattr(build, "LEAGUE_PATH", lpath)
    html = build_site(datetime(2026, 10, 12, tzinfo=UTC), history_path=tmp_path / "none.jsonl",
                      out_dir=tmp_path / "s").read_text()
    assert 'id="leaguechart"' in html


def test_site_data_has_no_model_files_by_default(tmp_path):
    data = build_site_data([], datetime(2026, 10, 12, tzinfo=UTC))
    assert data["metrics"] is None and data["backtest"] is None and data["holdout"] is None


def _render_league(tmp_path, log):
    from seahawks_ml.pipeline.league import validate_league
    lpath = tmp_path / "lg.jsonl"
    for r in log:
        append_record(r, lpath, validate_league)
    return build_site(datetime(2026, 10, 12, tzinfo=UTC), history_path=tmp_path / "none.jsonl",
                      out_dir=tmp_path / "s", league_path=lpath).read_text()


def test_league_card_wording_without_vegas_lines(tmp_path):
    from tests.test_league import _pred as lp
    from tests.test_league import _res
    html = _render_league(tmp_path, [lp("g0", 1, .8, None, .6), _res("g0", 7)])
    card = html[html.index('id="league-h"'):html.index('id="leaguechart"')]
    assert "have a Vegas line" not in card and "Vegas spread" not in card
    assert "1 completed game" in card


def test_league_card_wording_with_vegas_lines(tmp_path):
    html = _render_league(tmp_path, _league_log())
    assert "have a Vegas line" in html


def test_league_card_shows_dash_for_missing_metrics(tmp_path):
    from tests.test_league import _pred as lp
    from tests.test_league import _res
    html = _render_league(tmp_path, [lp("g0", 1, .6, .5, .5), _res("g0", 0)])  # all ties: accuracy undefined
    card = html[html.index('id="league-h"'):html.index('id="leaguechart"')]
    assert "nan" not in card.lower() and "None" not in card and "&mdash;" in card


def _sim_log():
    from tests.test_cli import _sim_snapshot
    old = {**_sim_snapshot(datetime(2025, 12, 1, tzinfo=UTC)), "season": 2025, "p_playoffs": 0.9}
    a = {**_sim_snapshot(datetime(2026, 10, 5, 0, 17, tzinfo=UTC)), "season": 2026, "p_playoffs": 0.41}
    b = {**_sim_snapshot(datetime(2026, 10, 6, 0, 17, tzinfo=UTC)), "season": 2026, "p_playoffs": 0.47,
         "wins_mean": 9.4, "wins_p10": 7.0, "wins_p90": 12.0, "p_division": 0.18, "p_top_seed": 0.004,
         "record_now": "3-2-0", "win_dist": [{"wins": w, "prob": p} for w, p in [(7, .2), (9, .5), (12, .3)]]}
    return [old, b, a]


def test_site_data_season_sim(tmp_path):
    data = build_site_data([], datetime(2026, 10, 6, 12, tzinfo=UTC), sim_log=_sim_log())
    sim = data["season_sim"]
    assert sim["latest"]["as_of"].startswith("2026-10-06") and sim["latest"]["wins_mean"] == 9.4
    assert sim["series"] == [{"date": "2026-10-05", "p_playoffs": 0.41}, {"date": "2026-10-06", "p_playoffs": 0.47}]
    assert build_site_data([], datetime(2026, 10, 6, tzinfo=UTC))["season_sim"] is None


def _render_sim(tmp_path, log):
    from seahawks_ml.pipeline.season_sim import validate_sim
    path = tmp_path / "sim.jsonl"
    for r in log:
        append_record(r, path, validate_sim)
    return build_site(datetime(2026, 10, 6, 12, tzinfo=UTC), history_path=tmp_path / "none.jsonl",
                      out_dir=tmp_path / "s", sim_path=path).read_text()


def test_build_site_renders_season_outlook(tmp_path):
    html = _render_sim(tmp_path, _sim_log())
    card = html[html.index('id="outlook-h"'):html.index('id="traj-h"')]
    assert "Season outlook" in html and 'id="winsdist"' in card and 'id="playoffchart"' in card
    assert "47%" in card and "18%" in card and "&lt;1%" in card and "9.4" in card and "7–12" in card
    assert "3-2" in card and "Simplified tiebreakers" in card


def test_build_site_season_outlook_empty_state(tmp_path):
    html = _render_sim(tmp_path, [])
    assert "Season outlook" in html and "No simulation yet" in html and 'id="winsdist"' not in html


def test_build_site_reads_sim_path_at_call_time(tmp_path, monkeypatch):
    from seahawks_ml.pipeline.season_sim import validate_sim
    from seahawks_ml.site import build
    path = tmp_path / "sim.jsonl"
    for r in _sim_log():
        append_record(r, path, validate_sim)
    monkeypatch.setattr(build, "SEASON_SIM_PATH", path)
    html = build_site(datetime(2026, 10, 6, tzinfo=UTC), history_path=tmp_path / "none.jsonl",
                      out_dir=tmp_path / "s").read_text()
    assert 'id="winsdist"' in html


def _site_with_holdout(tmp_path, monkeypatch, holdout):
    import json

    from seahawks_ml.site import build
    m = {"log_loss": 0.65, "log_loss_ci90": [0.6, 0.7], "brier": 0.23, "accuracy": 0.6}
    score = {k: dict(m) for k in ("model", "elo", "home", "vegas")}
    score["seahawks_only"] = None
    score["per_season"] = []
    bt, ho = tmp_path / "backtest.json", tmp_path / "holdout.json"
    bt.write_text(json.dumps({"seasons": [2020, 2021], "score": score}))
    ho.write_text(json.dumps({"seasons": [2024, 2025], "score": {"model": m, "vegas": m}, **holdout}))
    monkeypatch.setattr(build, "BACKTEST_PATH", bt)
    monkeypatch.setattr(build, "HOLDOUT_PATH", ho)
    out = build_site(datetime(2026, 10, 10, 19, tzinfo=UTC), history_path=_history(tmp_path), out_dir=tmp_path / "site")
    return out.read_text()


def test_holdout_tile_flags_previously_viewed(tmp_path, monkeypatch):
    html = _site_with_holdout(tmp_path, monkeypatch, {"previously_viewed": True})
    assert "Seen before — reference only" in html
    assert "Live this season" in html and "honest test" in html


def test_holdout_tile_not_flagged_on_first_run(tmp_path, monkeypatch):
    html = _site_with_holdout(tmp_path, monkeypatch, {"previously_viewed": False})
    assert "Seen before" not in html
    assert "Locked holdout" in html


def _schedule():
    def g(gid, kick, opp, home, us=None, them=None):
        return {"game_id": gid, "kickoff_utc": kick, "opponent": opp, "seahawks_home": home,
                "seahawks_score": us, "opponent_score": them}
    return [g("g0", "2026-09-27T20:25:00+00:00", "LA", False, 10, 13),   # past, unpredicted, loss
            g("g1", "2026-10-04T20:25:00+00:00", "SF", True, 24, 17),    # predicted
            g("g2", "2026-10-11T20:25:00+00:00", "SF", True),            # predicted, future
            g("g3", "2026-10-18T20:25:00+00:00", "ARI", False),          # future, unpredicted
            g("g4", "2026-10-25T13:30:00+00:00", "WAS", False)]          # neutral site => not home


def test_season_log_includes_unpredicted_games(tmp_path):
    from seahawks_ml.pipeline.history import read_history
    now = datetime(2026, 10, 10, 19, tzinfo=UTC)
    data = build_site_data(read_history(_history(tmp_path)), now, schedule=_schedule())
    log = data["season_log"]
    assert [g["game_id"] for g in log] == ["g0", "g1", "g2", "g3", "g4"]
    assert [g["status"] for g in log] == ["not_predicted", "predicted", "predicted", "scheduled", "scheduled"]
    assert log[0]["result"] == {"seahawks_score": 10, "opponent_score": 13, "margin_seahawks": -3}
    assert log[0]["trajectory"] == [] and log[3]["result"] is None
    assert data["record"] == {"games": 1, "correct": 1}  # unpredicted games never count
    assert data["next_game"]["game_id"] == "g2"


def test_build_site_renders_unpredicted_markers(tmp_path):
    out = build_site(datetime(2026, 10, 10, 19, tzinfo=UTC), history_path=_history(tmp_path),
                     out_dir=tmp_path / "site", schedule=_schedule())
    html = out.read_text()
    assert "Not predicted" in html and "Scheduled" in html and "ARI" in html


def test_seahawks_schedule_from_games_frame():
    import polars as pl

    from seahawks_ml.site.build import seahawks_schedule
    rows = [("a", "2026-09-27", "SEA", "LA", 10, 13, False), ("b", "2026-10-04", "SF", "SEA", 17, 24, False),
            ("c", "2026-10-11", "SEA", "WAS", None, None, True), ("d", "2026-10-11", "KC", "DEN", 1, 2, False),
            ("e", "2025-12-28", "SEA", "SF", 3, 1, False)]
    seasons = [2026, 2026, 2026, 2026, 2025]
    games = pl.DataFrame({"game_id": [r[0] for r in rows], "season": seasons,
                          "kickoff_utc": [datetime.fromisoformat(r[1]).replace(tzinfo=UTC) for r in rows],
                          "home_team": [r[2] for r in rows], "away_team": [r[3] for r in rows],
                          "home_score": [r[4] for r in rows], "away_score": [r[5] for r in rows],
                          "neutral": [r[6] for r in rows]})
    sched = seahawks_schedule(games, datetime(2026, 10, 5, tzinfo=UTC))
    assert [s["game_id"] for s in sched] == ["a", "b", "c"]
    assert sched[0]["seahawks_home"] is True and sched[0]["seahawks_score"] == 10
    assert sched[1]["seahawks_home"] is False and sched[1]["seahawks_score"] == 24 and sched[1]["opponent"] == "SF"
    assert sched[2]["seahawks_home"] is False and sched[2]["seahawks_score"] is None
