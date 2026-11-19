"""Render the static dashboard (public/index.html + public/data.json)."""

import json
from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from seahawks_ml.config import LEAGUE_PATH, PREDICTIONS_PATH, SEASON_SIM_PATH, SITE_DIR
from seahawks_ml.models.store import BACKTEST_PATH, HOLDOUT_PATH, METRICS_PATH
from seahawks_ml.pipeline.history import read_history, scored_predictions
from seahawks_ml.pipeline.league import league_scorecard
from seahawks_ml.pipeline.season_sim import snapshot_day
from seahawks_ml.site.changes import trajectory_changes
from seahawks_ml.site.labels import FEATURE_LABELS

TEMPLATES = Path(__file__).parent / "templates"


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def season_outlook(sim_log: list[dict]) -> dict | None:
    """Latest season-simulation snapshot plus the P(playoffs) series for that season."""
    if not sim_log:
        return None
    season = max(r["season"] for r in sim_log)
    snaps = sorted((r for r in sim_log if r["season"] == season), key=lambda r: _ts(r["as_of"]))
    # games_remaining is absent on old snapshots: treat those as not final
    return {"latest": snaps[-1], "final": snaps[-1].get("games_remaining") == 0,
            "series": [{"date": snapshot_day(r), "p_playoffs": r["p_playoffs"]} for r in snaps]}


def seahawks_schedule(games, now: datetime, team: str = "SEA") -> list[dict]:
    """The team's current-season games (from a prepared games frame) as plain dicts for the season log."""
    from seahawks_ml.data import current_season
    from seahawks_ml.pipeline.gate import team_schedule

    season = current_season(games, now)
    out = []
    for g in team_schedule(games.filter(games["season"] == season), team).iter_rows(named=True):
        home = g["home_team"] == team and not g["neutral"]
        us, them = (g["home_score"], g["away_score"]) if g["home_team"] == team else (g["away_score"], g["home_score"])
        out.append({"game_id": g["game_id"], "kickoff_utc": g["kickoff_utc"].isoformat(),
                    "opponent": g["away_team"] if g["home_team"] == team else g["home_team"],
                    "seahawks_home": home, "seahawks_score": us, "opponent_score": them})
    return out


def _unpredicted_games(schedule: list[dict], predicted: set[str], now: datetime) -> list[dict]:
    out = []
    for s in schedule:
        if s["game_id"] in predicted:
            continue
        scored = s.get("seahawks_score") is not None and s.get("opponent_score") is not None
        result = {"seahawks_score": s["seahawks_score"], "opponent_score": s["opponent_score"],
                  "margin_seahawks": s["seahawks_score"] - s["opponent_score"]} if scored else None
        out.append({"game_id": s["game_id"], "opponent": s["opponent"], "kickoff_utc": s["kickoff_utc"],
                    "seahawks_home": s["seahawks_home"], "trajectory": [], "changes": [], "latest": None,
                    "result": result, "status": "not_predicted" if _ts(s["kickoff_utc"]) <= now else "scheduled"})
    return out


def _week(game_id: str) -> int | None:
    """Week from an nflverse id (SEASON_WEEK_AWAY_HOME); None for ids without one."""
    parts = game_id.split("_")
    return int(parts[1]) if len(parts) == 4 and parts[1].isdigit() else None


def _regular_season(schedule: list[dict]) -> list[dict]:
    return sorted((s for s in schedule if (_week(s["game_id"]) or 99) <= 18), key=lambda s: s["kickoff_utc"])


def bye_week(schedule: list[dict], predicted_upcoming: bool, now: datetime) -> dict | None:
    """The Seahawks' bye, while it's the current gap: after the last game before a skipped
    regular-season week and until the next game has its first prediction."""
    from seahawks_ml.pipeline.gate import RUN_WINDOWS

    games = _regular_season(schedule)
    for prev, nxt in zip(games, games[1:]):
        gap = _week(nxt["game_id"]) - _week(prev["game_id"])
        if gap > 1 and _ts(prev["kickoff_utc"]) < now < _ts(nxt["kickoff_utc"]) and not predicted_upcoming:
            return {"week": _week(prev["game_id"]) + 1, "next_opponent": nxt["opponent"],
                    "next_home": nxt["seahawks_home"], "next_kickoff_utc": nxt["kickoff_utc"],
                    "first_prediction_utc": (_ts(nxt["kickoff_utc"]) - RUN_WINDOWS["midweek"][0]).isoformat()}
    return None


def _bye_rows(schedule: list[dict]) -> list[dict]:
    """Season-log rows for skipped regular-season weeks, placed between the games around them."""
    games = _regular_season(schedule)
    rows = []
    for prev, nxt in zip(games, games[1:]):
        for wk in range(_week(prev["game_id"]) + 1, _week(nxt["game_id"])):
            rows.append({"game_id": f"bye-{wk}", "bye_week": wk, "kickoff_utc": prev["kickoff_utc"] + "~bye",
                         "status": "bye", "latest": None, "result": None, "trajectory": [], "changes": []})
    return rows


def build_site_data(history: list[dict], now: datetime, league_log: list[dict] | None = None,
                    sim_log: list[dict] | None = None, schedule: list[dict] | None = None) -> dict:
    predictions = [r for r in history if r["type"] == "prediction"]
    results = {r["game_id"]: r for r in history if r["type"] == "result"}
    games: dict[str, dict] = {}
    for p in sorted(predictions, key=lambda r: r["predicted_at"]):
        g = games.setdefault(p["game_id"], {"game_id": p["game_id"], "opponent": p["opponent"],
                                            "kickoff_utc": p["kickoff_utc"], "seahawks_home": p["seahawks_home"],
                                            "trajectory": [], "_runs": []})
        g["_runs"].append(p)
        g["trajectory"].append({"run_type": p["run_type"], "predicted_at": p["predicted_at"],
                                "p_seahawks": p["p_seahawks"], "margin_seahawks": p["margin_seahawks"],
                                "p_vegas_seahawks": p["p_vegas_seahawks"]})
        g.update(opponent=p["opponent"], kickoff_utc=p["kickoff_utc"], seahawks_home=p["seahawks_home"])
        g["latest"] = p
        g["result"] = results.get(p["game_id"])
    for g in games.values():
        g["changes"] = trajectory_changes(g.pop("_runs"))
        g["status"] = "predicted"
    upcoming = [g for g in games.values() if g["result"] is None and _ts(g["kickoff_utc"]) > now]
    next_game = min(upcoming, key=lambda g: g["kickoff_utc"]) if upcoming else None
    for p in [next_game["latest"]] if next_game else []:
        for f in p["top_factors"]:
            f["label"] = FEATURE_LABELS.get(f["feature"], f["feature"])
    decided = [s for s in scored_predictions(history) if s["result"]["margin_seahawks"] != 0]  # ties excluded
    record = {"games": len(decided),
              "correct": sum((s["prediction"]["p_seahawks"] > 0.5) == (s["result"]["margin_seahawks"] > 0)
                             for s in decided)}
    backtest = _read_json(BACKTEST_PATH)
    if backtest:
        backtest = {k: v for k, v in backtest.items() if k != "trials"}  # keep the page small
    league_seasons = [r["season"] for r in league_log or [] if r["type"] == "league_prediction"]
    league = league_scorecard(league_log or [], max(league_seasons)) if league_seasons else None
    return {
        "generated_at": now.isoformat(),
        "next_game": next_game,
        "season_log": sorted([*games.values(), *_unpredicted_games(schedule or [], set(games), now),
                              *_bye_rows(schedule or [])], key=lambda g: g["kickoff_utc"]),
        "bye": bye_week(schedule or [], next_game is not None, now),
        "record": record,
        "metrics": _read_json(METRICS_PATH),
        "backtest": backtest,
        "holdout": _read_json(HOLDOUT_PATH),
        "league": league,
        "season_sim": season_outlook(sim_log or []),
    }


def build_site(now: datetime | None = None, history_path: Path = PREDICTIONS_PATH,
               out_dir: Path = SITE_DIR, league_path: Path | None = None, sim_path: Path | None = None,
               schedule: list[dict] | None = None) -> Path:
    now = now or datetime.now(UTC)
    league_path = LEAGUE_PATH if league_path is None else league_path  # looked up at call time
    sim_path = SEASON_SIM_PATH if sim_path is None else sim_path
    data = build_site_data(read_history(history_path), now, read_history(league_path), read_history(sim_path),
                           schedule)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.json").write_text(json.dumps(data, indent=2, default=str))
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]),
                      extensions=["jinja2.ext.loopcontrols"])
    # Escape "<" so no string in the data can close the inline <script> block.
    data_json = json.dumps(data, default=str).replace("<", "\\u003c")
    html = env.get_template("index.html.j2").render(data=data, data_json=data_json)
    (out_dir / "index.html").write_text(html)
    return out_dir / "index.html"
