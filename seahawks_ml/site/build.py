"""Render the static dashboard (public/index.html + public/data.json)."""

import json
from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from seahawks_ml.config import LEAGUE_PATH, PREDICTIONS_PATH, SITE_DIR
from seahawks_ml.models.store import BACKTEST_PATH, HOLDOUT_PATH, METRICS_PATH
from seahawks_ml.pipeline.history import read_history, scored_predictions
from seahawks_ml.pipeline.league import league_scorecard
from seahawks_ml.site.changes import trajectory_changes
from seahawks_ml.site.labels import FEATURE_LABELS

TEMPLATES = Path(__file__).parent / "templates"


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def build_site_data(history: list[dict], now: datetime, league_log: list[dict] | None = None) -> dict:
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
        "season_log": sorted(games.values(), key=lambda g: g["kickoff_utc"]),
        "record": record,
        "metrics": _read_json(METRICS_PATH),
        "backtest": backtest,
        "holdout": _read_json(HOLDOUT_PATH),
        "league": league,
    }


def build_site(now: datetime | None = None, history_path: Path = PREDICTIONS_PATH,
               out_dir: Path = SITE_DIR, league_path: Path = LEAGUE_PATH) -> Path:
    now = now or datetime.now(UTC)
    data = build_site_data(read_history(history_path), now, read_history(league_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.json").write_text(json.dumps(data, indent=2, default=str))
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    # Escape "<" so no string in the data can close the inline <script> block.
    data_json = json.dumps(data, default=str).replace("<", "\\u003c")
    html = env.get_template("index.html.j2").render(data=data, data_json=data_json)
    (out_dir / "index.html").write_text(html)
    return out_dir / "index.html"
