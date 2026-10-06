"""Render the static dashboard (public/index.html + public/data.json)."""

import json
from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from seahawks_ml.config import PREDICTIONS_PATH, SITE_DIR
from seahawks_ml.models.store import BACKTEST_PATH, HOLDOUT_PATH, METRICS_PATH
from seahawks_ml.pipeline.history import read_history, scored_predictions

TEMPLATES = Path(__file__).parent / "templates"
FEATURE_LABELS = {
    "home_field": "Home field", "hfa_trend": "Home-field trend", "no_crowd": "No-crowd season",
    "elo_diff": "Elo gap", "off_rating_diff": "Offense rating gap", "def_rating_diff": "Defense rating gap",
    "qb_epa_diff": "QB EPA gap", "qb_cpoe_diff": "QB accuracy gap",
    "home_qb_round_1": "Home QB 1st-rounder", "home_qb_day_2": "Home QB day-2 pick",
    "away_qb_round_1": "Away QB 1st-rounder", "away_qb_day_2": "Away QB day-2 pick",
    "home_new_coach": "Home new coach", "away_new_coach": "Away new coach",
    "home_off_out": "Home offensive starters out", "home_def_out": "Home defensive starters out",
    "away_off_out": "Away offensive starters out", "away_def_out": "Away defensive starters out",
    "availability_known": "Injury data available", "rest_diff": "Rest advantage",
    "home_post_bye": "Home off bye", "away_post_bye": "Away off bye",
    "home_short_week": "Home short week", "away_short_week": "Away short week",
    "travel_diff": "Travel gap", "home_tz_shift": "Home time-zone shift", "away_tz_shift": "Away time-zone shift",
    "home_body_clock": "Home body clock", "away_body_clock": "Away body clock", "primetime": "Primetime",
    "div_game": "Division game", "week_number": "Week of season", "is_final_regular_week": "Final week",
    "is_playoff": "Playoffs", "is_indoor": "Indoors", "temp_f": "Temperature", "wind_mph": "Wind",
    "precip_in": "Precipitation",
}


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def build_site_data(history: list[dict], now: datetime) -> dict:
    predictions = [r for r in history if r["type"] == "prediction"]
    results = {r["game_id"]: r for r in history if r["type"] == "result"}
    games: dict[str, dict] = {}
    for p in sorted(predictions, key=lambda r: r["predicted_at"]):
        g = games.setdefault(p["game_id"], {"game_id": p["game_id"], "opponent": p["opponent"],
                                            "kickoff_utc": p["kickoff_utc"], "seahawks_home": p["seahawks_home"],
                                            "trajectory": []})
        g["trajectory"].append({"run_type": p["run_type"], "predicted_at": p["predicted_at"],
                                "p_seahawks": p["p_seahawks"], "margin_seahawks": p["margin_seahawks"],
                                "p_vegas_seahawks": p["p_vegas_seahawks"]})
        g["latest"] = p
        g["result"] = results.get(p["game_id"])
    upcoming = [g for g in games.values() if g["result"] is None]
    next_game = min(upcoming, key=lambda g: g["kickoff_utc"]) if upcoming else None
    for p in [next_game["latest"]] if next_game else []:
        for f in p["top_factors"]:
            f["label"] = FEATURE_LABELS.get(f["feature"], f["feature"])
    scored = scored_predictions(history)
    record = {"games": len(scored),
              "correct": sum((s["prediction"]["p_seahawks"] > 0.5) == (s["result"]["margin_seahawks"] > 0)
                             for s in scored if s["result"]["margin_seahawks"] != 0)}
    return {
        "generated_at": now.isoformat(),
        "next_game": next_game,
        "season_log": sorted(games.values(), key=lambda g: g["kickoff_utc"]),
        "record": record,
        "metrics": _read_json(METRICS_PATH),
        "backtest": _read_json(BACKTEST_PATH),
        "holdout": _read_json(HOLDOUT_PATH),
    }


def build_site(now: datetime | None = None, history_path: Path = PREDICTIONS_PATH, out_dir: Path = SITE_DIR) -> Path:
    now = now or datetime.now(UTC)
    data = build_site_data(read_history(history_path), now)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.json").write_text(json.dumps(data, indent=2, default=str))
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    html = env.get_template("index.html.j2").render(data=data, data_json=json.dumps(data, default=str))
    (out_dir / "index.html").write_text(html)
    return out_dir / "index.html"
