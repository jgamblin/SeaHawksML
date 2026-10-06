"""Append-only prediction log (predictions/history.jsonl). Records are never rewritten."""

import json
from pathlib import Path

from seahawks_ml.config import PREDICTIONS_PATH

RUN_TYPES = ("midweek", "final_injury", "gameday")
PREDICTION_KEYS = {
    "type", "game_id", "run_type", "predicted_at", "kickoff_utc", "is_final_injury_report",
    "opponent", "seahawks_home", "p_seahawks", "margin_seahawks", "margin_lo", "margin_hi",
    "p_vegas_seahawks", "weather", "latest_injury_week", "top_factors", "model_version",
}
RESULT_KEYS = {"type", "game_id", "recorded_at", "seahawks_score", "opponent_score", "margin_seahawks"}


def read_history(path: Path = PREDICTIONS_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate(record: dict) -> None:
    expected = {"prediction": PREDICTION_KEYS, "result": RESULT_KEYS}.get(record.get("type"))
    if expected is None:
        raise ValueError(f"unknown record type {record.get('type')!r}")
    if set(record) != expected:
        raise ValueError(f"record keys mismatch: missing {expected - set(record)}, extra {set(record) - expected}")
    if record["type"] == "prediction" and record["run_type"] not in RUN_TYPES:
        raise ValueError(f"bad run_type {record['run_type']!r}")


def append_record(record: dict, path: Path = PREDICTIONS_PATH) -> None:
    validate(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def runs_done(history: list[dict], game_id: str) -> set[str]:
    return {r["run_type"] for r in history if r["type"] == "prediction" and r["game_id"] == game_id}


def scored_predictions(history: list[dict]) -> list[dict]:
    """Per game with a result: the last prediction made before kickoff, plus the result."""
    results = {r["game_id"]: r for r in history if r["type"] == "result"}
    last: dict[str, dict] = {}
    for r in history:
        if r["type"] == "prediction" and r["predicted_at"] < r["kickoff_utc"]:
            if r["game_id"] not in last or r["predicted_at"] > last[r["game_id"]]["predicted_at"]:
                last[r["game_id"]] = r
    ordered = sorted(last.items(), key=lambda kv: kv[1]["kickoff_utc"])
    return [{"prediction": p, "result": results[g]} for g, p in ordered if g in results]
