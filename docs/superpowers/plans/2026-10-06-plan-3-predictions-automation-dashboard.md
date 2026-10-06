# Plan 3 of 3: Prediction Runs, Automation, and Dashboard

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Predict each Seahawks game at three times before kickoff, log every run append-only, record results, publish a static dashboard, and automate it all with GitHub Actions.

**Architecture:** `pipeline/gate.py` decides from the schedule and history alone whether a `midweek`, `final_injury` or `gameday` run is due, so the hourly cron is cheap. `pipeline/predict.py` turns a feature row into a Seahawks-perspective record, `pipeline/history.py` appends it to `predictions/history.jsonl`, and `pipeline/results.py` adds outcomes. `site/build.py` renders `public/` (deployed as a Pages artifact, so `docs/` stays for documentation).

**Tech Stack:** Python 3.12, uv, polars, nflreadpy, httpx, scikit-learn, LightGBM, scipy, Jinja2, pytest, GitHub Actions + Pages.

**Spec:** `docs/superpowers/specs/2026-10-06-seahawks-win-model-design.md`

---

**Prerequisite:** Plans 1 and 2 complete (`models/config.json`, `models/model.pkl`, `models/metrics.json` exist).

## Decisions made while verifying against real data

- **Run windows** are measured back from kickoff: `midweek` from 4.5 days to 2 days before, `final_injury` from 30 hours to 6 hours before, and `gameday` from 4 hours to 90 minutes before. If the Seahawks' previous game kicked off less than 6 hours earlier, the gate waits so that game's result is in first. This covers Thursday, Monday and international games.
- **`is_final_injury_report`** is true for the `final_injury` and `gameday` runs. Each record also stores `latest_injury_week`, the latest injury-report week seen for SEA. Open-Meteo doesn't expose a forecast issue time, so `predicted_at` serves as the weather snapshot time.
- **Pages** is deployed from a build artifact (`public/`) rather than a committed `docs/` folder, so the specs and plans aren't published.
- **CI stays light:** the data cache is committed, so CI only refreshes the current season. Retraining uses the locked config, and no hyperparameter search runs in CI.

## File map

| File | Responsibility |
|---|---|
| `seahawks_ml/pipeline/gate.py` | Next game, previous kickoff, due run type |
| `seahawks_ml/pipeline/history.py` | Append-only JSONL log, schema validation, scoring rule |
| `seahawks_ml/pipeline/predict.py` | Seahawks-perspective prediction record + top factors |
| `seahawks_ml/pipeline/results.py` | Result records for finished, predicted games |
| `seahawks_ml/site/build.py`, `site/templates/index.html.j2` | Dashboard data + HTML |
| `seahawks_ml/cli.py` | Adds `predict`, `build-site` |
| `.github/workflows/{ci,predict,retrain}.yml` | Automation |
| `README.md` | How to run everything |

---

### Task 1: Run gate

**Files:**
- Create/replace: `seahawks_ml/pipeline/gate.py`
- Test: `tests/test_gate.py`

Pure function of (now, kickoff, runs already done, previous kickoff). Hourly cron runs land in a window; the first one in it does the run.

- [ ] **Step 1: Write the failing test**

`tests/test_gate.py`:

```python
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from seahawks_ml.pipeline.gate import due_run, next_game, previous_kickoff

SUN_1PM_ET = datetime(2026, 10, 11, 17, 0, tzinfo=UTC)
THU_820_ET = datetime(2026, 10, 16, 0, 15, tzinfo=UTC)
LONDON_930_ET = datetime(2026, 10, 25, 13, 30, tzinfo=UTC)


@pytest.mark.parametrize("kickoff", [SUN_1PM_ET, THU_820_ET, LONDON_930_ET])
@pytest.mark.parametrize("before,expected", [
    (timedelta(days=5), None),
    (timedelta(days=4), "midweek"),
    (timedelta(days=2, hours=1), "midweek"),
    (timedelta(days=1, hours=12), None),
    (timedelta(hours=24), "final_injury"),
    (timedelta(hours=5), None),
    (timedelta(hours=3), "gameday"),
    (timedelta(minutes=60), None),
])
def test_due_run_windows(kickoff, before, expected):
    assert due_run(kickoff - before, kickoff, done=set()) == expected


def test_due_run_skips_runs_already_done():
    assert due_run(SUN_1PM_ET - timedelta(hours=3), SUN_1PM_ET, done={"gameday"}) is None


def test_due_run_waits_for_previous_game_result():
    prev = THU_820_ET - timedelta(days=4)  # Sunday game before a Thursday game
    now = prev + timedelta(hours=2)
    assert due_run(now, THU_820_ET, set(), prev_kickoff=prev) is None
    assert due_run(prev + timedelta(hours=7), THU_820_ET, set(), prev_kickoff=prev) == "midweek"


def test_next_and_previous_game():
    games = pl.DataFrame({
        "game_id": ["a", "b", "c"],
        "home_team": ["SEA", "SF", "SEA"],
        "away_team": ["LA", "SEA", "KC"],
        "kickoff_utc": [SUN_1PM_ET - timedelta(days=7), SUN_1PM_ET, SUN_1PM_ET + timedelta(days=7)],
    }, schema_overrides={"kickoff_utc": pl.Datetime("us", "UTC")})
    now = SUN_1PM_ET - timedelta(days=1)
    assert next_game(games, "SEA", now)["game_id"] == "b"
    assert previous_kickoff(games, "SEA", now) == SUN_1PM_ET - timedelta(days=7)
    assert next_game(games, "SEA", SUN_1PM_ET + timedelta(days=8)) is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_gate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.pipeline.gate'`

- [ ] **Step 3: Write the implementation**

`seahawks_ml/pipeline/gate.py`:

```python
"""Decide whether a prediction run is due. Cheap: needs only the schedule and history.

Windows are relative to kickoff, so Thursday/Monday/Saturday and international games
work like Sunday games. Hourly cron runs land inside these windows; the first one in a
window does the run, later ones see it in history and skip.
"""

from datetime import datetime, timedelta

import polars as pl

# run_type -> (opens this long before kickoff, closes this long before kickoff)
RUN_WINDOWS = {
    "midweek": (timedelta(days=4, hours=12), timedelta(days=2)),
    "final_injury": (timedelta(hours=30), timedelta(hours=6)),
    "gameday": (timedelta(hours=4), timedelta(minutes=90)),
}
RESULT_GRACE = timedelta(hours=6)  # wait for the previous game's result before predicting


def team_schedule(games: pl.DataFrame, team: str) -> pl.DataFrame:
    return games.filter((pl.col("home_team") == team) | (pl.col("away_team") == team)).sort("kickoff_utc")


def next_game(games: pl.DataFrame, team: str, now: datetime) -> dict | None:
    upcoming = team_schedule(games, team).filter(pl.col("kickoff_utc") > now)
    return upcoming.row(0, named=True) if upcoming.height else None


def previous_kickoff(games: pl.DataFrame, team: str, now: datetime) -> datetime | None:
    past = team_schedule(games, team).filter(pl.col("kickoff_utc") <= now)
    return past["kickoff_utc"][-1] if past.height else None


def due_run(now: datetime, kickoff: datetime, done: set[str], prev_kickoff: datetime | None = None) -> str | None:
    if prev_kickoff is not None and now - prev_kickoff < RESULT_GRACE:
        return None
    for run_type, (opens, closes) in RUN_WINDOWS.items():
        if kickoff - opens <= now < kickoff - closes:
            return None if run_type in done else run_type
    return None
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_gate.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add seahawks_ml/pipeline/gate.py tests/test_gate.py
git commit -m "feat: prediction run gate"
```

---

### Task 2: Append-only prediction history

**Files:**
- Create/replace: `seahawks_ml/pipeline/history.py`
- Test: `tests/test_history.py`

Records are validated against fixed key sets and only ever appended. The track record scores the **last prediction made before kickoff** for each game.

- [ ] **Step 1: Write the failing test**

`tests/test_history.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_history.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.pipeline.history'`

- [ ] **Step 3: Write the implementation**

`seahawks_ml/pipeline/history.py`:

```python
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
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_history.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add seahawks_ml/pipeline/history.py tests/test_history.py
git commit -m "feat: append-only prediction history"
```

---

### Task 3: Prediction records and results

**Files:**
- Create/replace: `seahawks_ml/pipeline/predict.py`
- Create/replace: `seahawks_ml/pipeline/results.py`
- Test: `tests/test_predict.py`

Everything is flipped to the Seahawks' perspective: probability, margin, interval, Vegas benchmark, and factor signs.

- [ ] **Step 1: Write the failing test**

`tests/test_predict.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_predict.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.pipeline.predict'`

- [ ] **Step 3: Write the implementation**

`seahawks_ml/pipeline/predict.py`:

```python
"""Turn one feature row + the fitted model into a Seahawks-perspective prediction record."""

from datetime import datetime

import numpy as np
import polars as pl

from seahawks_ml.config import TEAM
from seahawks_ml.features.columns import FEATURE_COLUMNS
from seahawks_ml.models.pipeline import FittedModel

TOP_FACTORS = 6


def top_factors(model: FittedModel, row: pl.DataFrame, seahawks_home: bool, k: int = TOP_FACTORS) -> list[dict]:
    """Largest per-feature contributions in points, signed for the Seahawks."""
    contrib = model.contributions(row)[0] * (1 if seahawks_home else -1)
    order = np.argsort(-np.abs(contrib))[:k]
    return [{"feature": FEATURE_COLUMNS[i], "points": round(float(contrib[i]), 2)} for i in order]


def latest_injury_week(injuries: pl.DataFrame, season: int, team: str = TEAM) -> int | None:
    weeks = injuries.filter((pl.col("season") == season) & (pl.col("team") == team))["week"]
    return int(weeks.max()) if weeks.len() else None


def make_prediction(
    row: pl.DataFrame,
    model: FittedModel,
    run_type: str,
    now: datetime,
    model_version: str,
    injury_week: int | None,
) -> dict:
    g = row.row(0, named=True)
    home = g["home_team"] == TEAM
    out = model.predict(row)
    p_home, m = float(out["p_win"][0]), float(out["margin"][0])
    lo, hi = float(out["margin_lo"][0]), float(out["margin_hi"][0])
    spread = g["spread_line"]
    p_vegas_home = float(model.vegas_prob(np.array([spread]))[0]) if spread is not None else None
    return {
        "type": "prediction",
        "game_id": g["game_id"],
        "run_type": run_type,
        "predicted_at": now.isoformat(),
        "kickoff_utc": g["kickoff_utc"].isoformat(),
        "is_final_injury_report": run_type in ("final_injury", "gameday"),
        "opponent": g["away_team"] if home else g["home_team"],
        "seahawks_home": home and not g["neutral"],
        "p_seahawks": round(p_home if home else 1 - p_home, 4),
        "margin_seahawks": round(m if home else -m, 2),
        "margin_lo": round(lo if home else -hi, 1),
        "margin_hi": round(hi if home else -lo, 1),
        "p_vegas_seahawks": None if p_vegas_home is None else round(p_vegas_home if home else 1 - p_vegas_home, 4),
        "weather": {
            "source": g["weather_source"],
            "temp_f": round(g["temp_f"], 1),
            "wind_mph": round(g["wind_mph"], 1),
            "precip_in": round(g["precip_in"], 2),
        },
        "latest_injury_week": injury_week,
        "top_factors": top_factors(model, row, home),
        "model_version": model_version,
    }
```

`seahawks_ml/pipeline/results.py`:

```python
"""Append result records for predicted Seahawks games that have finished."""

from datetime import datetime

import polars as pl

from seahawks_ml.config import TEAM


def new_results(history: list[dict], games: pl.DataFrame, now: datetime, team: str = TEAM) -> list[dict]:
    predicted = {r["game_id"] for r in history if r["type"] == "prediction"}
    recorded = {r["game_id"] for r in history if r["type"] == "result"}
    pending = predicted - recorded
    out = []
    finished = games.filter(pl.col("game_id").is_in(list(pending)) & pl.col("margin").is_not_null())
    for g in finished.iter_rows(named=True):
        home = g["home_team"] == team
        us, them = (g["home_score"], g["away_score"]) if home else (g["away_score"], g["home_score"])
        out.append({"type": "result", "game_id": g["game_id"], "recorded_at": now.isoformat(),
                    "seahawks_score": us, "opponent_score": them, "margin_seahawks": us - them})
    return out
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_predict.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add seahawks_ml/pipeline/predict.py seahawks_ml/pipeline/results.py tests/test_predict.py
git commit -m "feat: Seahawks-perspective prediction and result records"
```

---

### Task 4: Dashboard

**Files:**
- Create/replace: `seahawks_ml/site/build.py`
- Create/replace: `seahawks_ml/site/templates/index.html.j2`
- Test: `tests/test_site.py`

The page has five parts: the next game (model vs. Vegas benchmark, margin range, weather), the week's trajectory chart, the top factors, the season log scored on the last pre-kickoff run, and the backtest report with a calibration chart. Chart.js comes from cdnjs and the page follows the system light/dark setting.

- [ ] **Step 1: Write the failing test**

`tests/test_site.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_site.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.site.build'`

- [ ] **Step 3: Write the implementation**

`seahawks_ml/site/build.py`:

```python
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
```

`seahawks_ml/site/templates/index.html.j2`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Seahawks Model</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root { --bg:#f7f8f9; --card:#fff; --ink:#1b2430; --muted:#5b6673; --line:#dfe3e8;
          --navy:#002244; --green:#4c8a1e; --grey:#a5acaf; --bad:#b3261e; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0f141a; --card:#171e26; --ink:#e7ecf1; --muted:#9aa6b2; --line:#2a3440;
            --navy:#8fb3d9; --green:#7fc04a; --grey:#7d868c; --bad:#f2827a; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
  main { max-width:960px; margin:0 auto; padding:24px 16px 64px; }
  h1 { font-size:1.6rem; margin:0 0 4px; } h2 { font-size:1.15rem; margin:0 0 12px; }
  .muted { color:var(--muted); font-size:.9rem; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:20px; margin-top:16px; }
  .hero { display:flex; flex-wrap:wrap; gap:24px; align-items:baseline; }
  .big { font-size:3rem; font-weight:700; color:var(--navy); font-variant-numeric:tabular-nums; }
  .stat { min-width:140px; } .stat .label { color:var(--muted); font-size:.85rem; }
  .stat .value { font-size:1.3rem; font-weight:600; font-variant-numeric:tabular-nums; }
  table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
  th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); font-size:.92rem; }
  th { color:var(--muted); font-weight:500; }
  .pos { color:var(--green); } .neg { color:var(--bad); }
  .chart { position:relative; height:260px; }
  .table-wrap { overflow-x:auto; }
</style>
</head>
<body>
<main>
  <h1>Seahawks Game Model</h1>
  <div class="muted">Last updated {{ data.generated_at[:16].replace("T", " ") }} UTC
    {% if data.metrics %}· model {{ data.metrics.model_version }}{% endif %}</div>

  {% set g = data.next_game %}
  <section class="card">
    {% if g %}
      {% set p = g.latest %}
      <h2>Next: {{ "vs" if p.seahawks_home else "at" }} {{ p.opponent }}</h2>
      <div class="muted">Kickoff {{ p.kickoff_utc[:16].replace("T", " ") }} UTC · latest run: {{ p.run_type }}</div>
      <div class="hero">
        <div><div class="big">{{ "%.0f"|format(p.p_seahawks * 100) }}%</div><div class="muted">Seahawks win probability</div></div>
        <div class="stat"><div class="label">Predicted margin</div>
          <div class="value">{{ "%+.1f"|format(p.margin_seahawks) }}</div>
          <div class="muted">80% range {{ "%+.0f"|format(p.margin_lo) }} to {{ "%+.0f"|format(p.margin_hi) }}</div></div>
        <div class="stat"><div class="label">Vegas-implied (benchmark)</div>
          <div class="value">{% if p.p_vegas_seahawks is not none %}{{ "%.0f"|format(p.p_vegas_seahawks * 100) }}%{% else %}n/a{% endif %}</div></div>
        <div class="stat"><div class="label">Weather ({{ p.weather.source }})</div>
          <div class="value">{{ "%.0f"|format(p.weather.temp_f) }}°F</div>
          <div class="muted">wind {{ "%.0f"|format(p.weather.wind_mph) }} mph · precip {{ p.weather.precip_in }} in</div></div>
      </div>
    {% else %}
      <h2>No upcoming prediction yet</h2>
      <div class="muted">The next run happens about four days before kickoff.</div>
    {% endif %}
  </section>

  {% if g %}
  <section class="card">
    <h2>This week's trajectory</h2>
    <div class="chart"><canvas id="trajectory"></canvas></div>
  </section>
  <section class="card">
    <h2>Why: biggest factors (points for Seattle)</h2>
    <table><tbody>
      {% for f in g.latest.top_factors %}
      <tr><td>{{ f.label }}</td><td class="{{ 'pos' if f.points >= 0 else 'neg' }}">{{ "%+.1f"|format(f.points) }}</td></tr>
      {% endfor %}
    </tbody></table>
  </section>
  {% endif %}

  <section class="card">
    <h2>Season log</h2>
    <div class="muted">Scored on the last prediction before kickoff · record {{ data.record.correct }}/{{ data.record.games }}</div>
    <div class="table-wrap"><table>
      <thead><tr><th>Kickoff</th><th>Opponent</th><th>Runs (win %)</th><th>Final pick</th><th>Result</th></tr></thead>
      <tbody>
      {% for game in data.season_log %}
        <tr>
          <td>{{ game.kickoff_utc[:10] }}</td>
          <td>{{ "vs" if game.seahawks_home else "at" }} {{ game.opponent }}</td>
          <td>{% for t in game.trajectory %}{{ "%.0f"|format(t.p_seahawks * 100) }}{% if not loop.last %} → {% endif %}{% endfor %}</td>
          <td>{{ "%.0f"|format(game.latest.p_seahawks * 100) }}%</td>
          <td>{% if game.result %}{{ "W" if game.result.margin_seahawks > 0 else ("L" if game.result.margin_seahawks < 0 else "T") }}
              {{ game.result.seahawks_score }}-{{ game.result.opponent_score }}{% else %}—{% endif %}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table></div>
  </section>

  {% if data.backtest %}
  <section class="card">
    <h2>Model report (walk-forward backtest {{ data.backtest.seasons[0] }}–{{ data.backtest.seasons[-1] }})</h2>
    <div class="table-wrap"><table>
      <thead><tr><th>Method</th><th>Log-loss</th><th>90% CI</th><th>Brier</th><th>Accuracy</th></tr></thead>
      <tbody>
      {% for name, label in [("model", "This model"), ("elo", "Elo only"), ("home", "Home team always"), ("vegas", "Vegas spread (benchmark)")] %}
        {% set m = data.backtest.score[name] %}
        <tr><td>{{ label }}</td><td>{{ "%.4f"|format(m.log_loss) }}</td>
            <td>{{ "%.4f"|format(m.log_loss_ci90[0]) }}–{{ "%.4f"|format(m.log_loss_ci90[1]) }}</td>
            <td>{{ "%.4f"|format(m.brier) }}</td><td>{{ "%.1f"|format(m.accuracy * 100) }}%</td></tr>
      {% endfor %}
      </tbody>
    </table></div>
    {% if data.holdout %}
    <p class="muted">Locked holdout ({{ data.holdout.seasons|join(", ") }}): log-loss {{ "%.4f"|format(data.holdout.score.model.log_loss) }}
      vs Vegas {{ "%.4f"|format(data.holdout.score.vegas.log_loss) }}.</p>
    {% endif %}
    <div class="chart"><canvas id="calibration"></canvas></div>
  </section>
  {% endif %}
  <p class="muted">Data: <a href="https://github.com/nflverse">nflverse</a> ·
    weather by <a href="https://open-meteo.com/">Open-Meteo</a>. An analytical model, not betting advice.</p>
</main>
<script>
const DATA = {{ data_json|safe }};
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
Chart.defaults.color = css("--muted");
Chart.defaults.borderColor = css("--line");
if (DATA.next_game) {
  const t = DATA.next_game.trajectory;
  new Chart(document.getElementById("trajectory"), {
    type: "line",
    data: { labels: t.map(r => r.run_type),
      datasets: [
        { label: "Model", data: t.map(r => r.p_seahawks * 100), borderColor: css("--navy"), backgroundColor: css("--navy"), tension: 0 },
        { label: "Vegas (benchmark)", data: t.map(r => r.p_vegas_seahawks == null ? null : r.p_vegas_seahawks * 100),
          borderColor: css("--grey"), backgroundColor: css("--grey"), borderDash: [4, 4], tension: 0 },
      ] },
    options: { maintainAspectRatio: false, scales: { y: { min: 0, max: 100, title: { display: true, text: "Seahawks win %" } } } },
  });
}
if (DATA.backtest) {
  const cal = DATA.backtest.score.model.calibration;
  new Chart(document.getElementById("calibration"), {
    type: "scatter",
    data: { datasets: [
      { label: "Model (by predicted-probability bin)", data: cal.map(b => ({ x: b.mean_pred * 100, y: b.mean_outcome * 100 })),
        backgroundColor: css("--navy"), pointRadius: 5 },
      { label: "Perfect calibration", type: "line", data: [{ x: 0, y: 0 }, { x: 100, y: 100 }],
        borderColor: css("--grey"), borderDash: [4, 4], pointRadius: 0 },
    ] },
    options: { maintainAspectRatio: false,
      scales: { x: { min: 0, max: 100, title: { display: true, text: "Predicted home win %" } },
                y: { min: 0, max: 100, title: { display: true, text: "Actual home win %" } } } },
  });
}
</script>
</body>
</html>
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_site.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add seahawks_ml/site/build.py seahawks_ml/site/templates/index.html.j2 tests/test_site.py
git commit -m "feat: static dashboard"
```

---

### Task 5: `predict` and `build-site` commands plus end-to-end smoke test

**Files:**
- Create/replace: `seahawks_ml/cli.py`
- Test: `tests/test_smoke.py`

Replaces `seahawks_ml/cli.py` with the final version. `predict` records new results, asks the gate, and only then downloads everything, pulls the forecast (if kickoff is within 16 days), predicts, and appends. It writes `changed=true` to `$GITHUB_OUTPUT` when the history changed, so the workflow knows to commit and deploy.

- [ ] **Step 1: Write the end-to-end smoke test**

`tests/test_smoke.py`:

```python
"""End-to-end on synthetic data: features -> train -> predict -> log -> site."""

from datetime import timedelta

import polars as pl

from seahawks_ml.features.build import build_features
from seahawks_ml.models.pipeline import ModelConfig
from seahawks_ml.models.store import ProjectConfig, load_model, save_model
from seahawks_ml.models.train import train_production
from seahawks_ml.pipeline.gate import due_run, next_game
from seahawks_ml.pipeline.history import append_record, read_history, runs_done
from seahawks_ml.pipeline.predict import latest_injury_week, make_prediction
from seahawks_ml.site.build import build_site
from seahawks_ml.stadiums import load_stadiums
from tests.synthetic import make_raw


def test_pipeline_end_to_end(tmp_path):
    stadiums = load_stadiums()
    raw = make_raw(seasons=(2009, 2010, 2011, 2012, 2013, 2014), unplayed_last_week=True)
    config = ProjectConfig(model=ModelConfig(inner_folds=3))
    frame = build_features(raw, stadiums, config.rating, config.qb)

    model, meta = train_production(frame, config, now=raw.games["kickoff_utc"].max())
    save_model(model, tmp_path / "model.pkl")
    model = load_model(tmp_path / "model.pkl")

    game = next_game(raw.games, "SEA", raw.games.filter(pl.col("margin").is_null())["kickoff_utc"].min()
                     - timedelta(hours=3))
    history_path = tmp_path / "history.jsonl"
    now = game["kickoff_utc"] - timedelta(hours=3)
    run_type = due_run(now, game["kickoff_utc"], runs_done(read_history(history_path), game["game_id"]))
    assert run_type == "gameday"

    row = frame.filter(pl.col("game_id") == game["game_id"])
    record = make_prediction(row, model, run_type, now, meta["model_version"],
                             latest_injury_week(raw.injuries, game["season"]))
    append_record(record, history_path)
    assert runs_done(read_history(history_path), game["game_id"]) == {"gameday"}

    html = build_site(now, history_path=history_path, out_dir=tmp_path / "site").read_text()
    assert f"{record['p_seahawks'] * 100:.0f}%" in html
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: PASS — every module it uses already exists. It guards the whole chain from here on.

- [ ] **Step 3: Replace the CLI**

`seahawks_ml/cli.py`:

```python
"""Command-line entry point: python -m seahawks_ml.cli <command>.

Local (offseason / manual):  features, backtest, holdout
CI (in season):              retrain, predict, build-site
"""

import argparse
import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import polars as pl

from seahawks_ml.config import BACKTEST_SEASONS, FEATURES_PATH, HOLDOUT_SEASONS, TEAM

RATING_GRID = [(0.5, 4.0, 2.0), (0.6, 4.0, 2.0), (0.7, 6.0, 3.0), (0.6, 6.0, 2.0)]
QB_PRIOR_GRID = [150.0, 250.0, 400.0]
FORECAST_HORIZON = timedelta(days=16)


def _now(args) -> datetime:
    return datetime.fromisoformat(args.now) if getattr(args, "now", None) else datetime.now(UTC)


def _set_output(key: str, value: str) -> None:
    """Expose a value to later GitHub Actions steps (no-op locally)."""
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            f.write(f"{key}={value}\n")


def _load(now: datetime):
    from seahawks_ml.data import load_raw
    from seahawks_ml.stadiums import load_stadiums

    stadiums = load_stadiums()
    return stadiums, load_raw(stadiums, now)


def _build(raw, stadiums, config) -> pl.DataFrame:
    from seahawks_ml.features.build import build_features

    return build_features(raw, stadiums, config.rating, config.qb)


def cmd_features(args) -> None:
    from seahawks_ml.models.store import ProjectConfig, load_config

    stadiums, raw = _load(_now(args))
    try:
        config = load_config()
    except FileNotFoundError:
        config = ProjectConfig()
    frame = _build(raw, stadiums, config)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(FEATURES_PATH)
    print(f"wrote {frame.height} games to {FEATURES_PATH}")


def cmd_backtest(args) -> None:
    """Local only: tune feature params + model config by walk-forward, lock models/config.json."""
    from seahawks_ml.features.qb import QBParams
    from seahawks_ml.features.ratings import RatingParams
    from seahawks_ml.models.backtest import score, tune, walk_forward, walk_forward_log_loss, write_json
    from seahawks_ml.models.pipeline import ModelConfig
    from seahawks_ml.models.store import BACKTEST_PATH, ProjectConfig, save_config

    stadiums, raw = _load(_now(args))
    seasons = list(BACKTEST_SEASONS)
    config = ProjectConfig()
    if args.tune_features:
        print("stage 0a: team-rating shrinkage")
        best = None
        for reg, k, k_new in RATING_GRID:
            cand = replace(config, rating=RatingParams(reg, k, k_new))
            ll = walk_forward_log_loss(_build(raw, stadiums, cand), ModelConfig(), seasons)
            print(f"  {ll:.5f}  {cand.rating}")
            if best is None or ll < best[1]:
                best = (cand, ll)
        config = best[0]
        print("stage 0b: QB prior strength")
        for prior in QB_PRIOR_GRID:
            cand = replace(config, qb=QBParams(prior_dropbacks=prior))
            ll = walk_forward_log_loss(_build(raw, stadiums, cand), ModelConfig(), seasons)
            print(f"  {ll:.5f}  {cand.qb}")
            if ll < best[1]:
                best = (cand, ll)
        config = best[0]
    frame = _build(raw, stadiums, config)
    model_config, trials = tune(frame, seasons)
    config = replace(config, model=model_config)
    save_config(config)
    preds = walk_forward(frame, model_config, seasons)
    write_json(BACKTEST_PATH, {"seasons": seasons, "config": config.to_dict(),
                               "score": score(preds), "trials": trials})
    print(f"locked config {config.fingerprint()} -> models/config.json; report -> {BACKTEST_PATH}")


def cmd_holdout(args) -> None:
    """Local only, run once after tuning: score the locked config on 2024-2025."""
    from seahawks_ml.models.backtest import score, walk_forward, write_json
    from seahawks_ml.models.store import HOLDOUT_PATH, load_config

    if HOLDOUT_PATH.exists() and not args.force:
        raise SystemExit(f"{HOLDOUT_PATH} exists - the holdout is evaluated once. Use --force to overwrite.")
    config = load_config()
    stadiums, raw = _load(_now(args))
    frame = _build(raw, stadiums, config)
    preds = walk_forward(frame, config.model, list(HOLDOUT_SEASONS))
    write_json(HOLDOUT_PATH, {"seasons": list(HOLDOUT_SEASONS), "config": config.to_dict(), "score": score(preds)})
    print(f"holdout written to {HOLDOUT_PATH}")


def cmd_retrain(args) -> None:
    """CI: refit final weights with the locked config. No tuning."""
    from seahawks_ml.models.backtest import write_json
    from seahawks_ml.models.store import METRICS_PATH, load_config, save_model
    from seahawks_ml.models.train import train_production

    now = _now(args)
    config = load_config()
    stadiums, raw = _load(now)
    frame = _build(raw, stadiums, config)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(FEATURES_PATH)
    model, meta = train_production(frame, config, now)
    save_model(model)
    write_json(METRICS_PATH, meta)
    print(f"trained {meta['model_version']} on {meta['n_games']} games")


def cmd_predict(args) -> None:
    """CI: if a run is due for the next Seahawks game, predict and log it."""
    from seahawks_ml.features.base import prepare_games
    from seahawks_ml.ingest import nflverse
    from seahawks_ml.ingest.weather import forecast_for_games
    from seahawks_ml.models.store import METRICS_PATH, load_config, load_model
    from seahawks_ml.pipeline.gate import due_run, next_game, previous_kickoff
    from seahawks_ml.pipeline.history import append_record, read_history, runs_done
    from seahawks_ml.pipeline.predict import latest_injury_week, make_prediction
    from seahawks_ml.pipeline.results import new_results
    from seahawks_ml.stadiums import load_stadiums

    now = _now(args)
    stadiums = load_stadiums()
    history = read_history()
    games = prepare_games(nflverse.load_schedules(), stadiums)

    results = new_results(history, games, now)
    for rec in results:
        append_record(rec)
        print(f"result recorded: {rec['game_id']} {rec['margin_seahawks']:+d}")
    history = read_history()
    _set_output("changed", "true" if results else "false")

    game = next_game(games, TEAM, now)
    if game is None:
        print("no upcoming Seahawks game")
        return
    run_type = args.run_type or due_run(now, game["kickoff_utc"], runs_done(history, game["game_id"]),
                                        previous_kickoff(games, TEAM, now))
    if run_type is None:
        print(f"no run due for {game['game_id']} (kickoff {game['kickoff_utc']:%Y-%m-%d %H:%M} UTC)")
        return

    print(f"{run_type} run for {game['game_id']}")
    config = load_config()
    stadiums, raw = _load(now)
    target = raw.games.filter(pl.col("game_id") == game["game_id"])
    if game["kickoff_utc"] - now <= FORECAST_HORIZON:
        forecast = forecast_for_games(target, stadiums)
        raw = replace(raw, weather=pl.concat([raw.weather, forecast]).unique(["stadium_id", "time_utc"], keep="last"))
    frame = _build(raw, stadiums, config)
    row = frame.filter(pl.col("game_id") == game["game_id"])
    model = load_model()
    version = json.loads(METRICS_PATH.read_text())["model_version"]
    record = make_prediction(row, model, run_type, now, version,
                             latest_injury_week(raw.injuries, game["season"]))
    append_record(record)
    print(f"SEA win probability {record['p_seahawks']:.1%}, margin {record['margin_seahawks']:+.1f}")
    _set_output("changed", "true")


def cmd_build_site(args) -> None:
    from seahawks_ml.site.build import build_site

    print(f"site written to {build_site(_now(args))}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="seahawks_ml")
    parser.add_argument("--now", help="override current time (ISO 8601, for testing)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("features", help="refresh data and rebuild the feature table").set_defaults(func=cmd_features)
    bt = sub.add_parser("backtest", help="LOCAL: tune by walk-forward and lock models/config.json")
    bt.add_argument("--tune-features", action="store_true", help="also tune rating/QB shrinkage (slow)")
    bt.set_defaults(func=cmd_backtest)
    ho = sub.add_parser("holdout", help="LOCAL: evaluate locked config on 2024-2025 (once)")
    ho.add_argument("--force", action="store_true")
    ho.set_defaults(func=cmd_holdout)
    sub.add_parser("retrain", help="CI: refit with locked config").set_defaults(func=cmd_retrain)
    pr = sub.add_parser("predict", help="CI: predict the next Seahawks game if a run is due")
    pr.add_argument("--run-type", choices=["midweek", "final_injury", "gameday"], help="force a run type")
    pr.set_defaults(func=cmd_predict)
    sub.add_parser("build-site", help="render public/").set_defaults(func=cmd_build_site)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Try a real prediction locally**

```bash
uv run python -m seahawks_ml.cli predict --run-type midweek
uv run python -m seahawks_ml.cli build-site
open public/index.html
```
Expected: `midweek run for 2026_..._SEA...`, then `SEA win probability ..%`. The page shows the next game, one trajectory point, the factors table, and the model report. If you want to keep this manual run out of the real track record, remove the last line of `predictions/history.jsonl` before committing.

- [ ] **Step 6: Commit**

```bash
git add seahawks_ml/cli.py tests/test_smoke.py
git commit -m "feat: predict and build-site commands; end-to-end smoke test"
```

---

### Task 6: GitHub Actions workflows

**Files:**
- Create/replace: `.github/workflows/ci.yml`
- Create/replace: `.github/workflows/predict.yml`
- Create/replace: `.github/workflows/retrain.yml`

`ci.yml` runs the tests. `predict.yml` runs hourly from September through February and deploys Pages only when the history changed. `retrain.yml` runs on Tuesdays. All writers share one concurrency group so pushes never race.

- [ ] **Step 1: Write the files**

`.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
    paths-ignore: ["predictions/**", "data/cache/**", "data/features/**", "models/**"]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uv sync --frozen
      - run: uv run pytest
```

`.github/workflows/predict.yml`:

```yaml
name: Predict

on:
  schedule:
    # Hourly during the season (Sep-Feb). The gate exits in seconds unless a run is due.
    - cron: "17 * * 1,2,9,10,11,12 *"
  workflow_dispatch:
    inputs:
      run_type:
        description: "Force a run type (empty = let the gate decide)"
        type: choice
        options: ["", "midweek", "final_injury", "gameday"]
        default: ""

permissions:
  contents: write
  pages: write
  id-token: write

concurrency:
  group: repo-writes
  cancel-in-progress: false

jobs:
  predict:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    outputs:
      changed: ${{ steps.predict.outputs.changed }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uv sync --frozen
      - id: predict
        env:
          RUN_TYPE: ${{ inputs.run_type }}
        run: uv run python -m seahawks_ml.cli predict ${RUN_TYPE:+--run-type "$RUN_TYPE"}
      - name: Commit prediction log and data cache
        if: steps.predict.outputs.changed == 'true'
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add predictions/ data/cache/
          git diff --cached --quiet || git commit -m "predict: $(date -u +%Y-%m-%dT%H:%MZ)"
          git pull --rebase
          git push
      - name: Build site
        if: steps.predict.outputs.changed == 'true'
        run: uv run python -m seahawks_ml.cli build-site
      - uses: actions/upload-pages-artifact@v3
        if: steps.predict.outputs.changed == 'true'
        with:
          path: public

  deploy:
    needs: predict
    if: needs.predict.outputs.changed == 'true'
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

`.github/workflows/retrain.yml`:

```yaml
name: Retrain

on:
  schedule:
    # Tuesdays 14:00 UTC during the season: refit weights with the locked config.
    - cron: "0 14 * 1,2,9,10,11,12 2"
  workflow_dispatch:

permissions:
  contents: write

concurrency:
  group: repo-writes
  cancel-in-progress: false

jobs:
  retrain:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uv sync --frozen
      - run: uv run python -m seahawks_ml.cli retrain
      - name: Commit model, metrics and data cache
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add models/model.pkl models/metrics.json data/cache/ data/features/
          git diff --cached --quiet || git commit -m "retrain: $(date -u +%Y-%m-%d)"
          git pull --rebase
          git push
```

- [ ] **Step 2: Validate YAML syntax**

```bash
uv run --with pyyaml python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows
git commit -m "ci: test, hourly predict + Pages deploy, weekly retrain workflows"
```

---

### Task 7: README and publishing to GitHub

**Files:**
- Create/replace: `README.md`

- [ ] **Step 1: Write the file**

`README.md`:

````markdown
# Seahawks ML

Predicts Seattle Seahawks games (win probability and margin) from free public data —
nflverse play-by-play, injuries, snap counts, rosters, and Open-Meteo weather — and
publishes a dashboard to GitHub Pages via scheduled GitHub Actions.

Design: `docs/superpowers/specs/2026-10-06-seahawks-win-model-design.md`

## Setup

```bash
brew install uv
uv sync
uv run pytest
```

## Local workflow (offseason or whenever you change features/models)

```bash
uv run python -m seahawks_ml.cli features                  # download + cache data, build features
uv run python -m seahawks_ml.cli backtest --tune-features  # tune, lock models/config.json (~15 min)
uv run python -m seahawks_ml.cli holdout                   # score 2024-2025 ONCE
uv run python -m seahawks_ml.cli retrain                   # fit production model
```

The first `features` run fetches ~17 seasons of game-time weather from Open-Meteo in
small paced requests. It can take an hour or more and may pause on rate limits; it saves
after every stadium, so re-running resumes where it stopped. Commit `data/cache/`
afterwards so CI never has to repeat it.

## In season (GitHub Actions)

- `predict.yml` — hourly; predicts ~4 days, ~1 day, and ~3 hours before each kickoff,
  records results, and redeploys the dashboard.
- `retrain.yml` — Tuesdays; refits weights with the locked config (no tuning in CI).
- `ci.yml` — tests on every push.

Manual run: `uv run python -m seahawks_ml.cli predict --run-type midweek`

Data: [nflverse](https://github.com/nflverse), weather by [Open-Meteo](https://open-meteo.com/).
An analytical model, not betting advice.
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README"
```

- [ ] **Step 3: Publish (ask the user first: this creates a remote repository)**

```bash
gh repo create SeaHawksML --private --source . --push
```

Then in the repository settings, go to **Pages → Build and deployment → Source** and choose **GitHub Actions**. The Pages site of a private repository needs a paid GitHub plan; on a free plan, make the repository public or leave Pages off and use the `public/` artifact.

- [ ] **Step 4: Trigger the first runs**

```bash
gh workflow run ci.yml
gh workflow run predict.yml -f run_type=midweek
gh run watch
```
Expected: CI passes. The predict run commits `predictions/history.jsonl`, and the deploy job prints the Pages URL.

---
