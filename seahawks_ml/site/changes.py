"""Plain-language notes on what changed between two prediction runs for the same game."""

from seahawks_ml.site.labels import FEATURE_LABELS

# Smallest moves worth mentioning.
FACTOR_MIN_PTS = 0.3
VEGAS_MIN_PCT = 1.0
TEMP_MIN_F = 5.0
WIND_MIN_MPH = 5.0
PRECIP_MIN_IN = 0.05
MAX_FACTOR_NOTES = 3
NO_CHANGE = "No meaningful input changes since the last run."


def _label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def _pts(x: float) -> str:
    return f"{x:+.1f}".replace("-", "−")


def _signed(x: float) -> str:
    return f"{x:.1f}".replace("-", "−")


def _injury_notes(prev: dict, cur: dict) -> list[str]:
    if cur["is_final_injury_report"] and not prev["is_final_injury_report"]:
        return ["Final injury report for this week now included"]
    week = cur["latest_injury_week"]
    if week is not None and week != prev["latest_injury_week"]:
        return [f"Week {week} injury report now included"]
    return []


def _weather_notes(prev: dict, cur: dict) -> list[str]:
    a, b = prev["weather"], cur["weather"]
    notes = []
    if a["source"] == "climatology" and b["source"] == "forecast":
        notes.append("Weather now from the forecast (was a monthly average)")
    if abs(b["temp_f"] - a["temp_f"]) >= TEMP_MIN_F:
        notes.append(f"Temperature {a['temp_f']:.0f} → {b['temp_f']:.0f}°F")
    if abs(b["wind_mph"] - a["wind_mph"]) >= WIND_MIN_MPH:
        notes.append(f"Wind {a['wind_mph']:.0f} → {b['wind_mph']:.0f} mph")
    if abs(b["precip_in"] - a["precip_in"]) >= PRECIP_MIN_IN:
        notes.append(f"Precipitation {a['precip_in']:.2f} → {b['precip_in']:.2f} in")
    return notes


def _vegas_notes(prev: dict, cur: dict) -> list[str]:
    a, b = prev["p_vegas_seahawks"], cur["p_vegas_seahawks"]
    if a is None or b is None or abs(b - a) * 100 < VEGAS_MIN_PCT:
        return []
    return [f"Vegas benchmark {a * 100:.0f}% → {b * 100:.0f}%"]


def _factor_notes(prev: dict, cur: dict) -> list[str]:
    """Moves in the top factors, largest first. Only the top factors are logged per run, so a
    factor that enters or leaves the list is reported as such rather than with a delta."""
    a = {f["feature"]: f["points"] for f in prev["top_factors"]}
    b = {f["feature"]: f["points"] for f in cur["top_factors"]}
    moves = []  # (size, note)
    for feat in a.keys() & b.keys():
        if abs(b[feat] - a[feat]) >= FACTOR_MIN_PTS:
            moves.append((abs(b[feat] - a[feat]), f"{_label(feat)} {_signed(a[feat])} → {_signed(b[feat])} pts"))
    for feat in b.keys() - a.keys():
        moves.append((abs(b[feat]), f"{_label(feat)} is now a top factor ({_pts(b[feat])} pts)"))
    for feat in a.keys() - b.keys():
        moves.append((abs(a[feat]), f"{_label(feat)} dropped out of the top factors"))
    moves.sort(key=lambda m: -m[0])
    return [note for _, note in moves[:MAX_FACTOR_NOTES]]


def run_changes(prev: dict, cur: dict) -> dict:
    """Compare two prediction records (Seahawks perspective) for the same game."""
    notes = (_injury_notes(prev, cur) + _factor_notes(prev, cur) + _weather_notes(prev, cur)
             + _vegas_notes(prev, cur))
    if cur["model_version"] != prev["model_version"]:
        notes.append(f"Model retrained ({cur['model_version']})")
    return {
        "from_run": prev["run_type"],
        "to_run": cur["run_type"],
        "p_delta": round((cur["p_seahawks"] - prev["p_seahawks"]) * 100, 1),
        "margin_delta": round(cur["margin_seahawks"] - prev["margin_seahawks"], 1),
        "notes": notes or [NO_CHANGE],
    }


def trajectory_changes(runs: list[dict]) -> list[dict]:
    """Change notes for each consecutive pair of runs (runs in time order)."""
    return [run_changes(a, b) for a, b in zip(runs, runs[1:])]
