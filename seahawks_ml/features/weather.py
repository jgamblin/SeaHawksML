"""Game-window weather features (mean temp/wind, total precipitation over 4 hours)."""

from collections import defaultdict
from datetime import timedelta

import polars as pl

from seahawks_ml.ingest.weather import GAME_WINDOW_HOURS

INDOOR_ROOFS = {"dome", "closed"}
INDOOR_VALUES = {"temp_f": 70.0, "wind_mph": 0.0, "precip_in": 0.0}


def compute_weather(games: pl.DataFrame, weather: pl.DataFrame) -> pl.DataFrame:
    """weather_source is one of indoor / archive / forecast / climatology."""
    by_hour = {(r["stadium_id"], r["time_utc"]): r for r in weather.iter_rows(named=True)}
    clim: dict[tuple[str, int], list[list[float]]] = defaultdict(lambda: [[], [], []])
    league_clim: dict[int, list[list[float]]] = defaultdict(lambda: [[], [], []])
    for r in weather.filter(pl.col("source") == "archive").iter_rows(named=True):
        for i, c in enumerate(("temp_f", "wind_mph", "precip_in")):
            if r[c] is not None:
                clim[(r["stadium_id"], r["time_utc"].month)][i].append(r[c])
                league_clim[r["time_utc"].month][i].append(r[c])

    def climatology(stadium_id: str, month: int) -> dict:
        vals = clim.get((stadium_id, month)) or league_clim.get(month)
        if not vals or not vals[0]:
            return dict(INDOOR_VALUES)
        temp, wind, precip = (sum(v) / len(v) if v else 0.0 for v in vals)
        return {"temp_f": temp, "wind_mph": wind, "precip_in": precip * GAME_WINDOW_HOURS}

    rows = []
    for g in games.iter_rows(named=True):
        base = {"game_id": g["game_id"]}
        if g["roof"] in INDOOR_ROOFS:
            rows.append(base | INDOOR_VALUES | {"is_indoor": 1, "weather_source": "indoor"})
            continue
        start = g["kickoff_utc"].replace(minute=0, second=0, microsecond=0)
        hours = [by_hour.get((g["stadium_id"], start + timedelta(hours=h))) for h in range(GAME_WINDOW_HOURS)]
        hours = [h for h in hours if h and h["temp_f"] is not None]
        if hours:
            vals = {
                "temp_f": sum(h["temp_f"] for h in hours) / len(hours),
                "wind_mph": sum(h["wind_mph"] or 0.0 for h in hours) / len(hours),
                "precip_in": sum(h["precip_in"] or 0.0 for h in hours) * GAME_WINDOW_HOURS / len(hours),
            }
            source = hours[0]["source"]
        else:
            vals, source = climatology(g["stadium_id"], g["kickoff_utc"].month), "climatology"
        rows.append(base | vals | {"is_indoor": 0, "weather_source": source})
    return pl.DataFrame(rows, schema={"game_id": pl.Utf8, "temp_f": pl.Float64, "wind_mph": pl.Float64,
                                      "precip_in": pl.Float64, "is_indoor": pl.Int64,
                                      "weather_source": pl.Utf8})
