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
    # Archive observations by (stadium, month) and by month league-wide, time-sorted so a
    # game's climatology only ever sees hours strictly before its own kickoff.
    clim: dict[tuple[str, int], list[tuple]] = defaultdict(list)
    league_clim: dict[int, list[tuple]] = defaultdict(list)
    archive = weather.filter(pl.col("source") == "archive").sort("time_utc")
    for r in archive.iter_rows(named=True):
        obs = (r["time_utc"], r["temp_f"], r["wind_mph"], r["precip_in"])
        clim[(r["stadium_id"], r["time_utc"].month)].append(obs)
        league_clim[r["time_utc"].month].append(obs)

    def _means(obs: list[tuple], before) -> list[list[float]]:
        vals: list[list[float]] = [[], [], []]
        for o in obs:
            if o[0] >= before:
                break
            for i in range(3):
                if o[i + 1] is not None:
                    vals[i].append(o[i + 1])
        return vals

    def climatology(stadium_id: str, kickoff) -> dict:
        month = kickoff.month
        vals = _means(clim.get((stadium_id, month), []), kickoff)
        if not vals[0]:
            vals = _means(league_clim.get(month, []), kickoff)
        if not vals[0]:
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
            vals, source = climatology(g["stadium_id"], g["kickoff_utc"]), "climatology"
        rows.append(base | vals | {"is_indoor": 0, "weather_source": source})
    return pl.DataFrame(rows, schema={"game_id": pl.Utf8, "temp_f": pl.Float64, "wind_mph": pl.Float64,
                                      "precip_in": pl.Float64, "is_indoor": pl.Int64,
                                      "weather_source": pl.Utf8})
