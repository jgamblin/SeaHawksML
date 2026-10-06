"""Open-Meteo weather for game hours (archive for history, forecast for upcoming games).

Only the hours around each outdoor kickoff are kept, so the cache stays small
enough to commit (data/cache/weather_games.parquet).
"""

import time
from datetime import UTC, date, datetime, timedelta

import httpx
import polars as pl

from seahawks_ml.config import CACHE_DIR, FIRST_TRAIN_SEASON
from seahawks_ml.stadiums import Stadium

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_LAG_DAYS = 6  # archive data trails real time by up to ~5 days
WINDOW_DAYS = 14  # Open-Meteo counts requests spanning > 2 weeks as multiple calls
RATE_LIMIT_SLEEP = 65  # seconds to wait after HTTP 429 (per-minute limit)
GAME_WINDOW_HOURS = 4  # kickoff hour plus the next three
BASE_PARAMS = {
    "hourly": "temperature_2m,wind_speed_10m,precipitation",
    "timezone": "UTC",
    "temperature_unit": "fahrenheit",
    "wind_speed_unit": "mph",
    "precipitation_unit": "inch",
}
WEATHER_SCHEMA = {
    "stadium_id": pl.Utf8,
    "time_utc": pl.Datetime("us", "UTC"),
    "temp_f": pl.Float64,
    "wind_mph": pl.Float64,
    "precip_in": pl.Float64,
    "source": pl.Utf8,  # "archive" or "forecast"
}


def parse_hourly(payload: dict, stadium_id: str, source: str) -> pl.DataFrame:
    h = payload["hourly"]
    n = len(h["time"])
    return pl.DataFrame(
        {
            "stadium_id": [stadium_id] * n,
            "time_utc": [datetime.fromisoformat(t).replace(tzinfo=UTC) for t in h["time"]],
            "temp_f": h["temperature_2m"],
            "wind_mph": h["wind_speed_10m"],
            "precip_in": h["precipitation"],
            "source": [source] * n,
        },
        schema=WEATHER_SCHEMA,
    )


def _get_json(client: httpx.Client, url: str, params: dict, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            resp = client.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            if attempt == retries - 1:
                raise
            limited = isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429
            time.sleep(RATE_LIMIT_SLEEP if limited else 2 ** (attempt + 1))
    raise AssertionError("unreachable")


def date_windows(dates: list[date], max_days: int = WINDOW_DAYS) -> list[tuple[date, date]]:
    """Group sorted dates into [start, end] windows no longer than max_days."""
    windows: list[tuple[date, date]] = []
    for d in sorted(set(dates)):
        if windows and (d - windows[-1][0]).days < max_days:
            windows[-1] = (windows[-1][0], d)
        else:
            windows.append((d, d))
    return windows


def fetch_archive(client: httpx.Client, stadium: Stadium, start: date, end: date) -> pl.DataFrame:
    params = {**BASE_PARAMS, "latitude": stadium.lat, "longitude": stadium.lon,
              "start_date": start.isoformat(), "end_date": end.isoformat()}
    return parse_hourly(_get_json(client, ARCHIVE_URL, params), stadium.stadium_id, "archive")


def fetch_forecast(client: httpx.Client, stadium: Stadium) -> pl.DataFrame:
    params = {**BASE_PARAMS, "latitude": stadium.lat, "longitude": stadium.lon, "forecast_days": 16}
    return parse_hourly(_get_json(client, FORECAST_URL, params), stadium.stadium_id, "forecast")


def game_hours(games: pl.DataFrame) -> pl.DataFrame:
    """(stadium_id, time_utc) for each hour in each outdoor game's window."""
    rows = []
    for g in games.filter(~pl.col("roof").is_in(["dome", "closed"])).iter_rows(named=True):
        start = g["kickoff_utc"].replace(minute=0, second=0, microsecond=0)
        for h in range(GAME_WINDOW_HOURS):
            rows.append({"stadium_id": g["stadium_id"], "time_utc": start + timedelta(hours=h)})
    schema = {"stadium_id": pl.Utf8, "time_utc": pl.Datetime("us", "UTC")}
    return pl.DataFrame(rows, schema=schema).unique().sort("stadium_id", "time_utc")


def update_archive_cache(
    games: pl.DataFrame,
    stadiums: dict[str, Stadium],
    now: datetime,
    client: httpx.Client | None = None,
    cache_path=CACHE_DIR / "weather_games.parquet",
    pause: float = 1.0,
) -> pl.DataFrame:
    """Fetch archive weather for outdoor game hours (2009+) not yet cached; return the cache.

    Requests are two-week windows around game dates, paced by `pause` seconds, and the
    cache is written after each stadium so an interrupted first run can resume.
    """
    cached = pl.read_parquet(cache_path) if cache_path.exists() else pl.DataFrame(schema=WEATHER_SCHEMA)
    cutoff = now - timedelta(days=ARCHIVE_LAG_DAYS)
    eligible = games.filter((pl.col("kickoff_utc") < cutoff) & (pl.col("season") >= FIRST_TRAIN_SEASON))
    needed = game_hours(eligible).join(
        cached.select("stadium_id", "time_utc"), on=["stadium_id", "time_utc"], how="anti"
    )
    if needed.height == 0:
        return cached
    own_client = client is None
    client = client or httpx.Client()
    try:
        for (stadium_id,), hours in needed.sort("stadium_id").group_by(["stadium_id"], maintain_order=True):
            hours_set = set(hours["time_utc"].to_list())
            windows = date_windows([t.date() for t in hours_set])
            print(f"weather archive: {stadium_id} ({len(windows)} requests)")
            parts = []
            for first, last in windows:
                parts.append(fetch_archive(client, stadiums[stadium_id], first, last)
                             .filter(pl.col("time_utc").is_in(list(hours_set))))
                time.sleep(pause)
            cached = pl.concat([cached, *parts]).unique(["stadium_id", "time_utc"]).sort("stadium_id", "time_utc")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cached.write_parquet(cache_path)
    finally:
        if own_client:
            client.close()
    return cached


def forecast_for_games(
    games: pl.DataFrame, stadiums: dict[str, Stadium], client: httpx.Client | None = None
) -> pl.DataFrame:
    """Forecast rows for the game windows of the given (upcoming) games."""
    hours = game_hours(games)
    if hours.height == 0:
        return pl.DataFrame(schema=WEATHER_SCHEMA)
    own_client = client is None
    client = client or httpx.Client()
    try:
        parts = [
            fetch_forecast(client, stadiums[sid]).join(h, on=["stadium_id", "time_utc"], how="inner")
            for (sid,), h in hours.group_by(["stadium_id"])
        ]
    finally:
        if own_client:
            client.close()
    return pl.concat(parts).cast(WEATHER_SCHEMA) if parts else pl.DataFrame(schema=WEATHER_SCHEMA)
