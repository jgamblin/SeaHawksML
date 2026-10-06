from datetime import UTC, datetime

import httpx
import polars as pl
import pytest

from seahawks_ml.ingest.weather import (
    date_windows,
    fetch_archive,
    forecast_for_games,
    game_hours,
    parse_hourly,
    update_archive_cache,
)
from seahawks_ml.stadiums import load_stadiums


def _games(rows):
    return pl.DataFrame(rows, schema={
        "game_id": pl.Utf8, "stadium_id": pl.Utf8, "roof": pl.Utf8,
        "kickoff_utc": pl.Datetime("us", "UTC"),
    }).with_columns(pl.col("kickoff_utc").dt.year().cast(pl.Int64).alias("season"))


def _payload(times):
    return {"hourly": {
        "time": times,
        "temperature_2m": [50.0 + i for i in range(len(times))],
        "wind_speed_10m": [10.0] * len(times),
        "precipitation": [0.1] * len(times),
    }}


def test_parse_hourly_builds_utc_rows():
    df = parse_hourly(_payload(["2024-09-08T20:00", "2024-09-08T21:00"]), "SEA00", "archive")
    assert df["time_utc"][0] == datetime(2024, 9, 8, 20, tzinfo=UTC)
    assert df["temp_f"].to_list() == [50.0, 51.0]
    assert df["source"].unique().to_list() == ["archive"]


def test_game_hours_skips_indoor_games_and_covers_window():
    games = _games([
        {"game_id": "a", "stadium_id": "SEA00", "roof": "outdoors",
         "kickoff_utc": datetime(2024, 9, 8, 20, 5, tzinfo=UTC)},
        {"game_id": "b", "stadium_id": "LAX01", "roof": "dome",
         "kickoff_utc": datetime(2024, 9, 8, 20, 5, tzinfo=UTC)},
    ])
    hours = game_hours(games)
    assert hours["stadium_id"].unique().to_list() == ["SEA00"]
    assert hours["time_utc"].to_list() == [datetime(2024, 9, 8, h, tzinfo=UTC) for h in (20, 21, 22, 23)]


def test_update_archive_cache_fetches_only_missing_hours(tmp_path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        times = [f"2024-09-08T{h:02d}:00" for h in range(24)]
        return httpx.Response(200, json=_payload(times))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    games = _games([{"game_id": "a", "stadium_id": "SEA00", "roof": "outdoors",
                     "kickoff_utc": datetime(2024, 9, 8, 20, 5, tzinfo=UTC)}])
    path = tmp_path / "w.parquet"
    now = datetime(2024, 10, 1, tzinfo=UTC)
    out = update_archive_cache(games, load_stadiums(), now, client=client, cache_path=path, pause=0)
    assert out.height == 4 and len(calls) == 1
    assert calls[0]["start_date"] == "2024-09-08"
    out2 = update_archive_cache(games, load_stadiums(), now, client=client, cache_path=path, pause=0)
    assert out2.height == 4 and len(calls) == 1  # nothing new fetched


def test_update_archive_cache_skips_games_inside_archive_lag(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: pytest.fail("no fetch expected")))
    games = _games([{"game_id": "a", "stadium_id": "SEA00", "roof": "outdoors",
                     "kickoff_utc": datetime(2024, 9, 8, 20, 5, tzinfo=UTC)}])
    out = update_archive_cache(games, load_stadiums(), datetime(2024, 9, 10, tzinfo=UTC),
                               client=client, cache_path=tmp_path / "w.parquet", pause=0)
    assert out.height == 0


def test_date_windows_groups_dates_within_two_weeks():
    from datetime import date
    ds = [date(2024, 9, 8), date(2024, 9, 15), date(2024, 9, 22), date(2024, 10, 20)]
    assert date_windows(ds) == [(date(2024, 9, 8), date(2024, 9, 15)),
                                (date(2024, 9, 22), date(2024, 9, 22)),
                                (date(2024, 10, 20), date(2024, 10, 20))]


def test_update_archive_cache_ignores_seasons_before_training(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: pytest.fail("no fetch expected")))
    games = _games([{"game_id": "a", "stadium_id": "SEA00", "roof": "outdoors",
                     "kickoff_utc": datetime(2005, 9, 8, 20, 5, tzinfo=UTC)}])
    out = update_archive_cache(games, load_stadiums(), datetime(2024, 9, 10, tzinfo=UTC),
                               client=client, cache_path=tmp_path / "w.parquet", pause=0)
    assert out.height == 0


def test_forecast_for_games_keeps_only_game_window():
    def handler(request):
        return httpx.Response(200, json=_payload([f"2024-09-08T{h:02d}:00" for h in range(24)]))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    games = _games([{"game_id": "a", "stadium_id": "SEA00", "roof": "outdoors",
                     "kickoff_utc": datetime(2024, 9, 8, 20, 5, tzinfo=UTC)}])
    out = forecast_for_games(games, load_stadiums(), client=client)
    assert out.height == 4 and out["source"].unique().to_list() == ["forecast"]


@pytest.mark.network
def test_fetch_archive_live():
    with httpx.Client() as client:
        df = fetch_archive(client, load_stadiums()["SEA00"],
                           datetime(2024, 9, 8).date(), datetime(2024, 9, 8).date())
    assert df.height == 24


def test_update_archive_cache_drops_null_temperature_rows(tmp_path):
    def handler(request):
        payload = _payload([f"2024-09-08T{h:02d}:00" for h in range(24)])
        payload["hourly"]["temperature_2m"][21] = None
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    games = _games([{"game_id": "a", "stadium_id": "SEA00", "roof": "outdoors",
                     "kickoff_utc": datetime(2024, 9, 8, 20, 5, tzinfo=UTC)}])
    path = tmp_path / "w.parquet"
    out = update_archive_cache(games, load_stadiums(), datetime(2024, 10, 1, tzinfo=UTC),
                               client=client, cache_path=path, pause=0)
    assert out.height == 3 and out["temp_f"].null_count() == 0
    assert pl.read_parquet(path).height == 3
