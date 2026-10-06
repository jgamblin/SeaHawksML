import polars as pl
import pytest

from seahawks_ml.features.weather import compute_weather
from tests.synthetic import make_raw


def test_indoor_games_get_neutral_weather():
    raw = make_raw()
    out = compute_weather(raw.games, raw.weather).join(raw.games.select("game_id", "roof"), on="game_id")
    indoor = out.filter(pl.col("roof").is_in(["dome", "closed"]))
    assert indoor.height > 0
    assert indoor["weather_source"].unique().to_list() == ["indoor"]
    assert indoor["temp_f"].unique().to_list() == [70.0]


def test_outdoor_games_average_game_window():
    raw = make_raw()
    g = raw.games.filter(pl.col("roof") == "outdoors").row(0, named=True)
    row = compute_weather(raw.games, raw.weather).filter(pl.col("game_id") == g["game_id"]).row(0, named=True)
    assert row["weather_source"] == "archive"
    assert row["temp_f"] == pytest.approx(60.0 - g["week"])
    assert row["wind_mph"] == pytest.approx(6.5)  # mean of 5,6,7,8


def test_missing_outdoor_weather_falls_back_to_climatology():
    raw = make_raw()
    g = raw.games.filter(pl.col("roof") == "outdoors").row(0, named=True)
    weather = raw.weather.filter(
        ~((pl.col("stadium_id") == g["stadium_id"]) & (pl.col("time_utc") >= g["kickoff_utc"])
          & (pl.col("time_utc") < g["kickoff_utc"] + pl.duration(hours=4)))
    )
    row = compute_weather(raw.games, weather).filter(pl.col("game_id") == g["game_id"]).row(0, named=True)
    assert row["weather_source"] == "climatology"
    assert 50 < row["temp_f"] < 60
