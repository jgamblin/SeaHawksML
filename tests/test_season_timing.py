import polars as pl

from seahawks_ml.features.season_timing import compute_season_timing


def test_final_regular_week_tracks_17_and_18_game_eras():
    games = pl.DataFrame({
        "game_id": ["a", "b", "c", "d", "e"],
        "season": [2020, 2020, 2020, 2021, 2021],
        "week": [16, 17, 18, 17, 18],
        "game_type": ["REG", "REG", "WC", "REG", "REG"],
    })
    out = {r["game_id"]: r for r in compute_season_timing(games).iter_rows(named=True)}
    assert out["b"]["is_final_regular_week"] == 1 and out["a"]["is_final_regular_week"] == 0
    assert out["c"]["is_final_regular_week"] == 0 and out["c"]["is_playoff"] == 1
    assert out["e"]["is_final_regular_week"] == 1 and out["d"]["is_final_regular_week"] == 0
    assert out["e"]["week_number"] == 18
