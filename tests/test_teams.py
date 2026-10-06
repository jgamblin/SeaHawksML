import polars as pl

from seahawks_ml.teams import normalize_team


def test_normalize_team_maps_relocated_franchises():
    df = pl.DataFrame({"team": ["OAK", "SD", "STL", "SEA"]})
    out = df.select(normalize_team("team"))["team"].to_list()
    assert out == ["LV", "LAC", "LA", "SEA"]
