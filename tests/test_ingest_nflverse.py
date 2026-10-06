import polars as pl

from seahawks_ml.ingest.nflverse import aggregate_qb_games, aggregate_team_epa


def _pbp():
    return pl.DataFrame({
        "game_id": ["g1"] * 6,
        "season": [2015] * 6,
        "posteam": ["STL", "STL", "STL", "SEA", "SEA", None],
        "defteam": ["SEA", "SEA", "SEA", "STL", "STL", None],
        "play_type": ["pass", "run", "punt", "pass", "pass", None],
        "epa": [0.5, -0.1, 2.0, 1.0, None, 0.3],
        "qb_dropback": [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        "passer_player_id": ["QB-STL", None, None, "QB-SEA", "QB-SEA", None],
        "qb_epa": [0.5, -0.1, 2.0, 1.0, None, 0.3],
        "cpoe": [10.0, None, None, None, -5.0, None],
    })


def test_aggregate_team_epa_uses_scrimmage_plays_and_normalizes_teams():
    out = aggregate_team_epa(_pbp())
    rows = {r["team"]: r for r in out.iter_rows(named=True)}
    assert set(rows) == {"LA", "SEA"}  # STL -> LA, null posteam dropped
    assert rows["LA"]["plays"] == 2 and abs(rows["LA"]["epa_sum"] - 0.4) < 1e-9  # punt excluded
    assert rows["LA"]["opponent"] == "SEA"
    assert rows["SEA"]["plays"] == 1  # null epa excluded


def test_aggregate_qb_games_sums_dropbacks():
    out = aggregate_qb_games(_pbp())
    rows = {r["qb_id"]: r for r in out.iter_rows(named=True)}
    assert rows["QB-STL"]["dropbacks"] == 1 and rows["QB-STL"]["team"] == "LA"
    assert rows["QB-STL"]["cpoe_n"] == 1 and rows["QB-STL"]["cpoe_sum"] == 10.0
    assert rows["QB-SEA"]["dropbacks"] == 1  # second SEA dropback has null qb_epa
