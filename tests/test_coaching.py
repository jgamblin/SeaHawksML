import polars as pl

from seahawks_ml.features.base import team_games
from seahawks_ml.features.coaching import new_head_coach
from tests.synthetic import make_raw


def test_new_head_coach_flags_first_season_after_change_only():
    raw = make_raw(seasons=(2011, 2012, 2013))
    flags = new_head_coach(team_games(raw.games)).join(
        team_games(raw.games).select("game_id", "team", "season"), on=["game_id", "team"]
    )
    by = flags.group_by("team", "season").agg(pl.col("new_head_coach").max()).sort("team", "season")
    lookup = {(r["team"], r["season"]): r["new_head_coach"] for r in by.iter_rows(named=True)}
    assert lookup[("SF", 2011)] == 0  # no previous season in data
    assert lookup[("SF", 2012)] == 1  # Coach A -> Coach B
    assert lookup[("SF", 2013)] == 0
    assert lookup[("SEA", 2012)] == 0
