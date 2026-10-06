import polars as pl

from tests.synthetic import make_raw


def test_as_of_hides_results_and_stats_from_cutoff_onward():
    raw = make_raw()
    target = raw.games.filter(pl.col("season") == 2013).row(4, named=True)
    cut = raw.as_of(target["kickoff_utc"])
    g = cut.games.filter(pl.col("game_id") == target["game_id"]).row(0, named=True)
    assert g["margin"] is None and g["home_score"] is None
    assert target["game_id"] not in cut.team_epa["game_id"].to_list()
    assert target["game_id"] not in cut.snaps["game_id"].to_list()
    assert cut.games.filter(pl.col("kickoff_utc") < target["kickoff_utc"])["margin"].null_count() == 0
    # the target game's own week of injury reports survives; later weeks do not
    assert cut.injuries.filter((pl.col("season") == 2013) & (pl.col("week") > target["week"])).height == 0
    assert cut.games.height == raw.games.height
