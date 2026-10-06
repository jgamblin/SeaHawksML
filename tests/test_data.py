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
    assert target["game_id"] in raw.player_stats["game_id"].to_list()
    assert target["game_id"] not in cut.player_stats["game_id"].to_list()
    assert cut.player_stats.height > 0
    assert cut.games.filter(pl.col("kickoff_utc") < target["kickoff_utc"])["margin"].null_count() == 0
    # the target game's own week of injury reports survives; later weeks do not
    assert cut.injuries.filter((pl.col("season") == 2013) & (pl.col("week") > target["week"])).height == 0
    assert cut.games.height == raw.games.height


def test_load_raw_reuses_supplied_games(monkeypatch):
    from seahawks_ml.data import load_raw
    from seahawks_ml.ingest import nflverse, weather
    from seahawks_ml.stadiums import load_stadiums

    raw = make_raw()

    def no_schedule_download():
        raise AssertionError("schedule must not be downloaded when games is supplied")

    monkeypatch.setattr(nflverse, "load_schedules", no_schedule_download)
    monkeypatch.setattr(nflverse, "load_season_tables", lambda *a, **k: {
        "team_epa": raw.team_epa, "qb_games": raw.qb_games, "injuries": raw.injuries, "snaps": raw.snaps,
        "player_stats": raw.player_stats})
    monkeypatch.setattr(nflverse, "load_players", lambda: raw.players)
    monkeypatch.setattr(weather, "update_archive_cache", lambda games, stadiums, now: raw.weather)
    loaded = load_raw(load_stadiums(), raw.games["kickoff_utc"].max(), raw.games)
    assert loaded.games.equals(raw.games)
    assert loaded.player_stats.equals(raw.player_stats)


def test_synthetic_player_stats_match_snap_players():
    raw = make_raw()
    ps = raw.player_stats
    assert ps["season"].min() == 2012
    assert set(ps["position_group"].unique().to_list()) == {"WR", "RB"}
    assert ps.filter(pl.col("position_group") == "RB")["carries"].min() > 0
    assert ps.filter(pl.col("position_group") == "WR")["targets"].min() > 0
    # every stat row belongs to a played game and a team that played in it
    played = raw.team_epa.select("game_id", "team")
    assert ps.join(played, on=["game_id", "team"], how="anti").height == 0


def test_current_season_looks_a_week_ahead():
    from datetime import UTC, datetime

    import polars as pl

    from seahawks_ml.data import current_season

    games = pl.DataFrame({"season": [2025, 2026],
                          "kickoff_utc": [datetime(2026, 2, 8, tzinfo=UTC), datetime(2026, 9, 10, tzinfo=UTC)]})
    assert current_season(games, datetime(2026, 9, 5, tzinfo=UTC)) == 2026
    assert current_season(games, datetime(2026, 8, 30, tzinfo=UTC)) == 2025
