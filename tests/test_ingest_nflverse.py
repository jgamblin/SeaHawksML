import polars as pl

from seahawks_ml.ingest.nflverse import FUMBLE_PLAY_EPA, aggregate_qb_games, aggregate_team_epa


def _pbp():
    # rows 0-7: scrimmage (and one punt); rows 8-11: special teams. nflverse posteam on a
    # kickoff is the receiving team.
    return pl.DataFrame({
        "game_id": ["g1"] * 12,
        "season": [2015] * 12,
        "posteam": ["STL", "STL", "STL", "SEA", "SEA", None, "SEA", "SEA", "SEA", "SEA", "SEA", None],
        "defteam": ["SEA", "SEA", "SEA", "STL", "STL", None, "STL", "STL", "STL", "STL", "STL", None],
        "play_type": ["pass", "run", "punt", "pass", "pass", None, "run", "pass",
                      "kickoff", "field_goal", "extra_point", "kickoff"],
        "epa": [0.5, -0.1, 2.0, 1.0, None, 0.3, 0.9, 0.7, -0.4, 1.2, None, 5.0],
        "qb_dropback": [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "passer_player_id": ["QB-STL", None, None, "QB-SEA", "QB-SEA", None, None, "QB-SEA", None, None, None, None],
        "passer_id": ["QB-STL", None, None, "QB-SEA", "QB-SEA", None, "QB-SEA", "QB-SEA", None, None, None, None],
        "qb_epa": [0.5, -0.1, 2.0, 1.0, None, 0.3, 0.9, 0.7, -0.4, 1.2, None, 5.0],
        "cpoe": [10.0, None, None, None, -5.0, None, None, None, None, None, None, None],
        "two_point_attempt": [0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        # row 6 is a QB scramble: play_type run, but pass == 1
        "pass": [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "rush": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "success": [1.0, 0.0, 1.0, 1.0, None, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        "fumble": [0.0] * 12,
    })


def test_aggregate_team_epa_uses_scrimmage_plays_and_normalizes_teams():
    out = aggregate_team_epa(_pbp())
    rows = {r["team"]: r for r in out.iter_rows(named=True)}
    assert set(rows) == {"LA", "SEA"}  # STL -> LA, null posteam dropped
    assert rows["LA"]["plays"] == 2 and abs(rows["LA"]["epa_sum"] - 0.4) < 1e-9  # punt excluded
    assert rows["LA"]["opponent"] == "SEA"
    assert rows["SEA"]["plays"] == 2  # null epa and two-point attempt excluded
    assert abs(rows["SEA"]["epa_sum"] - 1.9) < 1e-9


def test_aggregate_team_epa_pass_rush_success_split():
    rows = {r["team"]: r for r in aggregate_team_epa(_pbp()).iter_rows(named=True)}
    la, sea = rows["LA"], rows["SEA"]
    assert la["pass_plays"] == 1 and abs(la["pass_epa_sum"] - 0.5) < 1e-9
    assert la["rush_plays"] == 1 and abs(la["rush_epa_sum"] + 0.1) < 1e-9
    assert la["success_sum"] == 1.0
    # SEA: a dropback (1.0) and a scramble (0.9) are both pass plays
    assert sea["pass_plays"] == 2 and abs(sea["pass_epa_sum"] - 1.9) < 1e-9
    assert sea["rush_plays"] == 0 and sea["rush_epa_sum"] == 0.0
    assert sea["success_sum"] == 2.0


def _fumble_pbp():
    return _pbp().with_columns(pl.Series("fumble", [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 4))


def test_aggregate_team_epa_neutralizes_fumble_luck_with_constant():
    rows = {r["team"]: r for r in aggregate_team_epa(_fumble_pbp()).iter_rows(named=True)}
    m = FUMBLE_PLAY_EPA  # not the frame's own fumble mean (0.4667)
    assert abs(rows["LA"]["epa_sum"] - 2 * m) < 1e-9
    assert abs(rows["LA"]["pass_epa_sum"] - m) < 1e-9
    assert abs(rows["LA"]["rush_epa_sum"] - m) < 1e-9
    assert abs(rows["SEA"]["epa_sum"] - (m + 0.9)) < 1e-9
    assert abs(rows["SEA"]["pass_epa_sum"] - (m + 0.9)) < 1e-9
    assert rows["SEA"]["plays"] == 2
    # success is not neutralized
    assert rows["LA"]["success_sum"] == 1.0


def test_aggregate_team_epa_fumble_value_is_independent_of_frame():
    # dropping the later fumble play must not change the earlier ones' value
    pbp = _fumble_pbp()
    full = {r["team"]: r for r in aggregate_team_epa(pbp).iter_rows(named=True)}
    part = {r["team"]: r for r in aggregate_team_epa(pbp.head(3)).iter_rows(named=True)}
    assert part["LA"]["epa_sum"] == full["LA"]["epa_sum"]


def test_aggregate_team_epa_fumble_epa_override():
    rows = {r["team"]: r for r in aggregate_team_epa(_fumble_pbp(), fumble_epa=-3.0).iter_rows(named=True)}
    assert abs(rows["LA"]["epa_sum"] + 6.0) < 1e-9


def test_aggregate_team_epa_special_teams_sums():
    rows = {r["team"]: r for r in aggregate_team_epa(_pbp()).iter_rows(named=True)}
    # possession team's EPA on punts / kickoffs / field goals / extra points
    assert rows["LA"]["st_plays"] == 1 and abs(rows["LA"]["st_epa_sum"] - 2.0) < 1e-9
    # null-epa extra point and null-posteam kickoff excluded
    assert rows["SEA"]["st_plays"] == 2 and abs(rows["SEA"]["st_epa_sum"] - 0.8) < 1e-9
    # special-teams plays don't count as scrimmage plays
    assert rows["SEA"]["plays"] == 2 and rows["LA"]["plays"] == 2


def test_aggregate_team_epa_special_teams_not_fumble_neutralized():
    pbp = _pbp().with_columns(pl.lit(1.0).alias("fumble"))
    rows = {r["team"]: r for r in aggregate_team_epa(pbp).iter_rows(named=True)}
    assert abs(rows["LA"]["st_epa_sum"] - 2.0) < 1e-9
    assert abs(rows["SEA"]["st_epa_sum"] - 0.8) < 1e-9


def test_aggregate_team_epa_team_without_special_teams_plays_gets_zero():
    pbp = _pbp().filter(~((pl.col("posteam") == "STL") & (pl.col("play_type") == "punt")))
    rows = {r["team"]: r for r in aggregate_team_epa(pbp).iter_rows(named=True)}
    assert rows["LA"]["st_plays"] == 0 and rows["LA"]["st_epa_sum"] == 0.0


def test_aggregate_qb_games_sums_dropbacks():
    out = aggregate_qb_games(_pbp())
    rows = {r["qb_id"]: r for r in out.iter_rows(named=True)}
    assert rows["QB-STL"]["dropbacks"] == 1 and rows["QB-STL"]["team"] == "LA"
    assert rows["QB-STL"]["cpoe_n"] == 1 and rows["QB-STL"]["cpoe_sum"] == 10.0
    # null-qb_epa dropback excluded; scramble (null passer_player_id) counted
    assert rows["QB-SEA"]["dropbacks"] == 3
    assert abs(rows["QB-SEA"]["qb_epa_sum"] - 2.6) < 1e-9


def _boom(*_a, **_k):
    raise RuntimeError("no data")


def test_current_season_fetch_failure_returns_empty_and_is_not_cached(monkeypatch, tmp_path):
    import seahawks_ml.ingest.nflverse as nv

    monkeypatch.setattr(nv.nfl, "load_pbp", _boom)
    monkeypatch.setattr(nv.nfl, "load_injuries", _boom)
    monkeypatch.setattr(nv.nfl, "load_snap_counts", _boom)
    monkeypatch.setattr(nv.nfl, "load_player_stats", _boom)
    out = nv.load_season_tables([2099], current_season=2099, cache_dir=tmp_path)
    assert out["player_stats"].columns == list(nv.PLAYER_STATS_SCHEMA)
    assert all(df.height == 0 for df in out.values())
    assert list(tmp_path.glob("*.parquet")) == []


def test_completed_season_fetch_failure_propagates(monkeypatch, tmp_path):
    import pytest

    import seahawks_ml.ingest.nflverse as nv

    monkeypatch.setattr(nv.nfl, "load_pbp", _boom)
    with pytest.raises(RuntimeError):
        nv.load_season_tables([2099], current_season=2100, cache_dir=tmp_path)
    assert list(tmp_path.glob("*.parquet")) == []

    monkeypatch.setattr(nv.nfl, "load_pbp", lambda s: _pbp())
    monkeypatch.setattr(nv.nfl, "load_injuries", _boom)
    with pytest.raises(RuntimeError):
        nv.load_season_tables([2099], current_season=2100, cache_dir=tmp_path)
    assert not (tmp_path / "injuries_2099.parquet").exists()


def test_stale_before_refetches_old_cache_once(monkeypatch, tmp_path):
    import os
    from datetime import UTC, datetime

    import seahawks_ml.ingest.nflverse as nv

    calls = []

    def load_pbp(season):
        calls.append(season)
        return _pbp()

    monkeypatch.setattr(nv.nfl, "load_pbp", load_pbp)
    monkeypatch.setattr(nv.nfl, "load_injuries", _boom)
    monkeypatch.setattr(nv.nfl, "load_snap_counts", _boom)
    nv.load_season_tables([2005], current_season=2006, cache_dir=tmp_path)
    assert calls == [2005]
    cutoff = datetime(2020, 1, 1, tzinfo=UTC)
    old = cutoff.timestamp() - 86400
    for f in tmp_path.glob("*.parquet"):
        os.utime(f, (old, old))
    nv.load_season_tables([2005], current_season=2006, cache_dir=tmp_path)
    assert calls == [2005]  # no stale_before -> cache kept
    nv.load_season_tables([2005], 2006, cache_dir=tmp_path, stale_before={2005: cutoff})
    assert calls == [2005, 2005]  # refetched
    nv.load_season_tables([2005], 2006, cache_dir=tmp_path, stale_before={2005: cutoff})
    assert calls == [2005, 2005]  # fresh now


def test_old_schema_team_epa_cache_is_refetched(monkeypatch, tmp_path):
    import seahawks_ml.ingest.nflverse as nv

    calls = []

    def load_pbp(season):
        calls.append(season)
        return _pbp()

    monkeypatch.setattr(nv.nfl, "load_pbp", load_pbp)
    monkeypatch.setattr(nv.nfl, "load_injuries", _boom)
    monkeypatch.setattr(nv.nfl, "load_snap_counts", _boom)
    nv.load_season_tables([2005], current_season=2006, cache_dir=tmp_path)
    path = tmp_path / "team_epa_2005.parquet"
    pl.read_parquet(path).select("game_id", "season", "team", "opponent", "epa_sum", "plays").write_parquet(path)
    out = nv.load_season_tables([2005], current_season=2006, cache_dir=tmp_path)
    assert calls == [2005, 2005]
    # a cache from before the special-teams columns is outdated too
    pl.read_parquet(path).drop("st_epa_sum", "st_plays").write_parquet(path)
    out = nv.load_season_tables([2005], current_season=2006, cache_dir=tmp_path)
    assert calls == [2005, 2005, 2005]
    assert set(nv.TEAM_EPA_SCHEMA) <= set(pl.read_parquet(path).columns)
    assert out["team_epa"].columns == list(nv.TEAM_EPA_SCHEMA)


def test_team_divisions_constant_is_complete():
    from seahawks_ml.ingest.nflverse import TEAM_DIVISIONS

    assert len(TEAM_DIVISIONS) == 32
    assert {conf for conf, _ in TEAM_DIVISIONS.values()} == {"AFC", "NFC"}
    divisions = [div for _, div in TEAM_DIVISIONS.values()]
    assert len(set(divisions)) == 8 and all(divisions.count(d) == 4 for d in set(divisions))
    assert all(div.startswith(conf) for conf, div in TEAM_DIVISIONS.values())
    assert TEAM_DIVISIONS["SEA"] == ("NFC", "NFC West")


def test_load_teams_normalizes_and_dedups(monkeypatch):
    import seahawks_ml.ingest.nflverse as nv

    rows = [{"team_abbr": t, "team_conf": c, "team_division": d} for t, (c, d) in nv.TEAM_DIVISIONS.items()]
    rows += [{"team_abbr": "STL", "team_conf": "NFC", "team_division": "NFC West"},
             {"team_abbr": "OAK", "team_conf": "AFC", "team_division": "AFC West"},
             {"team_abbr": "LAR", "team_conf": "NFC", "team_division": "NFC West"}]
    monkeypatch.setattr(nv.nfl, "load_teams", lambda: pl.DataFrame(rows).with_columns(pl.lit("x").alias("team_name")))
    out = nv.load_teams()
    assert out.columns == ["team", "conf", "division"]
    assert out.height == 32 and out["team"].n_unique() == 32
    assert out.filter(pl.col("team") == "LA").row(0) == ("LA", "NFC", "NFC West")


def test_load_teams_falls_back_to_constant(monkeypatch, capsys):
    import seahawks_ml.ingest.nflverse as nv

    monkeypatch.setattr(nv.nfl, "load_teams", _boom)
    out = nv.load_teams()
    assert out.height == 32 and "warning" in capsys.readouterr().out
    # an incomplete table also falls back
    monkeypatch.setattr(nv.nfl, "load_teams", lambda: pl.DataFrame(
        {"team_abbr": ["SEA"], "team_conf": ["NFC"], "team_division": ["NFC West"]}))
    assert nv.load_teams().height == 32


def _player_stats_raw():
    return pl.DataFrame({
        "player_id": ["00-1", "00-2"], "player_name": ["A", "B"], "position_group": ["WR", "RB"],
        "season": [2015, 2015], "week": [1, 1], "season_type": ["REG", "REG"],
        "game_id": ["2015_01_SD_STL", "2015_01_SD_STL"], "team": ["STL", "SD"],
        "targets": [5, 0], "receiving_epa": [1.5, None], "carries": [0, 12], "rushing_epa": [None, -2.0],
    })


def test_player_stats_cached_per_season_with_normalized_teams(monkeypatch, tmp_path):
    import seahawks_ml.ingest.nflverse as nv

    calls = []

    def load_player_stats(season, summary_level):
        calls.append((season, summary_level))
        return _player_stats_raw()

    monkeypatch.setattr(nv.nfl, "load_pbp", lambda s: _pbp())
    monkeypatch.setattr(nv.nfl, "load_injuries", _boom)
    monkeypatch.setattr(nv.nfl, "load_snap_counts", _boom)
    monkeypatch.setattr(nv.nfl, "load_player_stats", load_player_stats)
    # 2011 is before player stats; 2012 is the first season fetched
    for f in ("injuries", "snaps"):
        for s in (2011, 2012):
            pl.DataFrame(schema=nv.INJURIES_SCHEMA if f == "injuries" else nv.SNAPS_SCHEMA).write_parquet(
                tmp_path / f"{f}_{s}.parquet")
    out = nv.load_season_tables([2011, 2012], current_season=2013, cache_dir=tmp_path)
    assert calls == [(2012, "week")]
    assert (tmp_path / "player_stats_2012.parquet").exists()
    ps = out["player_stats"]
    assert ps.columns == list(nv.PLAYER_STATS_SCHEMA)
    assert sorted(ps["team"].to_list()) == ["LA", "LAC"]
    nv.load_season_tables([2012], current_season=2013, cache_dir=tmp_path)
    assert calls == [(2012, "week")]  # cached


def test_completed_season_player_stats_failure_propagates(monkeypatch, tmp_path):
    import pytest

    import seahawks_ml.ingest.nflverse as nv

    monkeypatch.setattr(nv.nfl, "load_pbp", lambda s: _pbp())
    monkeypatch.setattr(nv.nfl, "load_injuries", _boom)
    monkeypatch.setattr(nv.nfl, "load_snap_counts", _boom)
    monkeypatch.setattr(nv.nfl, "load_player_stats", _boom)
    for f, schema in (("injuries", nv.INJURIES_SCHEMA), ("snaps", nv.SNAPS_SCHEMA)):
        pl.DataFrame(schema=schema).write_parquet(tmp_path / f"{f}_2015.parquet")
    with pytest.raises(RuntimeError):
        nv.load_season_tables([2015], current_season=2016, cache_dir=tmp_path)
    assert not (tmp_path / "player_stats_2015.parquet").exists()


def _injuries_raw():
    return pl.DataFrame({
        "season": [2015] * 6, "week": [3] * 6, "team": ["STL", "SEA", "SEA", "SEA", "SEA", "SEA"],
        "gsis_id": [f"00-{i}" for i in range(6)], "full_name": list("ABCDEF"),
        "position": ["WR"] * 6,
        "report_status": ["Questionable", "Out", "Questionable", "Doubtful", None, "Note"],
        "practice_status": ["Did Not Participate In Practice", "Limited Participation in Practice",
                            "Full Participation in Practice", " ", None, "Note"],
    })


def test_practice_bucket_normalizes_nflverse_values():
    from seahawks_ml.ingest.nflverse import practice_bucket

    assert practice_bucket("Did Not Participate In Practice") == "dnp"
    assert practice_bucket("Limited Participation in Practice") == "limited"
    assert practice_bucket("Full Participation in Practice") == "full"
    for raw in ("Note", " ", "", None, "\n"):
        assert practice_bucket(raw) is None


def test_injuries_cache_keeps_normalized_practice_status(monkeypatch, tmp_path):
    import seahawks_ml.ingest.nflverse as nv

    monkeypatch.setattr(nv.nfl, "load_pbp", lambda s: _pbp())
    monkeypatch.setattr(nv.nfl, "load_injuries", lambda s: _injuries_raw())
    monkeypatch.setattr(nv.nfl, "load_snap_counts", _boom)
    out = nv.load_season_tables([2010], current_season=2011, cache_dir=tmp_path)
    inj = out["injuries"]
    assert inj.columns == list(nv.INJURIES_SCHEMA)
    assert inj["practice_status"].to_list() == ["dnp", "limited", "full", None, None, None]
    assert inj["team"].to_list()[0] == "LA"


def test_old_schema_injuries_cache_is_refetched(monkeypatch, tmp_path):
    import seahawks_ml.ingest.nflverse as nv

    calls = []

    def load_injuries(season):
        calls.append(season)
        return _injuries_raw()

    monkeypatch.setattr(nv.nfl, "load_pbp", lambda s: _pbp())
    monkeypatch.setattr(nv.nfl, "load_injuries", load_injuries)
    monkeypatch.setattr(nv.nfl, "load_snap_counts", _boom)
    nv.load_season_tables([2010], current_season=2011, cache_dir=tmp_path)
    nv.load_season_tables([2010], current_season=2011, cache_dir=tmp_path)
    assert calls == [2010]  # cached
    path = tmp_path / "injuries_2010.parquet"
    pl.read_parquet(path).drop("practice_status").write_parquet(path)  # a cache from before practice status
    out = nv.load_season_tables([2010], current_season=2011, cache_dir=tmp_path)
    assert calls == [2010, 2010]
    assert "practice_status" in pl.read_parquet(path).columns
    assert out["injuries"]["practice_status"].to_list()[0] == "dnp"


def _contracts_raw():
    return pl.DataFrame({
        "player": ["A", "B", "C", "D"], "position": ["WR", "LT", "ED", "QB"], "team": ["SEA"] * 4,
        "is_active": [True] * 4, "year_signed": [2019, 0, 2021, 2022], "years": [4, 1, 2, 5],
        "value": [1.0] * 4, "apy": [1.0] * 4, "guaranteed": [1.0] * 4,
        "apy_cap_pct": [0.05, 0.004, 0.01, 0.2], "gsis_id": ["00-1", "00-2", None, "00-4"],
    }).with_columns(pl.col("year_signed").cast(pl.Int32))


def test_contracts_cached_and_refreshed_after_a_week(monkeypatch, tmp_path):
    import os
    import time

    import seahawks_ml.ingest.nflverse as nv

    calls = []

    def load_contracts():
        calls.append(1)
        return _contracts_raw()

    monkeypatch.setattr(nv.nfl, "load_contracts", load_contracts)
    out = nv.load_contracts(cache_dir=tmp_path)
    assert out.columns == list(nv.CONTRACTS_SCHEMA)
    assert out["gsis_id"].to_list() == ["00-1", "00-2", "00-4"]  # null gsis ids dropped
    assert (tmp_path / "contracts.parquet").exists()
    nv.load_contracts(cache_dir=tmp_path)
    assert len(calls) == 1  # fresh cache
    old = time.time() - 8 * 86400
    os.utime(tmp_path / "contracts.parquet", (old, old))
    nv.load_contracts(cache_dir=tmp_path)
    assert len(calls) == 2  # older than 7 days
    pl.read_parquet(tmp_path / "contracts.parquet").drop("apy_cap_pct").write_parquet(tmp_path / "contracts.parquet")
    out = nv.load_contracts(cache_dir=tmp_path)
    assert len(calls) == 3 and out.columns == list(nv.CONTRACTS_SCHEMA)  # missing column


def test_contracts_fetch_failure_falls_back_to_stale_cache_or_empty(monkeypatch, tmp_path, capsys):
    import os
    import time

    import seahawks_ml.ingest.nflverse as nv

    monkeypatch.setattr(nv.nfl, "load_contracts", _boom)
    out = nv.load_contracts(cache_dir=tmp_path)
    assert out.height == 0 and out.columns == list(nv.CONTRACTS_SCHEMA)
    assert not (tmp_path / "contracts.parquet").exists()
    assert "warning" in capsys.readouterr().out
    monkeypatch.setattr(nv.nfl, "load_contracts", _contracts_raw)
    nv.load_contracts(cache_dir=tmp_path)
    old = time.time() - 8 * 86400
    os.utime(tmp_path / "contracts.parquet", (old, old))
    monkeypatch.setattr(nv.nfl, "load_contracts", _boom)
    assert nv.load_contracts(cache_dir=tmp_path).height == 3
