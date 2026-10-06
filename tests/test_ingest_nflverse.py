import polars as pl

from seahawks_ml.ingest.nflverse import aggregate_qb_games, aggregate_team_epa


def _pbp():
    return pl.DataFrame({
        "game_id": ["g1"] * 8,
        "season": [2015] * 8,
        "posteam": ["STL", "STL", "STL", "SEA", "SEA", None, "SEA", "SEA"],
        "defteam": ["SEA", "SEA", "SEA", "STL", "STL", None, "STL", "STL"],
        "play_type": ["pass", "run", "punt", "pass", "pass", None, "run", "pass"],
        "epa": [0.5, -0.1, 2.0, 1.0, None, 0.3, 0.9, 0.7],
        "qb_dropback": [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        "passer_player_id": ["QB-STL", None, None, "QB-SEA", "QB-SEA", None, None, "QB-SEA"],
        "passer_id": ["QB-STL", None, None, "QB-SEA", "QB-SEA", None, "QB-SEA", "QB-SEA"],
        "qb_epa": [0.5, -0.1, 2.0, 1.0, None, 0.3, 0.9, 0.7],
        "cpoe": [10.0, None, None, None, -5.0, None, None, None],
        "two_point_attempt": [0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    })


def test_aggregate_team_epa_uses_scrimmage_plays_and_normalizes_teams():
    out = aggregate_team_epa(_pbp())
    rows = {r["team"]: r for r in out.iter_rows(named=True)}
    assert set(rows) == {"LA", "SEA"}  # STL -> LA, null posteam dropped
    assert rows["LA"]["plays"] == 2 and abs(rows["LA"]["epa_sum"] - 0.4) < 1e-9  # punt excluded
    assert rows["LA"]["opponent"] == "SEA"
    assert rows["SEA"]["plays"] == 2  # null epa and two-point attempt excluded
    assert abs(rows["SEA"]["epa_sum"] - 1.9) < 1e-9


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
    out = nv.load_season_tables([2099], current_season=2099, cache_dir=tmp_path)
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
