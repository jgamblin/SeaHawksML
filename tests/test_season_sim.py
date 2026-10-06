import random
import time
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from seahawks_ml.pipeline.season_sim import (
    append_snapshot,
    has_snapshot_for,
    season_in_progress,
    simulate_season,
    validate_sim,
)

NOW = datetime(2026, 10, 6, 12, tzinfo=UTC)
NFC = {"SEA": "NFC West", "SF": "NFC West", "LA": "NFC West", "ARI": "NFC West",
       "DAL": "NFC East", "NYG": "NFC East", "PHI": "NFC East", "WAS": "NFC East"}
AFC = {"KC": "AFC West", "BUF": "AFC East"}


def _teams(divs=None):
    divs = divs or {**NFC, **AFC}
    return pl.DataFrame({"team": list(divs), "conf": [d[:3] for d in divs.values()], "division": list(divs.values())})


def _games(rows):
    """rows: (home, away, margin or None)."""
    return pl.DataFrame(
        [{"game_id": f"g{i}", "season": 2026, "week": i + 1, "game_type": "REG", "home_team": h, "away_team": a,
          "margin": m} for i, (h, a, m) in enumerate(rows)],
        schema={"game_id": pl.Utf8, "season": pl.Int64, "week": pl.Int64, "game_type": pl.Utf8,
                "home_team": pl.Utf8, "away_team": pl.Utf8, "margin": pl.Int64})


def _sim(rows, probs=None, teams=None, n_sims=200, seed=0, **kw):
    return simulate_season(_games(rows), probs or {}, teams if teams is not None else _teams(), 2026,
                           n_sims=n_sims, seed=seed, now=NOW, **kw)


def test_head_to_head_breaks_two_team_division_tie():
    # SEA and SF both 2-1 (all division games); SEA won the head-to-head game
    rows = [("SEA", "SF", 3), ("SF", "LA", 7), ("SF", "ARI", 7), ("SEA", "ARI", 10), ("LA", "SEA", 3)]
    out = _sim(rows)
    assert out["record_now"] == "2-1-0"
    assert out["p_division"] == 1.0
    flipped = [("SEA", "SF", -3)] + rows[1:]  # SF wins the head-to-head: SEA 1-2 now
    assert _sim(flipped)["p_division"] == 0.0


def test_division_record_breaks_tie_without_head_to_head():
    # SEA and SF both 1-1 and never met; SEA is 1-0 in the division, SF 0-1
    rows = [("SEA", "LA", 7), ("DAL", "SEA", 3), ("ARI", "SF", 3), ("SF", "NYG", 7),
            ("DAL", "ARI", 7), ("NYG", "ARI", 7)]  # ARI 1-2
    assert _sim(rows)["p_division"] == 1.0


def test_conference_record_breaks_tie():
    # SEA and SF both 1-1, no division games: SEA 1-0 vs NFC, SF 0-1 vs NFC
    rows = [("SEA", "DAL", 7), ("KC", "SEA", 3), ("SF", "KC", 7), ("NYG", "SF", 3)]
    assert _sim(rows)["p_division"] == 1.0


def test_random_fallback_is_seeded():
    rows = [("SEA", "LA", 7), ("SF", "ARI", 7)]  # SEA and SF identical in every tiebreaker
    a, b = _sim(rows, n_sims=2000, seed=1), _sim(rows, n_sims=2000, seed=1)
    assert a == b
    assert 0.4 < a["p_division"] < 0.6


def test_wild_cards_and_top_seed():
    # one 8-team division: winner + 3 wild cards
    divs = {t: "NFC West" for t in ["SEA", "SF", "LA", "ARI", "DAL", "NYG", "PHI", "WAS"]} | AFC
    rows = ([("SF", "KC", 7)] * 3 + [("SEA", "KC", 7)] * 2 + [("KC", "SEA", 7)]
            + [("LA", "KC", 7), ("KC", "LA", 7), ("ARI", "KC", 7), ("KC", "ARI", 7), ("KC", "ARI", 7)]
            + [("KC", t, 7) for t in ["DAL", "NYG", "PHI", "WAS"]])
    out = _sim(rows, teams=_teams(divs))
    assert (out["p_playoffs"], out["p_division"], out["p_top_seed"]) == (1.0, 0.0, 0.0)
    worst = [r for r in rows if "SEA" not in r] + [("KC", "SEA", 7)] * 2 + [("DAL", "KC", 7)]  # LA, DAL, ARI
    assert _sim(worst, teams=_teams(divs))["p_playoffs"] == 0.0
    best = rows + [("SEA", "BUF", 7)] * 3 + [("KC", "SF", 7)]  # SEA 5-1, SF 3-1
    out = _sim(best, teams=_teams(divs))
    assert (out["p_playoffs"], out["p_division"], out["p_top_seed"]) == (1.0, 1.0, 1.0)


def test_head_to_head_orders_division_winners():
    # SEA and DAL win their divisions at 2-1; DAL won the head-to-head game
    rows = [("DAL", "SEA", 3), ("SEA", "LA", 7), ("SEA", "SF", 7), ("DAL", "NYG", 7), ("KC", "DAL", 7)]
    out = _sim(rows)
    assert out["p_division"] == 1.0 and out["p_top_seed"] == 0.0
    rows[0] = ("DAL", "SEA", -3)  # SEA 3-0 now
    assert _sim(rows)["p_top_seed"] == 1.0


def test_remaining_games_are_simulated_and_ties_count_half():
    rows = [("SEA", "SF", 0), ("SEA", "LA", 3), ("ARI", "SEA", None), ("SEA", "DAL", None)]
    out = _sim(rows, probs={"g2": 0.5, "g3": 0.5}, n_sims=4000)
    assert out["record_now"] == "1-0-1"
    assert out["wins_mean"] == pytest.approx(2.5, abs=0.05)
    dist = {d["wins"]: d["prob"] for d in out["win_dist"]}
    # final wins 1.5, 2.5, 3.5 -> histogram buckets 2, 3, 4 (round half up)
    assert set(dist) == {2, 3, 4} and sum(dist.values()) == pytest.approx(1.0)
    assert dist[3] == pytest.approx(0.5, abs=0.04)
    assert out["wins_p10"] <= out["wins_p50"] <= out["wins_p90"]
    assert out["n_sims"] == 4000 and out["season"] == 2026 and out["as_of"] == NOW.isoformat()


def test_certain_remaining_games():
    out = _sim([("SEA", "SF", None), ("LA", "SEA", None)], probs={"g0": 1.0, "g1": 1.0})
    assert out["wins_mean"] == 1.0 and out["win_dist"] == [{"wins": 1, "prob": 1.0}]


def test_non_regular_season_games_ignored_and_missing_prob_raises():
    games = _games([("SEA", "SF", 7), ("SF", "SEA", None)]).with_columns(
        pl.Series("game_type", ["REG", "WC"]))
    out = simulate_season(games, {}, _teams(), 2026, n_sims=10, now=NOW)
    assert out["record_now"] == "1-0-0"
    with pytest.raises(KeyError):
        _sim([("SEA", "SF", None)])


def test_unknown_team_raises():
    with pytest.raises(ValueError, match="XYZ"):
        _sim([("SEA", "XYZ", 3)])


def test_snapshot_validation_and_once_per_day(tmp_path):
    out = _sim([("SEA", "SF", 3)])
    rec = {**out, "model_version": "v1"}
    validate_sim(rec)
    with pytest.raises(ValueError):
        validate_sim({**rec, "extra": 1})
    path = tmp_path / "s.jsonl"
    assert append_snapshot(rec, path) is True
    assert append_snapshot(rec, path) is False  # same UTC date
    later = {**rec, "as_of": (NOW + timedelta(days=1)).isoformat()}
    assert append_snapshot(later, path) is True
    from seahawks_ml.pipeline.history import read_history
    log = read_history(path)
    assert len(log) == 2
    assert has_snapshot_for(log, NOW) and not has_snapshot_for(log, NOW + timedelta(days=2))


def test_season_in_progress():
    games = _games([("SEA", "SF", 3), ("SF", "SEA", None)]).with_columns(
        pl.Series("kickoff_utc", [NOW - timedelta(days=3), NOW + timedelta(days=4)]))
    assert season_in_progress(games, NOW)
    assert season_in_progress(games, NOW + timedelta(days=6))  # final results come in after the last game
    assert not season_in_progress(games, NOW + timedelta(days=8))  # regular season over


def _league_schedule(n_weeks=17, played_weeks=4, seed=0):
    from seahawks_ml.ingest.nflverse import TEAM_DIVISIONS
    rng = random.Random(seed)
    teams = list(TEAM_DIVISIONS)
    rows, probs = [], {}
    for w in range(n_weeks):
        rng.shuffle(teams)
        for i in range(0, 32, 2):
            gid = f"g{len(rows)}"
            margin = rng.choice([-7, -3, 3, 7, 10]) if w < played_weeks else None
            rows.append((teams[i], teams[i + 1], margin))
            if margin is None:
                probs[gid] = rng.uniform(0.2, 0.8)
    divs = pl.DataFrame([(t, c, d) for t, (c, d) in TEAM_DIVISIONS.items()],
                        schema=["team", "conf", "division"], orient="row")
    return rows, probs, divs


def test_full_league_simulation_is_fast():
    rows, probs, divs = _league_schedule()
    start = time.perf_counter()
    out = _sim(rows, probs=probs, teams=divs, n_sims=2000)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0
    assert 0 <= out["p_top_seed"] <= out["p_division"] <= out["p_playoffs"] <= 1
    assert sum(d["prob"] for d in out["win_dist"]) == pytest.approx(1.0)


def test_remaining_game_probs_uses_model_with_elo_fallback():
    import numpy as np

    from seahawks_ml.features.build import build_features
    from seahawks_ml.pipeline.season_sim import remaining_game_probs
    from seahawks_ml.stadiums import load_stadiums
    from tests.synthetic import make_raw

    frame = build_features(make_raw(seasons=(2012, 2013, 2014), unplayed_last_week=True), load_stadiums())
    unplayed = frame.filter((pl.col("season") == 2014) & pl.col("margin").is_null())["game_id"].to_list()
    broken = unplayed[0]
    frame = frame.with_columns(pl.when(pl.col("game_id") == broken).then(None).otherwise(pl.col("qb_epa_diff"))
                               .alias("qb_epa_diff"))

    class Fake:
        def predict(self, rows):
            assert broken not in rows["game_id"].to_list()
            return {"p_win": np.full(rows.height, 0.6)}

    probs = remaining_game_probs(frame, Fake(), 2014)
    assert set(probs) == set(unplayed) and len(unplayed) == 2
    assert all(probs[g] == 0.6 for g in unplayed if g != broken)
    assert 0 < probs[broken] < 1 and probs[broken] != 0.6
    assert remaining_game_probs(frame, Fake(), 2013) == {}
