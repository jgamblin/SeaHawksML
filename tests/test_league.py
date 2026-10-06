from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from seahawks_ml.features.elo import elo_win_prob
from seahawks_ml.models.pipeline import ModelConfig, fit_model
from seahawks_ml.pipeline.history import append_record, read_history
from seahawks_ml.pipeline.league import (
    due_league_games,
    league_scorecard,
    make_league_predictions,
    new_league_results,
    validate_league,
)
from tests.synthetic import make_feature_frame

KICK = datetime(2026, 10, 11, 17, tzinfo=UTC)


def _games(*kicks):
    return pl.DataFrame({"game_id": [f"g{i}" for i in range(len(kicks))], "kickoff_utc": list(kicks),
                         "margin": [None] * len(kicks)})


def test_due_window_edges():
    games = _games(KICK)
    assert due_league_games(games, KICK - timedelta(hours=24, minutes=1), set()) == []
    assert due_league_games(games, KICK - timedelta(hours=24), set()) == ["g0"]
    assert due_league_games(games, KICK - timedelta(hours=6, minutes=1), set()) == ["g0"]
    assert due_league_games(games, KICK - timedelta(hours=6), set()) == []


def test_due_skips_logged():
    games = _games(KICK, KICK + timedelta(hours=3))
    assert due_league_games(games, KICK - timedelta(hours=10), {"g0"}) == ["g1"]


@pytest.fixture(scope="module")
def model_and_frame():
    frame = make_feature_frame()
    model = fit_model(frame, ModelConfig(inner_folds=3), list(range(2009, 2015)))
    rows = frame.filter(pl.col("season") == 2015).head(3).with_columns(
        pl.lit(5).alias("week"), pl.lit(KICK).alias("kickoff_utc"))
    return model, rows


def test_make_league_predictions(model_and_frame):
    model, rows = model_and_frame
    rows = rows.with_columns(pl.when(pl.col("game_id") == rows["game_id"][0]).then(None)
                             .otherwise(pl.col("spread_line")).alias("spread_line"))
    now = KICK - timedelta(hours=12)
    recs = make_league_predictions(rows, model, now, "v1")
    assert len(recs) == 3
    for r in recs:
        validate_league(r)
        assert r["type"] == "league_prediction" and r["predicted_at"] == now.isoformat()
        assert 0 < r["p_home"] < 1 and r["model_version"] == "v1"
    assert recs[0]["p_vegas_home"] is None and recs[1]["p_vegas_home"] is not None
    g = rows.row(1, named=True)
    assert recs[1]["p_elo_home"] == round(elo_win_prob(g["elo_home_pre"] - g["elo_away_pre"], g["neutral"]), 4)


def test_validate_rejects_bad_keys():
    with pytest.raises(ValueError):
        validate_league({"type": "league_result", "game_id": "g"})
    with pytest.raises(ValueError):
        validate_league({"type": "nope"})


def test_validate_rejects_non_finite_or_out_of_range():
    base = _pred("g0", 1, .6, .55, .5)
    validate_league(base)
    validate_league({**base, "p_vegas_home": None})
    for key, bad in [("p_home", 0.0), ("p_home", 1.0), ("p_home", float("nan")), ("margin_home", float("nan")),
                     ("margin_home", float("inf")), ("p_elo_home", float("nan")), ("p_elo_home", None),
                     ("p_vegas_home", float("nan")), ("p_vegas_home", 1.0), ("p_vegas_home", 0.0)]:
        with pytest.raises(ValueError):
            validate_league({**base, key: bad})


def test_make_league_predictions_skips_non_finite(model_and_frame, capsys):
    from seahawks_ml.features.columns import FEATURE_COLUMNS
    model, rows = model_and_frame
    col = FEATURE_COLUMNS[0]
    ids = rows["game_id"].to_list()
    bad = rows.with_columns(pl.when(pl.col("game_id") == ids[0]).then(None).otherwise(pl.col(col)).alias(col))
    recs = make_league_predictions(bad, model, KICK - timedelta(hours=12), "v1")
    assert [r["game_id"] for r in recs] == ids[1:]
    assert ids[0] in capsys.readouterr().out
    nan = rows.with_columns(pl.when(pl.col("game_id") == ids[1]).then(float("nan")).otherwise(pl.col(col))
                            .cast(pl.Float64).alias(col))
    assert [r["game_id"] for r in make_league_predictions(nan, model, KICK, "v1")] == [ids[0], ids[2]]


def _pred(gid, week, p, vegas, elo, season=2026):
    return {"type": "league_prediction", "game_id": gid, "season": season, "week": week,
            "predicted_at": "2026-10-10T12:00:00+00:00", "kickoff_utc": "2026-10-11T17:00:00+00:00",
            "home_team": "SEA", "away_team": "SF", "p_home": p, "margin_home": 3.0,
            "p_vegas_home": vegas, "p_elo_home": elo, "model_version": "v1"}


def _res(gid, margin):
    return {"type": "league_result", "game_id": gid, "recorded_at": "2026-10-12T00:00:00+00:00",
            "home_score": 20 + max(margin, 0), "away_score": 20 + max(-margin, 0), "margin_home": margin}


def test_new_league_results():
    log = [_pred("g0", 5, .6, .55, .5), _pred("g1", 5, .6, .55, .5), _pred("g2", 5, .6, .55, .5), _res("g2", 3)]
    games = pl.DataFrame({"game_id": ["g0", "g1", "g2"], "margin": [7, None, 3],
                          "home_score": [27, None, 23], "away_score": [20, None, 20]})
    out = new_league_results(log, games, KICK)
    assert [r["game_id"] for r in out] == ["g0"]
    validate_league(out[0])
    assert out[0]["margin_home"] == 7 and out[0]["home_score"] == 27


def test_scorecard_empty():
    sc = league_scorecard([], 2026)
    assert sc["n"] == 0 and sc["by_week"]["weeks"] == []
    assert league_scorecard([_pred("g0", 5, .6, .5, .5)], 2026)["n"] == 0


def test_scorecard_values_and_common_set():
    log = [_pred("g0", 1, .8, .7, .6), _res("g0", 7),
           _pred("g1", 2, .8, None, .6), _res("g1", -3),
           _pred("g2", 2, .5, .5, .5), _res("g2", 0),
           _pred("old", 1, .9, .9, .9, season=2025), _res("old", 3)]
    sc = league_scorecard(log, 2026)
    assert sc["season"] == 2026 and sc["n"] == 2 and sc["n_all"] == 3  # g1 has no vegas
    assert sc["model"]["n"] == sc["vegas"]["n"] == sc["elo"]["n"] == 2
    assert sc["model"]["log_loss"] == pytest.approx((0.2231435513 + 0.6931471806) / 2, abs=1e-6)
    assert sc["by_week"]["weeks"] == [1, 2]
    assert sc["by_week"]["model"][0] == pytest.approx(0.2231435513, abs=1e-4)
    assert sc["by_week"]["model"][1] == pytest.approx(sc["model"]["log_loss"], abs=1e-4)


def test_league_log_roundtrip(tmp_path):
    p = tmp_path / "l.jsonl"
    append_record(_pred("g0", 1, .5, .5, .5), p, validate_league)
    append_record(_res("g0", 3), p, validate_league)
    assert len(read_history(p)) == 2
    with pytest.raises(ValueError):
        append_record({"type": "prediction"}, p, validate_league)
