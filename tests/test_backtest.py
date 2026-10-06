import numpy as np

from seahawks_ml.models import backtest
from seahawks_ml.models.backtest import score, tune, walk_forward
from seahawks_ml.models.pipeline import ModelConfig
from tests.synthetic import make_feature_frame


def test_walk_forward_predicts_only_test_seasons():
    frame = make_feature_frame()
    preds = walk_forward(frame, ModelConfig(), [2014, 2015])
    assert sorted(preds["season"].unique().to_list()) == [2014, 2015]
    assert preds.height == 240
    for col in ("p_model", "p_elo", "p_home", "p_vegas"):
        p = preds[col].to_numpy()
        assert ((p > 0) & (p < 1)).all(), col


def test_score_reports_model_and_baselines():
    frame = make_feature_frame()
    result = score(walk_forward(frame, ModelConfig(), [2014, 2015]))
    for name in ("model", "elo", "home", "vegas"):
        assert {"log_loss", "brier", "accuracy", "log_loss_ci90", "calibration"} <= set(result[name])
    assert result["model"]["log_loss"] < result["home"]["log_loss"]
    assert result["seahawks_only"]["n"] == 30
    assert [r["season"] for r in result["per_season"]] == [2014, 2015]


def test_tune_prefers_ridge_within_tolerance(monkeypatch):
    monkeypatch.setattr(backtest, "RIDGE_GRID", [{"alpha": 10.0}])
    monkeypatch.setattr(backtest, "LGBM_GRID", [{"n_estimators": 20}])
    monkeypatch.setattr(backtest, "HALF_LIFE_GRID", (6.0,))
    monkeypatch.setattr(backtest, "PROB_METHOD_GRID", [("gaussian", 2.0), ("empirical", 6.0)])
    monkeypatch.setattr(backtest, "ENSEMBLE_GRID", (0.0,))
    frame = make_feature_frame()
    best, trials = tune(frame, [2014, 2015], log=lambda *_: None)
    assert best.kind == "ridge"  # data is linear, so LightGBM shouldn't win by the margin
    assert len(trials) == 4  # ridge, lgbm, empirical, platt=True
    assert all(np.isfinite(t["log_loss"]) for t in trials)
