import numpy as np
import polars as pl
import pytest

from seahawks_ml.features.columns import FEATURE_COLUMNS
from seahawks_ml.models.pipeline import ModelConfig, fit_model, recency_weights
from tests.synthetic import make_feature_frame


def test_recency_weights_halve_each_half_life():
    w = recency_weights(np.array([2020, 2016, 2012]), 2020, half_life=4)
    assert w.tolist() == [1.0, 0.5, 0.25]


@pytest.mark.parametrize("config", [
    ModelConfig(),
    ModelConfig(kind="lgbm", params={"n_estimators": 50}),
    ModelConfig(ensemble_weight=0.5, use_platt=True),
    ModelConfig(prob_method="empirical", bandwidth=3.0),
])
def test_fit_model_predicts_sensible_probabilities(config):
    frame = make_feature_frame()
    model = fit_model(frame, config, list(range(2009, 2015)))
    test = frame.filter(pl.col("season") == 2015)
    out = model.predict(test)
    assert ((out["p_win"] > 0) & (out["p_win"] < 1)).all()
    assert np.corrcoef(out["p_win"], test["elo_diff"].to_numpy())[0, 1] > 0.5
    assert (out["margin_lo"] <= out["margin_hi"]).all()
    assert model.contributions(test).shape == (test.height, len(FEATURE_COLUMNS))
    assert 0.4 < model.vegas_prob(np.array([0.0]))[0] < 0.6


def test_fit_model_ignores_rows_outside_train_seasons():
    frame = make_feature_frame()
    poisoned = frame.with_columns(
        pl.when(pl.col("season") == 2015).then(pl.col("margin") * -10).otherwise(pl.col("margin")).alias("margin"))
    a = fit_model(frame, ModelConfig(), list(range(2009, 2015))).predict(frame.head(20))
    b = fit_model(poisoned, ModelConfig(), list(range(2009, 2015))).predict(frame.head(20))
    assert np.allclose(a["p_win"], b["p_win"])


def test_fit_model_needs_enough_seasons():
    with pytest.raises(ValueError):
        fit_model(make_feature_frame(), ModelConfig(), [2009, 2010])
