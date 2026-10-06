import numpy as np
import pytest

from seahawks_ml.models.regressors import make_regressor


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    margin = np.round(6 * X[:, 0] - 3 * X[:, 1] + rng.normal(0, 12, n))
    return X, margin


@pytest.mark.parametrize("kind,params", [("ridge", {"alpha": 1.0}), ("lgbm", {"n_estimators": 50})])
def test_regressors_fit_predict_and_contributions(kind, params):
    X, y = _data()
    model = make_regressor(kind, params).fit(X, y)
    pred = model.predict(X)
    assert np.corrcoef(pred, y)[0, 1] > 0.3
    contrib = model.contributions(X[:5])
    assert contrib.shape == (5, 3)
    assert np.isfinite(contrib).all()


def test_lgbm_contributions_plus_bias_equal_prediction():
    X, y = _data()
    model = make_regressor("lgbm", {"n_estimators": 50}).fit(X, y)
    full = model.model.predict(X[:5], pred_contrib=True)
    assert np.allclose(model.contributions(X[:5]).sum(axis=1) + full[:, -1], model.predict(X[:5]))


def test_ridge_contributions_sum_to_prediction_minus_intercept():
    X, y = _data()
    model = make_regressor("ridge", {"alpha": 1.0}).fit(X, y)
    assert np.allclose(model.contributions(X[:5]).sum(axis=1) + model.model.intercept_, model.predict(X[:5]))


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        make_regressor("svm", {})
