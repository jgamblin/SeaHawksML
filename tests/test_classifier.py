import numpy as np

from seahawks_ml.models.classifier import PlattScaler, WinClassifier


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    margin = np.round(6 * X[:, 0] - 3 * X[:, 1] + rng.normal(0, 12, n))
    return X, margin


def test_win_classifier_handles_ties():
    X, margin = _data()
    margin[:20] = 0
    clf = WinClassifier(c=1.0).fit(X, margin)
    p = clf.predict_proba(X)
    assert ((p > 0) & (p < 1)).all()
    assert np.corrcoef(p, X[:, 0])[0, 1] > 0.5


def test_platt_scaler_corrects_overconfidence():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.3, 0.7, 4000)
    margin = np.where(rng.uniform(size=4000) < true_p, 3.0, -3.0)
    overconfident = 1 / (1 + np.exp(-3 * np.log(true_p / (1 - true_p))))
    fixed = PlattScaler().fit(overconfident, margin).transform(overconfident)
    assert np.abs(fixed - true_p).mean() < np.abs(overconfident - true_p).mean()
