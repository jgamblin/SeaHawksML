"""Win/loss classifier and Platt scaling. Ties enter as half a win and half a loss."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from seahawks_ml.models.metrics import outcome


def _expand_ties(X: np.ndarray, margin: np.ndarray, w: np.ndarray | None):
    """Duplicate every row as (y=1, weight*o) and (y=0, weight*(1-o))."""
    o = outcome(margin)
    w = np.ones(len(o)) if w is None else np.asarray(w, float)
    X2 = np.vstack([X, X])
    y2 = np.concatenate([np.ones(len(o)), np.zeros(len(o))])
    w2 = np.concatenate([w * o, w * (1 - o)])
    keep = w2 > 0
    return X2[keep], y2[keep], w2[keep]


class WinClassifier:
    def __init__(self, c: float = 1.0):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=c, max_iter=2000)

    def fit(self, X: np.ndarray, margin: np.ndarray, w: np.ndarray | None = None) -> "WinClassifier":
        Xs = self.scaler.fit_transform(X)
        X2, y2, w2 = _expand_ties(Xs, margin, w)
        self.model.fit(X2, y2, sample_weight=w2)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class PlattScaler:
    def __init__(self):
        self.model = LogisticRegression(C=1e6, max_iter=1000)

    def fit(self, p: np.ndarray, margin: np.ndarray) -> "PlattScaler":
        X2, y2, w2 = _expand_ties(_logit(p)[:, None], margin, None)
        self.model.fit(X2, y2, sample_weight=w2)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(_logit(p)[:, None])[:, 1]
