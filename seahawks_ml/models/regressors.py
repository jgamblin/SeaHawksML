"""Margin regressors with a shared interface: fit / predict / contributions."""

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

LGBM_DEFAULTS = {
    "num_leaves": 4,
    "n_estimators": 200,
    "learning_rate": 0.03,
    "min_child_samples": 50,
    "reg_lambda": 5.0,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "random_state": 0,
}


class RidgeMargin:
    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> "RidgeMargin":
        self.model.fit(self.scaler.fit_transform(X), y, sample_weight=w)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """Per-feature contribution in points, relative to the training-average game."""
        return self.scaler.transform(X) * self.model.coef_


class LGBMMargin:
    def __init__(self, **params):
        self.params = {**LGBM_DEFAULTS, **params}
        self.model = LGBMRegressor(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> "LGBMMargin":
        self.model.fit(X, y, sample_weight=w)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def contributions(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, pred_contrib=True)[:, :-1]  # last column is the bias


def make_regressor(kind: str, params: dict):
    if kind == "ridge":
        return RidgeMargin(**params)
    if kind == "lgbm":
        return LGBMMargin(**params)
    raise ValueError(f"unknown model kind {kind!r}")
