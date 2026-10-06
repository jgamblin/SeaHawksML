"""Margin -> win probability.

EmpiricalMarginDist: for a predicted margin m, weight historical games by a Gaussian
kernel on (their predicted margin - m) and use their *actual* margins as the outcome
distribution. Keeps NFL key-number clustering (3, 7, 10, 14) and ties.

GaussianMarginDist: Phi(m / sigma). Smoother; on 2012-2023 real data it scored slightly
better for win probability (key numbers matter more for spreads than for win/loss).

Which one is used is a tuned config choice (ModelConfig.prob_method).
"""

import numpy as np
from scipy.stats import norm


class EmpiricalMarginDist:
    def __init__(self, bandwidth: float = 2.0):
        self.bandwidth = bandwidth
        self.pred: np.ndarray | None = None
        self.actual: np.ndarray | None = None

    def fit(self, pred_margin: np.ndarray, actual_margin: np.ndarray) -> "EmpiricalMarginDist":
        self.pred = np.asarray(pred_margin, dtype=float)
        self.actual = np.asarray(actual_margin, dtype=float)
        return self

    def _weights(self, m: np.ndarray) -> np.ndarray:
        """(len(m), n_train) normalized kernel weights; stable for m outside the training range."""
        z2 = ((self.pred[None, :] - np.asarray(m, float)[:, None]) / self.bandwidth) ** 2
        w = np.exp(-0.5 * (z2 - z2.min(axis=1, keepdims=True)))
        return w / w.sum(axis=1, keepdims=True)

    def prob_win(self, m: np.ndarray) -> np.ndarray:
        """P(margin > 0) + 0.5 * P(margin == 0)."""
        w = self._weights(m)
        return w @ (self.actual > 0).astype(float) + 0.5 * (w @ (self.actual == 0).astype(float))

    def interval(self, m: np.ndarray, level: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
        w = self._weights(m)
        order = np.argsort(self.actual)
        cdf = np.cumsum(w[:, order], axis=1)
        sorted_actual = self.actual[order]
        lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
        lo = sorted_actual[np.argmax(cdf >= lo_q, axis=1)]
        hi = sorted_actual[np.argmax(cdf >= hi_q, axis=1)]
        return lo, hi


class GaussianMarginDist:
    """Normal outcome distribution around the predicted margin, sigma from out-of-fold residuals."""

    def __init__(self):
        self.sigma: float | None = None

    def fit(self, pred_margin: np.ndarray, actual_margin: np.ndarray) -> "GaussianMarginDist":
        self.sigma = float(np.std(np.asarray(actual_margin, float) - np.asarray(pred_margin, float)))
        return self

    def prob_win(self, m: np.ndarray) -> np.ndarray:
        return norm.cdf(np.asarray(m, float) / self.sigma)

    def interval(self, m: np.ndarray, level: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
        z = norm.ppf(1 - (1 - level) / 2)
        m = np.asarray(m, float)
        return m - z * self.sigma, m + z * self.sigma


def make_margin_dist(method: str, bandwidth: float):
    if method == "empirical":
        return EmpiricalMarginDist(bandwidth)
    if method == "gaussian":
        return GaussianMarginDist()
    raise ValueError(f"unknown prob_method {method!r}")
