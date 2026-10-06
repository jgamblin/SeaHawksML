# Plan 2 of 3: Models, Walk-Forward Backtest, and Locked Config

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a margin model, convert margins to calibrated win probabilities with nested in-fold fitting, tune by walk-forward backtest against Elo/home/Vegas baselines, and lock the chosen config for CI.

**Architecture:** `models/regressors.py` (ridge, LightGBM) predicts home margin. `models/margin_dist.py` converts margin → P(win) (Gaussian or empirical conditional distribution, chosen by tuning); `models/classifier.py` adds an optional win/loss ensemble and Platt scaling. `models/pipeline.fit_model` fits everything for a set of training seasons, fitting anything that consumes predictions only on inner out-of-fold predictions. `models/backtest.py` runs the outer walk-forward and staged tuning; `models/store.py` persists `models/config.json` and `models/model.pkl`.

**Tech Stack:** Python 3.12, uv, polars, nflreadpy, httpx, scikit-learn, LightGBM, scipy, Jinja2, pytest, GitHub Actions + Pages.

**Spec:** `docs/superpowers/specs/2026-10-06-seahawks-win-model-design.md`

---

**Prerequisite:** Plan 1 complete (`data/features/games.parquet` builds).

## Notes from a dry run on real data (2012–2023 walk-forward, partial weather cache)

| Method | Log-loss |
|---|---|
| Tuned model (ridge α=100, Gaussian conversion) | **0.629** |
| Elo only | 0.633 |
| Home team always | 0.686 |
| Vegas spread (benchmark) | 0.613 |

- The **empirical** margin distribution scored 0.631–0.636, slightly *worse* than the Gaussian conversion (0.629), so `prob_method` is a tuned choice rather than fixed. Both are implemented and tested, and the backtest picks.
- LightGBM did not beat ridge (best 0.630), and the classifier ensemble and Platt scaling didn't help. All stay in the search, so this can change as features improve.
- The full `tune` takes about 2 minutes. `--tune-features` rebuilds features 7 times and adds a few minutes.

## File map

| File | Responsibility |
|---|---|
| `seahawks_ml/models/metrics.py` | Log-loss/Brier/accuracy with ties = 0.5, bootstrap CI, calibration bins |
| `seahawks_ml/models/regressors.py` | `RidgeMargin`, `LGBMMargin`: fit / predict / contributions |
| `seahawks_ml/models/margin_dist.py` | `EmpiricalMarginDist`, `GaussianMarginDist` |
| `seahawks_ml/models/classifier.py` | `WinClassifier`, `PlattScaler` (ties as half win/half loss) |
| `seahawks_ml/models/pipeline.py` | `ModelConfig`, `FittedModel`, nested `fit_model` |
| `seahawks_ml/models/backtest.py` | `walk_forward`, `score`, staged `tune` |
| `seahawks_ml/models/store.py` | `ProjectConfig`, config/model persistence |
| `seahawks_ml/models/train.py` | Production refit with locked config |
| `seahawks_ml/cli.py` | Adds `backtest`, `holdout`, `retrain` |

---

### Task 1: Metrics with explicit tie handling

**Files:**
- Create/replace: `seahawks_ml/models/metrics.py`
- Test: `tests/test_metrics.py`

Spec rule: a tie is an outcome of 0.5 in log-loss and Brier, excluded from accuracy, and margin 0 for MAE.

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:

```python
import math

import numpy as np
import pytest

from seahawks_ml.models.metrics import (
    accuracy,
    bootstrap_ci,
    brier,
    calibration_table,
    log_loss,
    outcome,
)


def test_outcome_scores_ties_as_half():
    assert outcome(np.array([3, -7, 0])).tolist() == [1.0, 0.0, 0.5]


def test_log_loss_with_tie():
    p = np.array([0.8, 0.5])
    y = np.array([0.5, 0.5])
    expected = -(0.5 * math.log(0.8) + 0.5 * math.log(0.2) + math.log(0.5)) / 2
    assert log_loss(p, y) == pytest.approx(expected)


def test_brier_with_tie():
    assert brier(np.array([0.8]), np.array([0.5])) == pytest.approx(0.09)


def test_accuracy_excludes_ties():
    p = np.array([0.7, 0.4, 0.9])
    y = np.array([1.0, 1.0, 0.5])
    assert accuracy(p, y) == 0.5


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.2, 0.8, 500)
    y = (rng.uniform(size=500) < p).astype(float)
    lo, hi = bootstrap_ci(p, y, n_boot=200)
    assert lo < log_loss(p, y) < hi


def test_calibration_table_bins():
    rows = calibration_table(np.array([0.05, 0.15, 0.95]), np.array([0.0, 1.0, 1.0]), bins=10)
    assert [r["n"] for r in rows] == [1, 1, 1]
    assert rows[-1]["bin_low"] == pytest.approx(0.9)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.models.metrics'`

- [ ] **Step 3: Write the implementation**

`seahawks_ml/models/metrics.py`:

```python
"""Scoring with explicit tie handling: a tie counts as an outcome of 0.5."""

import numpy as np

EPS = 1e-12


def outcome(margin: np.ndarray) -> np.ndarray:
    """Home outcome: 1 win, 0 loss, 0.5 tie."""
    m = np.asarray(margin, dtype=float)
    return np.where(m > 0, 1.0, np.where(m < 0, 0.0, 0.5))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    """Share of non-tie games where the favored side won. Ties are excluded."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    keep = y != 0.5
    if not keep.any():
        return float("nan")
    return float(np.mean((p[keep] > 0.5) == (y[keep] == 1.0)))


def margin_mae(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(actual, float))))


def summarize(p: np.ndarray, pred_margin: np.ndarray, actual_margin: np.ndarray) -> dict:
    y = outcome(actual_margin)
    return {
        "n": len(y),
        "log_loss": log_loss(p, y),
        "brier": brier(p, y),
        "accuracy": accuracy(p, y),
        "margin_mae": margin_mae(pred_margin, actual_margin) if pred_margin is not None else None,
    }


def bootstrap_ci(p: np.ndarray, y: np.ndarray, metric=log_loss, n_boot: int = 1000,
                 level: float = 0.9, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    p, y = np.asarray(p, float), np.asarray(y, float)
    stats = [metric(p[idx], y[idx]) for idx in (rng.integers(0, len(p), len(p)) for _ in range(n_boot))]
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            rows.append({"bin_low": float(edges[b]), "bin_high": float(edges[b + 1]),
                         "n": int(mask.sum()), "mean_pred": float(p[mask].mean()),
                         "mean_outcome": float(y[mask].mean())})
    return rows
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add seahawks_ml/models/metrics.py tests/test_metrics.py
git commit -m "feat: metrics with tie handling"
```

---

### Task 2: Margin regressors

**Files:**
- Create/replace: `seahawks_ml/models/regressors.py`
- Test: `tests/test_regressors.py`

Ridge (standardized inputs) and conservative LightGBM behind one interface. `contributions` powers the dashboard's "why" table: ridge gives coefficient × standardized value, LightGBM uses native `pred_contrib`.

- [ ] **Step 1: Write the failing test**

`tests/test_regressors.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_regressors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.models.regressors'`

- [ ] **Step 3: Write the implementation**

`seahawks_ml/models/regressors.py`:

```python
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
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_regressors.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add seahawks_ml/models/regressors.py tests/test_regressors.py
git commit -m "feat: ridge and LightGBM margin regressors"
```

---

### Task 3: Margin → probability distributions

**Files:**
- Create/replace: `seahawks_ml/models/margin_dist.py`
- Test: `tests/test_margin_dist.py`

Empirical: kernel-weight past games by predicted margin and use their actual margins, so key numbers and ties carry through (`P(win) = P(>0) + 0.5·P(=0)`). Gaussian: Φ(m/σ) with σ from out-of-fold residuals. Both give an 80% margin interval.

- [ ] **Step 1: Write the failing test**

`tests/test_margin_dist.py`:

```python
import numpy as np
import pytest

from seahawks_ml.models.margin_dist import EmpiricalMarginDist, GaussianMarginDist, make_margin_dist


def test_margin_dist_keeps_ties_and_key_numbers():
    pred = np.zeros(4)
    actual = np.array([3.0, -3.0, 0.0, 7.0])
    dist = EmpiricalMarginDist(bandwidth=1.0).fit(pred, actual)
    # wins: 3, 7 (2 of 4) plus half of the tie
    assert dist.prob_win(np.array([0.0]))[0] == pytest.approx(0.625)


def test_margin_dist_is_monotonic_and_stable_out_of_range():
    rng = np.random.default_rng(0)
    pred = rng.normal(0, 6, 2000)
    actual = np.round(pred + rng.normal(0, 13, 2000))
    dist = EmpiricalMarginDist(bandwidth=2.0).fit(pred, actual)
    p = dist.prob_win(np.array([-10.0, 0.0, 10.0, 80.0]))
    assert p[0] < p[1] < p[2] and np.isfinite(p).all()
    lo, hi = dist.interval(np.array([0.0, 10.0]), level=0.8)
    assert (lo < hi).all() and lo[1] > lo[0]


def test_gaussian_margin_dist_uses_residual_sigma():
    pred = np.zeros(1000)
    actual = np.random.default_rng(0).normal(0, 13.5, 1000)
    dist = GaussianMarginDist().fit(pred, actual)
    assert dist.sigma == pytest.approx(13.5, rel=0.05)
    assert dist.prob_win(np.array([0.0]))[0] == pytest.approx(0.5)
    lo, hi = dist.interval(np.array([3.0]), level=0.8)
    assert hi[0] - 3.0 == pytest.approx(3.0 - lo[0])


def test_make_margin_dist():
    assert isinstance(make_margin_dist("gaussian", 2.0), GaussianMarginDist)
    assert isinstance(make_margin_dist("empirical", 2.0), EmpiricalMarginDist)
    with pytest.raises(ValueError):
        make_margin_dist("beta", 2.0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_margin_dist.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.models.margin_dist'`

- [ ] **Step 3: Write the implementation**

`seahawks_ml/models/margin_dist.py`:

```python
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
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_margin_dist.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add seahawks_ml/models/margin_dist.py tests/test_margin_dist.py
git commit -m "feat: empirical and Gaussian margin distributions"
```

---

### Task 4: Win classifier and Platt scaling

**Files:**
- Create/replace: `seahawks_ml/models/classifier.py`
- Test: `tests/test_classifier.py`

Both duplicate each game as a half-weighted win and loss when it's a tie, so ties never get dropped or treated as losses.

- [ ] **Step 1: Write the failing test**

`tests/test_classifier.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.models.classifier'`

- [ ] **Step 3: Write the implementation**

`seahawks_ml/models/classifier.py`:

```python
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
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add seahawks_ml/models/classifier.py tests/test_classifier.py
git commit -m "feat: tie-aware win classifier and Platt scaler"
```

---

### Task 5: Model pipeline with nested out-of-fold fitting

**Files:**
- Create/replace: `tests/synthetic.py`
- Create/replace: `seahawks_ml/models/pipeline.py`
- Test: `tests/test_pipeline.py`

`fit_model(frame, config, train_seasons)` touches only `train_seasons`. It runs an inner walk-forward over those seasons to get out-of-fold margins, fits the margin distribution, Platt scaling and Vegas benchmark distribution on them, then fits the final regressor on all training seasons with recency weights. This satisfies the spec's calibration-leakage rule. `make_feature_frame` (appended to `tests/synthetic.py`) is a model-ready frame where `elo_diff` and `qb_epa_diff` drive the margin.

- [ ] **Step 1: Write the test helper**

Replace `tests/synthetic.py` with the version below; it is the Plan 1 file plus `make_feature_frame` at the end.

`tests/synthetic.py`:

```python
"""Small, fully synthetic RawData for fast offline tests.

Four NFC West teams play a 6-week double round robin each season. Team strength,
EPA, QB stats, snaps, injuries and weather are random but internally consistent.
"""

import random
from datetime import UTC, datetime, timedelta

import polars as pl

from seahawks_ml.data import RawData
from seahawks_ml.features.base import GAMES_SCHEMA
from seahawks_ml.ingest.nflverse import (
    INJURIES_SCHEMA,
    PLAYERS_SCHEMA,
    QB_GAMES_SCHEMA,
    SNAPS_SCHEMA,
    TEAM_EPA_SCHEMA,
)
from seahawks_ml.ingest.weather import GAME_WINDOW_HOURS, WEATHER_SCHEMA

TEAMS = ["SEA", "SF", "LA", "ARI"]
HOME_STADIUM = {"SEA": "SEA00", "SF": "SFO01", "LA": "LAX01", "ARI": "PHO00"}
ROOF = {"SEA00": "outdoors", "SFO01": "outdoors", "LAX01": "dome", "PHO00": "closed"}
ROUNDS = [[("SEA", "SF"), ("LA", "ARI")], [("SEA", "LA"), ("SF", "ARI")], [("SEA", "ARI"), ("SF", "LA")]]


def _coach(team: str, season: int, seasons) -> str:
    """SF changes head coach after the first season."""
    return f"{team} Coach B" if (team == "SF" and season >= seasons[1]) else f"{team} Coach A"


def make_raw(seasons=(2011, 2012, 2013, 2014), seed: int = 0, unplayed_last_week: bool = False) -> RawData:
    rng = random.Random(seed)
    strength = {t: rng.gauss(0, 4) for t in TEAMS}
    games, team_epa, qb_games, snaps, injuries, weather = [], [], [], [], [], []
    players = []
    for t in TEAMS:
        players.append({"gsis_id": f"{t}-QB1", "pfr_id": f"{t}QB100", "position": "QB",
                        "draft_round": {"SEA": 3, "SF": 1, "LA": 1, "ARI": None}[t], "rookie_season": 2008})
        players.append({"gsis_id": f"{t}-QB2", "pfr_id": f"{t}QB200", "position": "QB",
                        "draft_round": 6, "rookie_season": 2012})
        for i in range(10):
            players.append({"gsis_id": f"{t}-P{i}", "pfr_id": f"{t}P{i:03d}", "position": "WR" if i < 5 else "LB",
                            "draft_round": 2, "rookie_season": 2010})
    last_season = max(seasons)
    for season in seasons:
        for week in range(1, 7):
            pairs = ROUNDS[(week - 1) % 3]
            kickoff = datetime(season, 9, 8, 17, 0, tzinfo=UTC) + timedelta(days=7 * (week - 1))
            for a, b in pairs:
                home, away = (a, b) if week <= 3 else (b, a)
                game_id = f"{season}_{week:02d}_{away}_{home}"
                stadium = HOME_STADIUM[home]
                unplayed = unplayed_last_week and season == last_season and week == 6
                exp = strength[home] - strength[away] + 2
                margin = None if unplayed else round(rng.gauss(exp, 13))
                home_score = None if unplayed else 20 + max(margin, 0)
                away_score = None if unplayed else 20 + max(-margin, 0)
                home_qb = f"{home}-QB2" if (home == "SF" and season == last_season and week >= 4) else f"{home}-QB1"
                away_qb = f"{away}-QB1"
                games.append({
                    "game_id": game_id, "season": season, "week": week, "game_type": "REG",
                    "kickoff_utc": kickoff, "home_team": home, "away_team": away,
                    "home_score": home_score, "away_score": away_score, "margin": margin,
                    "neutral": False, "roof": ROOF[stadium], "stadium_id": stadium,
                    "home_rest": 7, "away_rest": 7 if week > 1 else 10, "div_game": True,
                    "home_qb_id": home_qb, "away_qb_id": away_qb,
                    "home_coach": _coach(home, season, seasons), "away_coach": _coach(away, season, seasons),
                    "spread_line": round(exp * 2) / 2,
                })
                if unplayed:
                    continue
                for team, opp, qb, sign in ((home, away, home_qb, 1), (away, home, away_qb, -1)):
                    epa = sign * margin / 30 + rng.gauss(0, 3)
                    team_epa.append({"game_id": game_id, "season": season, "team": team,
                                     "opponent": opp, "epa_sum": epa, "plays": 60})
                    qb_games.append({"game_id": game_id, "season": season, "team": team, "qb_id": qb,
                                     "dropbacks": 35, "qb_epa_sum": epa * 0.8,
                                     "cpoe_sum": rng.gauss(0, 30), "cpoe_n": 30})
                    if season >= 2013:
                        for i in range(10):
                            snaps.append({"game_id": game_id, "season": season, "week": week,
                                          "team": team, "pfr_player_id": f"{team}P{i:03d}",
                                          "position": "WR" if i < 5 else "LB",
                                          "offense_pct": 0.9 if i < 5 else 0.0,
                                          "defense_pct": 0.0 if i < 5 else 0.9})
                if ROOF[stadium] == "outdoors":
                    for h in range(GAME_WINDOW_HOURS):
                        weather.append({"stadium_id": stadium, "time_utc": kickoff + timedelta(hours=h),
                                        "temp_f": 60.0 - week, "wind_mph": 5.0 + h, "precip_in": 0.0,
                                        "source": "archive"})
            for t in TEAMS:
                if rng.random() < 0.5:
                    injuries.append({"season": season, "week": week, "team": t,
                                     "gsis_id": f"{t}-P{rng.randrange(10)}", "position": "WR",
                                     "report_status": rng.choice(["Out", "Doubtful", "Questionable"])})
    return RawData(
        games=pl.DataFrame(games, schema=GAMES_SCHEMA).sort("kickoff_utc", "game_id"),
        team_epa=pl.DataFrame(team_epa, schema=TEAM_EPA_SCHEMA),
        qb_games=pl.DataFrame(qb_games, schema=QB_GAMES_SCHEMA),
        injuries=pl.DataFrame(injuries, schema=INJURIES_SCHEMA),
        snaps=pl.DataFrame(snaps, schema=SNAPS_SCHEMA),
        players=pl.DataFrame(players, schema=PLAYERS_SCHEMA),
        weather=pl.DataFrame(weather, schema=WEATHER_SCHEMA).unique(["stadium_id", "time_utc"]),
    )


def make_feature_frame(seasons=range(2009, 2016), games_per_season=120, seed=0) -> pl.DataFrame:
    """Model-ready feature frame where elo_diff and qb_epa_diff drive the margin."""
    import numpy as np

    from seahawks_ml.features.columns import FEATURE_COLUMNS

    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for i in range(games_per_season):
            feats = {c: 0.0 for c in FEATURE_COLUMNS}
            feats["elo_diff"] = rng.normal(0, 80)
            feats["qb_epa_diff"] = rng.normal(0, 0.1)
            feats["home_field"] = 1.0
            margin = round(feats["elo_diff"] / 25 + 30 * feats["qb_epa_diff"] + 1.5 + rng.normal(0, 13))
            rows.append({"game_id": f"{season}_{i:03d}", "season": season, "margin": margin,
                         "home_team": "SEA" if i % 8 == 0 else "SF", "away_team": "LA",
                         "neutral": False, "elo_home_pre": 1500 + feats["elo_diff"],
                         "elo_away_pre": 1500.0,
                         "spread_line": round(2 * (feats["elo_diff"] / 25 + 1.5)) / 2, **feats})
    return pl.DataFrame(rows)
```

- [ ] **Step 2: Write the failing test**

`tests/test_pipeline.py`:

```python
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
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.models.pipeline'`

- [ ] **Step 4: Write the implementation**

`seahawks_ml/models/pipeline.py`:

```python
"""Fit a complete model for a set of training seasons, with nested out-of-fold fitting.

Everything fit on *predictions* (the empirical margin distribution, the classifier
ensemble weight's inputs, Platt scaling) only ever sees out-of-fold predictions from an
inner walk-forward over the training seasons. Nothing here touches seasons outside
`train_seasons`, so calling fit_model inside an outer walk-forward fold is leak-free.
"""

from dataclasses import asdict, dataclass, field

import numpy as np
import polars as pl

from seahawks_ml.features.columns import FEATURE_COLUMNS
from seahawks_ml.models.classifier import PlattScaler, WinClassifier
from seahawks_ml.models.margin_dist import make_margin_dist
from seahawks_ml.models.regressors import make_regressor

MIN_INNER_TRAIN_SEASONS = 2


@dataclass
class ModelConfig:
    kind: str = "ridge"
    params: dict = field(default_factory=lambda: {"alpha": 100.0})
    half_life: float = 3.0  # seasons; recency weight = 0.5 ** (age / half_life)
    prob_method: str = "gaussian"  # "gaussian" or "empirical" (see margin_dist.py)
    bandwidth: float = 2.0  # empirical kernel width in points
    ensemble_weight: float = 0.0  # weight on the binary classifier (0 = margin model only)
    classifier_c: float = 0.1
    use_platt: bool = False
    inner_folds: int = 20  # most recent training seasons used for out-of-fold predictions

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**d)


def recency_weights(seasons: np.ndarray, ref_season: int, half_life: float) -> np.ndarray:
    return 0.5 ** ((ref_season - np.asarray(seasons, float)) / half_life)


def _xy(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = frame.select(FEATURE_COLUMNS).to_numpy().astype(float)
    return X, frame["margin"].to_numpy().astype(float), frame["season"].to_numpy()


@dataclass
class FittedModel:
    config: ModelConfig
    regressor: object
    margin_dist: object  # EmpiricalMarginDist or GaussianMarginDist
    vegas_dist: object
    classifier: WinClassifier | None
    platt: PlattScaler | None
    train_seasons: list[int]

    def predict(self, frame: pl.DataFrame) -> dict[str, np.ndarray]:
        X = frame.select(FEATURE_COLUMNS).to_numpy().astype(float)
        margin = self.regressor.predict(X)
        p = self.margin_dist.prob_win(margin)
        if self.classifier is not None:
            w = self.config.ensemble_weight
            p = (1 - w) * p + w * self.classifier.predict_proba(X)
        if self.platt is not None:
            p = self.platt.transform(p)
        lo, hi = self.margin_dist.interval(margin, level=0.8)
        return {"margin": margin, "p_win": p, "margin_lo": lo, "margin_hi": hi}

    def contributions(self, frame: pl.DataFrame) -> np.ndarray:
        return self.regressor.contributions(frame.select(FEATURE_COLUMNS).to_numpy().astype(float))

    def vegas_prob(self, spread_line: np.ndarray) -> np.ndarray:
        """Home win probability implied by the spread (benchmark only)."""
        return self.vegas_dist.prob_win(np.asarray(spread_line, float))


def _fit_parts(frame: pl.DataFrame, config: ModelConfig, ref_season: int):
    X, y, seasons = _xy(frame)
    w = recency_weights(seasons, ref_season, config.half_life)
    reg = make_regressor(config.kind, dict(config.params)).fit(X, y, w)
    clf = WinClassifier(config.classifier_c).fit(X, y, w) if config.ensemble_weight > 0 else None
    return reg, clf


def _out_of_fold(frame: pl.DataFrame, config: ModelConfig, seasons: list[int]):
    """Inner walk-forward: predictions for each of the last `inner_folds` seasons."""
    inner = seasons[MIN_INNER_TRAIN_SEASONS:][-config.inner_folds:]
    margins, probs, actual = [], [], []
    for s in inner:
        fit_rows = frame.filter(pl.col("season") < s)
        pred_rows = frame.filter(pl.col("season") == s)
        reg, clf = _fit_parts(fit_rows, config, ref_season=s - 1)
        X, y, _ = _xy(pred_rows)
        margins.append(reg.predict(X))
        probs.append(clf.predict_proba(X) if clf else np.zeros(len(y)))
        actual.append(y)
    if not inner:
        raise ValueError(f"need more than {MIN_INNER_TRAIN_SEASONS} training seasons, got {seasons}")
    return np.concatenate(margins), np.concatenate(probs), np.concatenate(actual)


def fit_model(frame: pl.DataFrame, config: ModelConfig, train_seasons: list[int]) -> FittedModel:
    seasons = sorted(set(train_seasons))
    train = frame.filter(pl.col("season").is_in(seasons) & pl.col("margin").is_not_null())

    oof_margin, oof_class, oof_actual = _out_of_fold(train, config, seasons)
    margin_dist = make_margin_dist(config.prob_method, config.bandwidth).fit(oof_margin, oof_actual)
    platt = None
    if config.use_platt:
        p = margin_dist.prob_win(oof_margin)
        if config.ensemble_weight > 0:
            p = (1 - config.ensemble_weight) * p + config.ensemble_weight * oof_class
        platt = PlattScaler().fit(p, oof_actual)

    with_spread = train.filter(pl.col("spread_line").is_not_null())
    vegas_dist = make_margin_dist(config.prob_method, config.bandwidth).fit(
        with_spread["spread_line"].to_numpy(), with_spread["margin"].to_numpy())

    reg, clf = _fit_parts(train, config, ref_season=seasons[-1])
    return FittedModel(config, reg, margin_dist, vegas_dist, clf, platt, seasons)
```

- [ ] **Step 5: Run it to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/synthetic.py seahawks_ml/models/pipeline.py tests/test_pipeline.py
git commit -m "feat: model pipeline with nested out-of-fold calibration"
```

---

### Task 6: Walk-forward backtest and staged tuning

**Files:**
- Create/replace: `seahawks_ml/models/backtest.py`
- Test: `tests/test_backtest.py`

Outer walk-forward: for each season Y, fit on 2009..Y−1 and predict Y. Baselines are scored on the same games (rows with a spread). Tuning is staged: (1) model family × hyperparameters × recency half-life, with ridge kept unless LightGBM wins by more than 0.002; (2) Gaussian vs. empirical; (3) classifier ensemble weight; (4) Platt on/off.

- [ ] **Step 1: Write the failing test**

`tests/test_backtest.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_backtest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.models.backtest'`

- [ ] **Step 3: Write the implementation**

`seahawks_ml/models/backtest.py`:

```python
"""Walk-forward backtest, staged tuning, model selection and the one-time holdout."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from seahawks_ml.config import FIRST_TRAIN_SEASON, TEAM
from seahawks_ml.features.elo import elo_win_prob
from seahawks_ml.models.metrics import bootstrap_ci, calibration_table, outcome, summarize
from seahawks_ml.models.pipeline import ModelConfig, fit_model

LINEAR_PREFERENCE = 0.002  # pick ridge unless LightGBM beats it by more than this log-loss

RIDGE_GRID = [{"alpha": a} for a in (10.0, 100.0, 300.0, 1000.0)]
LGBM_GRID = [
    {"num_leaves": 4, "n_estimators": 200, "learning_rate": 0.03, "min_child_samples": 50, "reg_lambda": 5.0},
    {"num_leaves": 8, "n_estimators": 300, "learning_rate": 0.02, "min_child_samples": 80, "reg_lambda": 10.0},
]
HALF_LIFE_GRID = (3.0, 6.0, 12.0)
PROB_METHOD_GRID = [("gaussian", 2.0), ("empirical", 1.0), ("empirical", 2.0), ("empirical", 4.0)]
ENSEMBLE_GRID = (0.0, 0.3, 0.5)


def walk_forward(frame: pl.DataFrame, config: ModelConfig, test_seasons) -> pl.DataFrame:
    """For each season Y: fit on [FIRST_TRAIN_SEASON, Y), predict completed games of Y."""
    parts = []
    for season in test_seasons:
        train_seasons = list(range(FIRST_TRAIN_SEASON, season))
        model = fit_model(frame, config, train_seasons)
        test = frame.filter((pl.col("season") == season) & pl.col("margin").is_not_null())
        pred = model.predict(test)
        train = frame.filter(pl.col("season").is_in(train_seasons) & (pl.col("neutral").not_())
                             & pl.col("margin").is_not_null())
        home_rate = float(outcome(train["margin"].to_numpy()).mean())
        elo_diff = (test["elo_home_pre"] - test["elo_away_pre"]).to_numpy()
        spread = test["spread_line"].to_numpy().astype(float)
        p_vegas = np.where(np.isnan(spread), np.nan, model.vegas_prob(np.nan_to_num(spread)))
        parts.append(test.select("game_id", "season", "home_team", "away_team", "margin").with_columns(
            pl.Series("pred_margin", pred["margin"]),
            pl.Series("p_model", pred["p_win"]),
            pl.Series("p_elo", [elo_win_prob(d, n) for d, n in zip(elo_diff, test["neutral"].to_list())]),
            pl.Series("p_home", np.where(test["neutral"].to_numpy(), 0.5, home_rate)),
            pl.Series("p_vegas", p_vegas),
        ))
    return pl.concat(parts)


def score(preds: pl.DataFrame) -> dict:
    """Metrics for the model and each baseline, on games where every method has a prediction."""
    common = preds.filter(pl.col("p_vegas").is_not_nan())
    y_all = outcome(common["margin"].to_numpy())
    result = {}
    for name in ("p_model", "p_elo", "p_home", "p_vegas"):
        p = common[name].to_numpy()
        pred_margin = common["pred_margin"].to_numpy() if name == "p_model" else None
        stats = summarize(p, pred_margin, common["margin"].to_numpy())
        stats["log_loss_ci90"] = bootstrap_ci(p, y_all, n_boot=300)
        stats["calibration"] = calibration_table(p, y_all)
        result[name.removeprefix("p_")] = stats
    sea = common.filter((pl.col("home_team") == TEAM) | (pl.col("away_team") == TEAM))
    if sea.height:
        p_sea = np.where(sea["home_team"].to_numpy() == TEAM, sea["p_model"].to_numpy(), 1 - sea["p_model"].to_numpy())
        m_sea = np.where(sea["home_team"].to_numpy() == TEAM, sea["margin"].to_numpy(), -sea["margin"].to_numpy())
        result["seahawks_only"] = summarize(p_sea, None, m_sea)
    result["per_season"] = [
        {"season": int(s), **summarize(g["p_model"].to_numpy(), g["pred_margin"].to_numpy(), g["margin"].to_numpy())}
        for (s,), g in common.group_by(["season"], maintain_order=True)
    ]
    return result


def walk_forward_log_loss(frame: pl.DataFrame, config: ModelConfig, seasons) -> float:
    from seahawks_ml.models.metrics import log_loss
    preds = walk_forward(frame, config, seasons)
    return log_loss(preds["p_model"].to_numpy(), outcome(preds["margin"].to_numpy()))


def tune(frame: pl.DataFrame, seasons, log=print) -> tuple[ModelConfig, list[dict]]:
    """Staged (coordinate-descent) search. Every evaluation is a full walk-forward."""
    trials: list[dict] = []

    def evaluate(cfg: ModelConfig) -> float:
        ll = walk_forward_log_loss(frame, cfg, seasons)
        trials.append({"config": cfg.to_dict(), "log_loss": ll})
        log(f"  {ll:.5f}  {cfg.kind} {cfg.params} hl={cfg.half_life} {cfg.prob_method} bw={cfg.bandwidth} "
            f"ens={cfg.ensemble_weight} platt={cfg.use_platt}")
        return ll

    log("stage 1: model family, hyperparameters, recency half-life")
    best_by_kind = {}
    for kind, grid in (("ridge", RIDGE_GRID), ("lgbm", LGBM_GRID)):
        for params in grid:
            for hl in HALF_LIFE_GRID:
                cfg = ModelConfig(kind=kind, params=params, half_life=hl)
                ll = evaluate(cfg)
                if kind not in best_by_kind or ll < best_by_kind[kind][1]:
                    best_by_kind[kind] = (cfg, ll)
    ridge, lgbm = best_by_kind["ridge"], best_by_kind["lgbm"]
    best, best_ll = lgbm if lgbm[1] < ridge[1] - LINEAR_PREFERENCE else ridge

    log("stage 2: margin -> probability method")
    for method, bw in PROB_METHOD_GRID:
        if (best.prob_method, best.bandwidth) == (method, bw):
            continue
        cfg = replace(best, prob_method=method, bandwidth=bw)
        ll = evaluate(cfg)
        if ll < best_ll:
            best, best_ll = cfg, ll

    for stage, field, values in (("3: classifier ensemble", "ensemble_weight", ENSEMBLE_GRID),
                                 ("4: Platt scaling", "use_platt", (False, True))):
        log(f"stage {stage}")
        for v in values:
            if getattr(best, field) == v:
                continue
            cfg = replace(best, **{field: v})
            ll = evaluate(cfg)
            if ll < best_ll:
                best, best_ll = cfg, ll
    return best, trials


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_backtest.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add seahawks_ml/models/backtest.py tests/test_backtest.py
git commit -m "feat: walk-forward backtest, baselines, staged tuning"
```

---

### Task 7: Config/model store and production training

**Files:**
- Create/replace: `seahawks_ml/models/store.py`
- Create/replace: `seahawks_ml/models/train.py`
- Test: `tests/test_store_train.py`

`ProjectConfig` bundles model config with the feature shrinkage parameters. Its fingerprint goes into the model version (`YYYYMMDD-<hash>`) so every logged prediction names the exact config.

- [ ] **Step 1: Write the failing test**

`tests/test_store_train.py`:

```python
from datetime import UTC, datetime

import polars as pl

from seahawks_ml.features.ratings import RatingParams
from seahawks_ml.models.pipeline import ModelConfig
from seahawks_ml.models.store import ProjectConfig, load_config, load_model, save_config, save_model
from seahawks_ml.models.train import completed_seasons, train_production
from tests.synthetic import make_feature_frame


def test_config_round_trip(tmp_path):
    cfg = ProjectConfig(model=ModelConfig(kind="lgbm", params={"num_leaves": 4}), rating=RatingParams(0.5, 3, 1))
    save_config(cfg, tmp_path / "c.json")
    loaded = load_config(tmp_path / "c.json")
    assert loaded == cfg and loaded.fingerprint() == cfg.fingerprint()


def test_completed_seasons_skips_unplayed():
    frame = make_feature_frame(seasons=range(2009, 2013)).with_columns(
        pl.when(pl.col("season") == 2012).then(None).otherwise(pl.col("margin")).alias("margin"))
    assert completed_seasons(frame) == [2009, 2010, 2011]


def test_train_production_and_model_round_trip(tmp_path):
    frame = make_feature_frame()
    now = datetime(2026, 10, 6, tzinfo=UTC)
    model, meta = train_production(frame, ProjectConfig(), now)
    assert meta["train_seasons"] == list(range(2009, 2016))
    assert meta["model_version"].startswith("20261006-")
    save_model(model, tmp_path / "m.pkl")
    again = load_model(tmp_path / "m.pkl")
    head = frame.head(5)
    assert (again.predict(head)["p_win"] == model.predict(head)["p_win"]).all()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_store_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.models.store'`

- [ ] **Step 3: Write the implementation**

`seahawks_ml/models/store.py`:

```python
"""Locked project config (models/config.json) and the trained model artifact."""

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

from seahawks_ml.config import MODELS_DIR
from seahawks_ml.features.qb import QBParams
from seahawks_ml.features.ratings import RatingParams
from seahawks_ml.models.pipeline import FittedModel, ModelConfig

CONFIG_PATH = MODELS_DIR / "config.json"
MODEL_PATH = MODELS_DIR / "model.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
BACKTEST_PATH = MODELS_DIR / "backtest.json"
HOLDOUT_PATH = MODELS_DIR / "holdout.json"


@dataclass
class ProjectConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    rating: RatingParams = field(default_factory=RatingParams)
    qb: QBParams = field(default_factory=QBParams)

    def to_dict(self) -> dict:
        return {"model": self.model.to_dict(), "rating": asdict(self.rating), "qb": asdict(self.qb)}

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectConfig":
        return cls(ModelConfig.from_dict(d["model"]), RatingParams(**d["rating"]), QBParams(**d["qb"]))

    def fingerprint(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:8]


def save_config(config: ProjectConfig, path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2))


def load_config(path: Path = CONFIG_PATH) -> ProjectConfig:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m seahawks_ml.cli backtest` locally first")
    return ProjectConfig.from_dict(json.loads(path.read_text()))


def save_model(model: FittedModel, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)


def load_model(path: Path = MODEL_PATH) -> FittedModel:
    with path.open("rb") as f:
        return pickle.load(f)
```

`seahawks_ml/models/train.py`:

```python
"""Production refit with locked hyperparameters (what CI runs - no tuning)."""

from datetime import datetime

import polars as pl

from seahawks_ml.config import FIRST_TRAIN_SEASON
from seahawks_ml.models.pipeline import FittedModel, fit_model
from seahawks_ml.models.store import ProjectConfig


def completed_seasons(frame: pl.DataFrame) -> list[int]:
    done = frame.filter(pl.col("margin").is_not_null() & (pl.col("season") >= FIRST_TRAIN_SEASON))
    return sorted(done["season"].unique().to_list())


def train_production(frame: pl.DataFrame, config: ProjectConfig, now: datetime) -> tuple[FittedModel, dict]:
    seasons = completed_seasons(frame)
    model = fit_model(frame, config.model, seasons)
    n_games = frame.filter(pl.col("season").is_in(seasons) & pl.col("margin").is_not_null()).height
    meta = {
        "model_version": f"{now:%Y%m%d}-{config.fingerprint()}",
        "trained_at": now.isoformat(),
        "train_seasons": seasons,
        "n_games": n_games,
        "config": config.to_dict(),
    }
    return model, meta
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_store_train.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add seahawks_ml/models/store.py seahawks_ml/models/train.py tests/test_store_train.py
git commit -m "feat: locked config store and production training"
```

---

### Task 8: `backtest`, `holdout`, `retrain` commands and the locked config

**Files:**
- Create/replace: `seahawks_ml/cli.py`

Replaces `seahawks_ml/cli.py`. `features` now uses the locked config when present. `backtest` (local only) optionally tunes rating/QB shrinkage, then the model, and writes `models/config.json` + `models/backtest.json`. `holdout` refuses to run twice. `retrain` is what CI runs.

- [ ] **Step 1: Write the file**

`seahawks_ml/cli.py`:

```python
"""Command-line entry point: python -m seahawks_ml.cli <command>.

Local (offseason / manual):  features, backtest, holdout
CI (in season):              retrain
"""

import argparse
from dataclasses import replace
from datetime import UTC, datetime

import polars as pl

from seahawks_ml.config import BACKTEST_SEASONS, FEATURES_PATH, HOLDOUT_SEASONS

RATING_GRID = [(0.5, 4.0, 2.0), (0.6, 4.0, 2.0), (0.7, 6.0, 3.0), (0.6, 6.0, 2.0)]
QB_PRIOR_GRID = [150.0, 250.0, 400.0]


def _now(args) -> datetime:
    return datetime.fromisoformat(args.now) if getattr(args, "now", None) else datetime.now(UTC)


def _load(now: datetime):
    from seahawks_ml.data import load_raw
    from seahawks_ml.stadiums import load_stadiums

    stadiums = load_stadiums()
    return stadiums, load_raw(stadiums, now)


def _build(raw, stadiums, config) -> pl.DataFrame:
    from seahawks_ml.features.build import build_features

    return build_features(raw, stadiums, config.rating, config.qb)


def cmd_features(args) -> None:
    from seahawks_ml.models.store import ProjectConfig, load_config

    stadiums, raw = _load(_now(args))
    try:
        config = load_config()
    except FileNotFoundError:
        config = ProjectConfig()
    frame = _build(raw, stadiums, config)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(FEATURES_PATH)
    print(f"wrote {frame.height} games to {FEATURES_PATH}")


def cmd_backtest(args) -> None:
    """Local only: tune feature params + model config by walk-forward, lock models/config.json."""
    from seahawks_ml.features.qb import QBParams
    from seahawks_ml.features.ratings import RatingParams
    from seahawks_ml.models.backtest import score, tune, walk_forward, walk_forward_log_loss, write_json
    from seahawks_ml.models.pipeline import ModelConfig
    from seahawks_ml.models.store import BACKTEST_PATH, ProjectConfig, save_config

    stadiums, raw = _load(_now(args))
    seasons = list(BACKTEST_SEASONS)
    config = ProjectConfig()
    if args.tune_features:
        print("stage 0a: team-rating shrinkage")
        best = None
        for reg, k, k_new in RATING_GRID:
            cand = replace(config, rating=RatingParams(reg, k, k_new))
            ll = walk_forward_log_loss(_build(raw, stadiums, cand), ModelConfig(), seasons)
            print(f"  {ll:.5f}  {cand.rating}")
            if best is None or ll < best[1]:
                best = (cand, ll)
        config = best[0]
        print("stage 0b: QB prior strength")
        for prior in QB_PRIOR_GRID:
            cand = replace(config, qb=QBParams(prior_dropbacks=prior))
            ll = walk_forward_log_loss(_build(raw, stadiums, cand), ModelConfig(), seasons)
            print(f"  {ll:.5f}  {cand.qb}")
            if ll < best[1]:
                best = (cand, ll)
        config = best[0]
    frame = _build(raw, stadiums, config)
    model_config, trials = tune(frame, seasons)
    config = replace(config, model=model_config)
    save_config(config)
    preds = walk_forward(frame, model_config, seasons)
    write_json(BACKTEST_PATH, {"seasons": seasons, "config": config.to_dict(),
                               "score": score(preds), "trials": trials})
    print(f"locked config {config.fingerprint()} -> models/config.json; report -> {BACKTEST_PATH}")


def cmd_holdout(args) -> None:
    """Local only, run once after tuning: score the locked config on 2024-2025."""
    from seahawks_ml.models.backtest import score, walk_forward, write_json
    from seahawks_ml.models.store import HOLDOUT_PATH, load_config

    if HOLDOUT_PATH.exists() and not args.force:
        raise SystemExit(f"{HOLDOUT_PATH} exists - the holdout is evaluated once. Use --force to overwrite.")
    config = load_config()
    stadiums, raw = _load(_now(args))
    frame = _build(raw, stadiums, config)
    preds = walk_forward(frame, config.model, list(HOLDOUT_SEASONS))
    write_json(HOLDOUT_PATH, {"seasons": list(HOLDOUT_SEASONS), "config": config.to_dict(), "score": score(preds)})
    print(f"holdout written to {HOLDOUT_PATH}")


def cmd_retrain(args) -> None:
    """CI: refit final weights with the locked config. No tuning."""
    from seahawks_ml.models.backtest import write_json
    from seahawks_ml.models.store import METRICS_PATH, load_config, save_model
    from seahawks_ml.models.train import train_production

    now = _now(args)
    config = load_config()
    stadiums, raw = _load(now)
    frame = _build(raw, stadiums, config)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(FEATURES_PATH)
    model, meta = train_production(frame, config, now)
    save_model(model)
    write_json(METRICS_PATH, meta)
    print(f"trained {meta['model_version']} on {meta['n_games']} games")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="seahawks_ml")
    parser.add_argument("--now", help="override current time (ISO 8601, for testing)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("features", help="refresh data and rebuild the feature table").set_defaults(func=cmd_features)
    bt = sub.add_parser("backtest", help="LOCAL: tune by walk-forward and lock models/config.json")
    bt.add_argument("--tune-features", action="store_true", help="also tune rating/QB shrinkage (slow)")
    bt.set_defaults(func=cmd_backtest)
    ho = sub.add_parser("holdout", help="LOCAL: evaluate locked config on 2024-2025 (once)")
    ho.add_argument("--force", action="store_true")
    ho.set_defaults(func=cmd_holdout)
    sub.add_parser("retrain", help="CI: refit with locked config").set_defaults(func=cmd_retrain)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 3: Tune locally and lock the config**

```bash
uv run python -m seahawks_ml.cli backtest --tune-features
```
Expected: staged log lines (`stage 0a` … `stage 4`), then `locked config <hash> -> models/config.json`. Open `models/backtest.json` and check `score.model.log_loss` is below `score.elo.log_loss` (dry run: 0.629 vs 0.633). If it isn't, stop and investigate features before continuing.

- [ ] **Step 4: Evaluate the holdout exactly once**

```bash
uv run python -m seahawks_ml.cli holdout
```
Expected: `holdout written to .../models/holdout.json`. Don't change features or config based on this number. If you do change them later, say so in the commit message, because the holdout is no longer clean.

- [ ] **Step 5: Fit the production model**

```bash
uv run python -m seahawks_ml.cli retrain
```
Expected: `trained <YYYYMMDD-hash> on <n> games`.

- [ ] **Step 6: Commit**

```bash
git add seahawks_ml/cli.py models/ data/cache data/features
git commit -m "feat: backtest/holdout/retrain commands; lock tuned config"
```

---
