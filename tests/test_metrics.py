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
