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


def test_gaussian_sigma_is_rmse_including_bias():
    dist = GaussianMarginDist().fit(np.zeros(4), np.array([3.0, 3.0, 3.0, 3.0]))
    assert dist.sigma == pytest.approx(3.0)  # std would give 0
