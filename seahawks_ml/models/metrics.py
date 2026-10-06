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
    """Share of non-tie games where the favored side won; p == 0.5 earns half credit. Ties are excluded."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    keep = y != 0.5
    if not keep.any():
        return float("nan")
    p, y = p[keep], y[keep]
    credit = np.where(p == 0.5, 0.5, ((p > 0.5) == (y == 1.0)).astype(float))
    return float(np.mean(credit))


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
