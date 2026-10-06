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
