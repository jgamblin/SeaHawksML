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
