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
