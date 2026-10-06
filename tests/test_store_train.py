from datetime import UTC, datetime

import polars as pl
import pytest

from seahawks_ml.features.ratings import RatingParams
from seahawks_ml.models.pipeline import ModelConfig
from seahawks_ml.models.store import (
    ProjectConfig,
    check_model_matches,
    load_config,
    load_model,
    save_config,
    save_model,
)
from seahawks_ml.models.train import completed_seasons, train_production
from tests.synthetic import make_feature_frame


def test_config_round_trip(tmp_path):
    cfg = ProjectConfig(model=ModelConfig(kind="lgbm", params={"num_leaves": 4}), rating=RatingParams(0.5, 3, 1))
    save_config(cfg, tmp_path / "c.json")
    loaded = load_config(tmp_path / "c.json")
    assert loaded == cfg and loaded.fingerprint() == cfg.fingerprint()


def test_config_round_trip_with_rating_toggles(tmp_path):
    cfg = ProjectConfig(rating=RatingParams(0.6, 6.0, 2.0, opponent_adjust=True, extra_stats=False))
    save_config(cfg, tmp_path / "c.json")
    assert load_config(tmp_path / "c.json") == cfg


def test_old_config_without_rating_toggles_loads_off(tmp_path):
    import json

    d = ProjectConfig().to_dict()
    d["rating"] = {"prior_regression": 0.6, "prior_games": 6.0, "prior_games_new_coach": 2.0}
    (tmp_path / "c.json").write_text(json.dumps(d))
    loaded = load_config(tmp_path / "c.json")
    # an old config means what it meant: toggles that did not exist are off
    assert loaded.rating == RatingParams(0.6, 6.0, 2.0, opponent_adjust=False, extra_stats=False)
    assert loaded.rating.opponent_adjust is False
    assert loaded.rating.extra_stats is False


def test_config_round_trip_with_availability(tmp_path):
    from seahawks_ml.features.availability import AvailabilityParams

    avail = AvailabilityParams(mode="values", prior_opps=30.0, quality_scale=6.0,
                               lineman_quality={"round_1": 1.5, "day_2": 1.2, "later": 1.0, "udfa": 0.8},
                               durability_weight=0.25)
    cfg = ProjectConfig(availability=avail)
    save_config(cfg, tmp_path / "c.json")
    loaded = load_config(tmp_path / "c.json")
    assert loaded == cfg and loaded.availability == avail
    assert loaded.fingerprint() == cfg.fingerprint() != ProjectConfig().fingerprint()


def test_old_config_without_availability_means_count(tmp_path):
    import json

    d = ProjectConfig().to_dict()
    del d["availability"]
    (tmp_path / "c.json").write_text(json.dumps(d))
    assert load_config(tmp_path / "c.json").availability.mode == "count"


def test_saved_config_lineman_quality_stays_a_json_object(tmp_path):
    import json

    save_config(ProjectConfig(), tmp_path / "c.json")
    saved = json.loads((tmp_path / "c.json").read_text())
    assert saved["availability"]["lineman_quality"] == {"round_1": 1.3, "day_2": 1.1, "later": 1.0, "udfa": 0.9}
    # a trained model's metadata (read back from JSON) still matches the config
    assert saved == ProjectConfig().to_dict()


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


def test_check_model_matches():
    frame = make_feature_frame()
    cfg = ProjectConfig()
    _, meta = train_production(frame, cfg, datetime(2026, 10, 6, tzinfo=UTC))
    check_model_matches(meta, cfg)  # no error
    other = ProjectConfig(rating=RatingParams(0.5, 3, 1))
    with pytest.raises(SystemExit, match="stale"):
        check_model_matches(meta, other)
    with pytest.raises(SystemExit, match="stale"):
        check_model_matches({**meta, "feature_columns": meta["feature_columns"][:-1]}, cfg)
    with pytest.raises(SystemExit, match="stale"):
        check_model_matches({k: v for k, v in meta.items() if k != "feature_columns"}, cfg)
