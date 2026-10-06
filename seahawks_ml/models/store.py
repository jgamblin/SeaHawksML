"""Locked project config (models/config.json) and the trained model artifact."""

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

from seahawks_ml.config import MODELS_DIR
from seahawks_ml.features.columns import FEATURE_COLUMNS
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
        # toggles added later default to off so an old config means what it meant
        rating = {"opponent_adjust": False, "extra_stats": False, **d["rating"]}
        return cls(ModelConfig.from_dict(d["model"]), RatingParams(**rating), QBParams(**d["qb"]))

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


def check_model_matches(meta: dict, config: ProjectConfig) -> None:
    """Raise SystemExit when the trained model was built from a different config or feature set."""
    if meta.get("config") != config.to_dict() or meta.get("feature_columns") != FEATURE_COLUMNS:
        raise SystemExit("model.pkl is stale: run `python -m seahawks_ml.cli retrain`")
