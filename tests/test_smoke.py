"""End-to-end on synthetic data: features -> train -> predict -> log -> site."""

from datetime import timedelta

import polars as pl

from seahawks_ml.features.build import build_features
from seahawks_ml.models.pipeline import ModelConfig
from seahawks_ml.models.store import ProjectConfig, load_model, save_model
from seahawks_ml.models.train import train_production
from seahawks_ml.pipeline.gate import due_run, next_game
from seahawks_ml.pipeline.history import append_record, read_history, runs_done
from seahawks_ml.pipeline.predict import latest_injury_week, make_prediction
from seahawks_ml.site.build import build_site
from seahawks_ml.stadiums import load_stadiums
from tests.synthetic import make_raw


def test_pipeline_end_to_end(tmp_path):
    stadiums = load_stadiums()
    raw = make_raw(seasons=(2009, 2010, 2011, 2012, 2013, 2014), unplayed_last_week=True)
    config = ProjectConfig(model=ModelConfig(inner_folds=3))
    frame = build_features(raw, stadiums, config.rating, config.qb)

    model, meta = train_production(frame, config, now=raw.games["kickoff_utc"].max())
    save_model(model, tmp_path / "model.pkl")
    model = load_model(tmp_path / "model.pkl")

    game = next_game(raw.games, "SEA", raw.games.filter(pl.col("margin").is_null())["kickoff_utc"].min()
                     - timedelta(hours=3))
    history_path = tmp_path / "history.jsonl"
    now = game["kickoff_utc"] - timedelta(hours=3)
    run_type = due_run(now, game["kickoff_utc"], runs_done(read_history(history_path), game["game_id"]))
    assert run_type == "gameday"

    row = frame.filter(pl.col("game_id") == game["game_id"])
    record = make_prediction(row, model, run_type, now, meta["model_version"],
                             latest_injury_week(raw.injuries, game["season"]))
    append_record(record, history_path)
    assert runs_done(read_history(history_path), game["game_id"]) == {"gameday"}

    html = build_site(now, history_path=history_path, out_dir=tmp_path / "site").read_text()
    assert f"{record['p_seahawks'] * 100:.0f}%" in html
