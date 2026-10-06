import polars as pl
import pytest

from seahawks_ml.features.availability import compute_availability
from seahawks_ml.ingest.nflverse import INJURIES_SCHEMA
from tests.synthetic import make_raw


def test_availability_is_null_before_snap_era():
    raw = make_raw()
    out = compute_availability(raw.games, raw.snaps, raw.injuries, raw.players)
    early = out.join(raw.games.select("game_id", "season"), on="game_id").filter(pl.col("season") < 2013)
    assert early["home_off_out"].null_count() == early.height


def test_availability_weights_out_starters_by_snap_share():
    raw = make_raw()
    game = raw.games.filter((pl.col("season") == 2014) & (pl.col("week") == 5)).row(0, named=True)
    team = game["home_team"]
    base = {"season": 2014, "week": 5, "team": team}
    injuries = pl.DataFrame([
        base | {"gsis_id": f"{team}-P0", "position": "WR", "report_status": "Out"},
        base | {"gsis_id": f"{team}-P7", "position": "LB", "report_status": "Doubtful"},
        base | {"gsis_id": f"{team}-P8", "position": "LB", "report_status": "Questionable"},
    ], schema=INJURIES_SCHEMA)
    out = compute_availability(raw.games, raw.snaps, injuries, raw.players)
    row = out.filter(pl.col("game_id") == game["game_id"]).row(0, named=True)
    assert row["home_off_out"] == pytest.approx(0.9)
    assert row["home_def_out"] == pytest.approx(0.9)
    assert row["away_off_out"] == 0.0
