import pytest


@pytest.fixture(autouse=True)
def _isolate_site_inputs(tmp_path_factory, monkeypatch):
    """Site builds must never read real predictions/ or models/ files; point them at missing paths."""
    from seahawks_ml.site import build

    missing = tmp_path_factory.mktemp("no_such_inputs")
    for name, fname in [("LEAGUE_PATH", "league.jsonl"), ("SEASON_SIM_PATH", "season_sim.jsonl"),
                        ("METRICS_PATH", "metrics.json"),
                        ("BACKTEST_PATH", "backtest.json"), ("HOLDOUT_PATH", "holdout.json")]:
        monkeypatch.setattr(build, name, missing / fname)
