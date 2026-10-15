from datetime import UTC, datetime

import polars as pl

from seahawks_ml.features.starters import apply_expected_starters

NOW = datetime(2026, 10, 15, 19, 0, tzinfo=UTC)


def _games():
    return pl.DataFrame({
        "game_id": ["past", "tonight", "later"],
        "kickoff_utc": [datetime(2026, 10, 11, 20, 25, tzinfo=UTC), datetime(2026, 10, 16, 0, 15, tzinfo=UTC),
                        datetime(2026, 10, 25, 20, 25, tzinfo=UTC)],
        "home_team": ["SEA", "DEN", "SEA"],
        "away_team": ["SF", "SEA", "KC"],
        "margin": [-3, None, None],
        "home_qb_id": ["darnold", "nix", None],
        "away_qb_id": ["purdy", "lock", None],
    }, schema_overrides={"kickoff_utc": pl.Datetime("us", "UTC")})


def _depth():
    return pl.DataFrame({
        "team": ["SEA", "SEA", "SEA", "DEN"],
        "dt": [datetime(2026, 10, 1, tzinfo=UTC), datetime(2026, 10, 15, 12, tzinfo=UTC),
               datetime(2026, 10, 15, 20, tzinfo=UTC), datetime(2026, 10, 14, tzinfo=UTC)],
        "qb_id": ["lock", "darnold", "milroe", "nix"],
    }, schema_overrides={"dt": pl.Datetime("us", "UTC")})


def test_upcoming_games_use_latest_depth_chart_qb_before_now():
    out = {r["game_id"]: r for r in apply_expected_starters(_games(), _depth(), NOW).iter_rows(named=True)}
    assert out["tonight"]["away_qb_id"] == "darnold"  # stale schedule said lock; 20:00 snapshot is after NOW
    assert out["tonight"]["home_qb_id"] == "nix"
    assert out["later"]["home_qb_id"] == "darnold" and out["later"]["away_qb_id"] is None  # no KC depth chart


def test_played_games_are_untouched():
    out = apply_expected_starters(_games(), _depth(), NOW).filter(pl.col("game_id") == "past").row(0, named=True)
    assert out["home_qb_id"] == "darnold" and out["away_qb_id"] == "purdy"


def test_empty_depth_chart_is_a_no_op():
    empty = pl.DataFrame(schema={"team": pl.Utf8, "dt": pl.Datetime("us", "UTC"), "qb_id": pl.Utf8})
    assert apply_expected_starters(_games(), empty, NOW).equals(_games())
