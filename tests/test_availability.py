import polars as pl
import pytest

from seahawks_ml.features.availability import (
    GROUPS,
    AvailabilityParams,
    compute_availability,
    position_group,
)
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
        {"season": 2014, "week": 5, "team": game["away_team"], "gsis_id": "x", "position": "LB",
         "report_status": "Questionable"},
    ], schema=INJURIES_SCHEMA)
    out = compute_availability(raw.games, raw.snaps, injuries, raw.players)
    row = out.filter(pl.col("game_id") == game["game_id"]).row(0, named=True)
    assert row["home_off_out"] == pytest.approx(0.9)
    assert row["home_def_out"] == pytest.approx(0.9)
    assert row["away_off_out"] == 0.0


def test_snaps_with_null_position_are_kept():
    raw = make_raw()
    game = raw.games.filter((pl.col("season") == 2014) & (pl.col("week") == 5)).row(0, named=True)
    team = game["home_team"]
    injuries = pl.DataFrame([
        {"season": 2014, "week": 5, "team": team, "gsis_id": f"{team}-P0", "position": "WR",
         "report_status": "Out"},
    ], schema=INJURIES_SCHEMA)
    snaps = raw.snaps.with_columns(pl.lit(None, dtype=pl.Utf8).alias("position"))
    out = compute_availability(raw.games, snaps, injuries, raw.players)
    row = out.filter(pl.col("game_id") == game["game_id"]).row(0, named=True)
    assert row["home_off_out"] == pytest.approx(0.9)


def test_no_injury_report_rows_means_unknown_not_zero():
    raw = make_raw()
    game = raw.games.filter((pl.col("season") == 2014) & (pl.col("week") == 5)).row(0, named=True)
    injuries = raw.injuries.filter(~((pl.col("season") == 2014) & (pl.col("week") == 5)
                                     & (pl.col("team") == game["home_team"])))
    out = compute_availability(raw.games, raw.snaps, injuries, raw.players)
    row = out.filter(pl.col("game_id") == game["game_id"]).row(0, named=True)
    assert row["home_off_out"] is None and row["home_def_out"] is None
    assert row["away_off_out"] is not None


# --- position groups and player values -------------------------------------------------

GROUP_COLS = [f"{side}_out_{g}" for side in ("home", "away") for g in GROUPS]


def _week5_game(raw):
    return raw.games.filter((pl.col("season") == 2014) & (pl.col("week") == 5)).row(0, named=True)


def _home_injuries(game, statuses: dict[str, str]):
    team = game["home_team"]
    rows = [{"season": 2014, "week": 5, "team": team, "gsis_id": f"{team}-P{i}", "position": "X",
             "report_status": st} for i, st in statuses.items()]
    rows.append({"season": 2014, "week": 5, "team": game["away_team"], "gsis_id": "x", "position": "LB",
                 "report_status": "Questionable"})
    return pl.DataFrame(rows, schema=INJURIES_SCHEMA)


def _row(raw, injuries, params, snaps=None, player_stats=None):
    game = _week5_game(raw)
    out = compute_availability(raw.games, raw.snaps if snaps is None else snaps, injuries, raw.players,
                               raw.player_stats if player_stats is None else player_stats, params)
    return out.filter(pl.col("game_id") == game["game_id"]).row(0, named=True)


def test_position_groups():
    expected = {"T": "ol", "G": "ol", "C": "ol", "OT": "ol", "OL": "ol",
                "WR": "wrte", "TE": "wrte", "RB": "rb", "FB": "rb",
                "DE": "dl", "DT": "dl", "NT": "dl", "DL": "dl",
                "LB": "lb", "OLB": "lb", "ILB": "lb", "MLB": "lb",
                "CB": "db", "S": "db", "SS": "db", "FS": "db", "DB": "db", "SAF": "db"}
    for pos, group in expected.items():
        assert position_group(pos) == group, pos
    # real snap counts carry combined labels; the first listed position decides
    for pos, group in {"C/G": "ol", "G/T": "ol", "TE/D": "wrte", "WR/R": "wrte", "FB/T": "rb",
                       "DE/L": "dl", "DT/D": "dl", "LB/S": "lb", "CB/R": "db", "DB/L": "db"}.items():
        assert position_group(pos) == group, pos
    for pos in ("QB", "K", "P", "LS", "K/P", None, "??"):
        assert position_group(pos) is None, pos
    assert GROUPS == ("ol", "wrte", "rb", "dl", "lb", "db")


def test_groups_mode_weights_out_starters_by_snap_share_per_group():
    raw = make_raw()
    game = _week5_game(raw)
    inj = _home_injuries(game, {0: "Out", 4: "Doubtful", 7: "Out", 8: "Questionable"})
    row = _row(raw, inj, AvailabilityParams(mode="groups"))
    assert row["home_out_wrte"] == pytest.approx(0.9)
    assert row["home_out_rb"] == pytest.approx(0.9)
    assert row["home_out_lb"] == pytest.approx(0.9)
    assert row["home_out_ol"] == row["home_out_dl"] == row["home_out_db"] == 0.0
    assert all(row[f"away_out_{g}"] == 0.0 for g in GROUPS)
    # the old columns are still produced
    assert row["home_off_out"] == pytest.approx(1.8)
    assert row["home_def_out"] == pytest.approx(0.9)


def test_group_falls_back_to_player_position_when_snap_position_missing():
    raw = make_raw()
    inj = _home_injuries(_week5_game(raw), {0: "Out"})
    snaps = raw.snaps.with_columns(pl.lit(None, dtype=pl.Utf8).alias("position"))
    row = _row(raw, inj, AvailabilityParams(mode="groups"), snaps=snaps)
    assert row["home_out_wrte"] == pytest.approx(0.9)


def test_group_columns_null_when_availability_unknown():
    raw = make_raw()
    out = compute_availability(raw.games, raw.snaps, raw.injuries, raw.players, raw.player_stats,
                               AvailabilityParams(mode="values"))
    early = out.join(raw.games.select("game_id", "season"), on="game_id").filter(pl.col("season") < 2013)
    for c in GROUP_COLS:
        assert early[c].null_count() == early.height, c
    late = out.join(raw.games.select("game_id", "season"), on="game_id").filter(pl.col("season") == 2014)
    assert (late["home_off_out"].is_null() == late["home_out_wrte"].is_null()).all()


def _expected_skill_quality(raw, player: str, group_stats: set[str], params: AvailabilityParams,
                            player_stats: pl.DataFrame | None = None) -> float:
    """Independent re-derivation of the skill-player quality for the week-5 2014 game.

    Receiving and rushing are kept apart and combined in the player's own targets/carries mix:
    shrink toward the group mean, measure against replacement (25th percentile of player-season
    EPA per target / per carry among players with >= 20 of them, earlier seasons).
    """
    game = _week5_game(raw)
    stats = raw.player_stats if player_stats is None else player_stats
    ps = stats.join(raw.games.select("game_id", "kickoff_utc"), on="game_id").with_columns(
        pl.col("receiving_epa").fill_null(0.0), pl.col("rushing_epa").fill_null(0.0))
    before = ps.filter(pl.col("season") < 2014, pl.col("position_group").is_in(list(group_stats)))
    seasons = before.group_by("player_id", "season").agg(pl.col("receiving_epa", "targets", "rushing_epa",
                                                                 "carries").sum())

    def mean_and_repl(epa: str, n: str) -> tuple[float, float]:
        mean = before[epa].sum() / before[n].sum() if before[n].sum() else 0.0
        qualified = seasons.filter(pl.col(n) >= 20)
        repl = (qualified[epa] / qualified[n]).quantile(0.25, "linear") if qualified.height else mean
        return mean, repl

    (mean_rec, repl_rec), (mean_rush, repl_rush) = mean_and_repl("receiving_epa", "targets"), \
        mean_and_repl("rushing_epa", "carries")
    own = ps.filter(pl.col("player_id") == player, pl.col("season") >= 2013,
                    pl.col("kickoff_utc") < game["kickoff_utc"])
    t, c = own["targets"].sum(), own["carries"].sum()
    mix = t / (t + c) if t + c else before["targets"].sum() / (before["targets"].sum() + before["carries"].sum())
    k = params.prior_opps
    shrunk = (k * (mix * mean_rec + (1 - mix) * mean_rush) + own["receiving_epa"].sum()
              + own["rushing_epa"].sum()) / (k + t + c)
    return max(0.0, 1 + params.quality_scale * (shrunk - (mix * repl_rec + (1 - mix) * repl_rush)))


@pytest.mark.parametrize("player,group,stats", [(0, "wrte", {"WR", "TE"}), (4, "rb", {"RB"})])
def test_values_mode_weights_skill_players_by_shrunk_epa(player, group, stats):
    raw = make_raw()
    game = _week5_game(raw)
    params = AvailabilityParams(mode="values", prior_opps=30.0, quality_scale=6.0)
    row = _row(raw, _home_injuries(game, {player: "Out"}), params)
    quality = _expected_skill_quality(raw, f"{game['home_team']}-P{player}", stats, params)
    assert quality != pytest.approx(1.0)
    assert row[f"home_out_{group}"] == pytest.approx(0.9 * quality)
    assert row["home_off_out"] == pytest.approx(0.9)  # old column stays a plain snap share


def test_values_skill_baseline_uses_players_own_receiving_rushing_mix():
    """An RB who also catches passes is measured against receiving and rushing baselines in his own mix,
    not against one pooled EPA/opportunity (receiving EPA/target runs far above rushing EPA/carry)."""
    raw = make_raw()
    game = _week5_game(raw)
    target = f"{game['home_team']}-P4"
    is_target = pl.col("player_id") == target
    stats = raw.player_stats.with_columns(
        pl.when(is_target).then(4).otherwise(pl.col("targets")).alias("targets"),
        pl.when(is_target).then(8.0).otherwise(pl.col("receiving_epa")).alias("receiving_epa"))
    params = AvailabilityParams(mode="values", prior_opps=30.0, quality_scale=6.0)
    row = _row(raw, _home_injuries(game, {4: "Out"}), params, player_stats=stats)
    quality = _expected_skill_quality(raw, target, {"RB"}, params, player_stats=stats)
    assert row["home_out_rb"] == pytest.approx(0.9 * quality)


def test_values_skill_quality_is_measured_against_replacement_not_the_mean():
    """A player with no history is shrunk all the way to the group mean, which sits above
    replacement level (the 25th percentile), so he weighs more than 1."""
    raw = make_raw()
    game = _week5_game(raw)
    target = f"{game['home_team']}-P0"
    stats = raw.player_stats.filter(pl.col("player_id") != target)
    params = AvailabilityParams(mode="values", prior_opps=30.0, quality_scale=6.0)
    row = _row(raw, _home_injuries(game, {0: "Out"}), params, player_stats=stats)
    quality = _expected_skill_quality(raw, target, {"WR", "TE"}, params, player_stats=stats)
    assert quality > 1.0
    assert row["home_out_wrte"] == pytest.approx(0.9 * quality)


def test_values_quality_is_floored_at_zero():
    raw = make_raw()
    game = _week5_game(raw)
    target = f"{game['home_team']}-P0"
    stats = raw.player_stats.with_columns(
        pl.when(pl.col("player_id") == target).then(-50.0).otherwise(pl.col("receiving_epa")).alias("receiving_epa"))
    row = _row(raw, _home_injuries(game, {0: "Out"}), AvailabilityParams(mode="values"), player_stats=stats)
    assert row["home_out_wrte"] == 0.0


def test_values_ignore_stats_from_the_game_and_later():
    raw = make_raw()
    game = _week5_game(raw)
    later = raw.games.filter(pl.col("kickoff_utc") >= game["kickoff_utc"])["game_id"].to_list()
    noisy = raw.player_stats.with_columns(
        pl.when(pl.col("game_id").is_in(later)).then(99.0).otherwise(pl.col("receiving_epa")).alias("receiving_epa"))
    inj = _home_injuries(game, {0: "Out", 4: "Out"})
    params = AvailabilityParams(mode="values")
    assert _row(raw, inj, params) == _row(raw, inj, params, player_stats=noisy)


def test_values_mode_linemen_use_draft_bucket_and_durability():
    raw = make_raw()
    game = _week5_game(raw)
    team = game["home_team"]
    params = AvailabilityParams(mode="values", durability_weight=0.5)
    # P5 round 1, P7 round 5 ("later"), P8 undrafted; all played 0.9 of snaps in the last 8 games
    row = _row(raw, _home_injuries(game, {5: "Out"}), params)
    assert row["home_out_lb"] == pytest.approx(0.9 * 1.3 * (0.5 + 0.5 * 0.9))
    row = _row(raw, _home_injuries(game, {7: "Out", 8: "Out"}), params)
    assert row["home_out_lb"] == pytest.approx(0.9 * (1.0 + 0.9) * (0.5 + 0.5 * 0.9))
    # P5 missed the team's 2013 games: still a starter (last 4 games) but only 4 of the last 8
    snaps = raw.snaps.filter(~((pl.col("pfr_player_id") == f"{team}P005") & (pl.col("season") == 2013)))
    row = _row(raw, _home_injuries(game, {5: "Out"}), params, snaps=snaps)
    assert row["home_out_lb"] == pytest.approx(0.9 * 1.3 * (0.5 + 0.5 * 0.45))


def test_availability_params_are_hashable_and_lineman_quality_immutable():
    default = AvailabilityParams()
    assert hash(default) == hash(AvailabilityParams())
    assert default.lineman_quality == (("round_1", 1.3), ("day_2", 1.1), ("later", 1.0), ("udfa", 0.9))
    assert isinstance(default.lineman_quality, tuple)
    # a mapping (old configs, JSON) is normalized to the same immutable form
    as_dict = AvailabilityParams(lineman_quality={"round_1": 1.3, "day_2": 1.1, "later": 1.0, "udfa": 0.9})
    assert as_dict == default and hash(as_dict) == hash(default)
    as_lists = AvailabilityParams(lineman_quality=[["round_1", 1.5], ["day_2", 1.2], ["later", 1.0], ["udfa", 0.8]])
    assert dict(as_lists.lineman_quality) == {"round_1": 1.5, "day_2": 1.2, "later": 1.0, "udfa": 0.8}
    assert {default, as_dict, as_lists} == {default, as_lists}
    with pytest.raises(ValueError):
        AvailabilityParams(lineman_quality={"round_1": 1.3})


def test_availability_params_to_dict_keeps_lineman_quality_a_json_object():
    d = AvailabilityParams(mode="values").to_dict()
    assert d["lineman_quality"] == {"round_1": 1.3, "day_2": 1.1, "later": 1.0, "udfa": 0.9}
    assert AvailabilityParams(**d) == AvailabilityParams(mode="values")
