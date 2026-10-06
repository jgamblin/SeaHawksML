"""Starter availability from snap shares and the final pre-game injury report.

A "starter" is a non-QB who averaged >= 50% of offense (or defense) snaps in the
games he appeared in among his team's previous `lookback` games (min 2 appearances).
The feature is the snap-share-weighted count of starters listed Out or Doubtful.
QBs are excluded because the QB features already handle starter changes.
Null before the first season with snap counts.
"""

from collections import defaultdict

import polars as pl

from seahawks_ml.config import FIRST_SNAP_SEASON

OUT_STATUSES = {"Out", "Doubtful"}
STARTER_SHARE = 0.5
MIN_APPEARANCES = 2


def compute_availability(
    games: pl.DataFrame,
    snaps: pl.DataFrame,
    injuries: pl.DataFrame,
    players: pl.DataFrame,
    lookback: int = 4,
) -> pl.DataFrame:
    pfr_to_gsis = {r["pfr_id"]: r["gsis_id"] for r in players.iter_rows(named=True) if r["pfr_id"]}
    kickoff = {r["game_id"]: r["kickoff_utc"] for r in games.iter_rows(named=True)}

    team_game_snaps: dict[tuple[str, str], list[tuple[str, float, float]]] = defaultdict(list)
    for r in snaps.filter(pl.col("position").ne_missing("QB")).iter_rows(named=True):
        gsis = pfr_to_gsis.get(r["pfr_player_id"])
        if gsis and r["game_id"] in kickoff:
            team_game_snaps[(r["team"], r["game_id"])].append(
                (gsis, r["offense_pct"] or 0.0, r["defense_pct"] or 0.0))
    team_history: dict[str, list[str]] = defaultdict(list)
    for team, game_id in sorted(team_game_snaps, key=lambda k: kickoff[k[1]]):
        team_history[team].append(game_id)

    out_lists: dict[tuple[int, int, str], set[str]] = defaultdict(set)
    for r in injuries.filter(pl.col("report_status").is_in(list(OUT_STATUSES))).iter_rows(named=True):
        out_lists[(r["season"], r["week"], r["team"])].add(r["gsis_id"])

    def team_out(team: str, game: dict) -> tuple[float | None, float | None]:
        if game["season"] < FIRST_SNAP_SEASON:
            return None, None
        prior = [gid for gid in team_history.get(team, []) if kickoff[gid] < game["kickoff_utc"]]
        recent = prior[-lookback:]
        if not recent:
            return None, None
        shares: dict[str, list[list[float]]] = defaultdict(lambda: [[], []])
        for gid in recent:
            for gsis, off, de in team_game_snaps[(team, gid)]:
                shares[gsis][0].append(off)
                shares[gsis][1].append(de)
        out_set = out_lists.get((game["season"], game["week"], team), set())
        off_out = def_out = 0.0
        for gsis, (offs, defs) in shares.items():
            if gsis not in out_set or len(offs) < MIN_APPEARANCES:
                continue
            off_avg, def_avg = sum(offs) / len(offs), sum(defs) / len(defs)
            if off_avg >= STARTER_SHARE:
                off_out += off_avg
            if def_avg >= STARTER_SHARE:
                def_out += def_avg
        return off_out, def_out

    rows = []
    for g in games.iter_rows(named=True):
        h_off, h_def = team_out(g["home_team"], g)
        a_off, a_def = team_out(g["away_team"], g)
        rows.append({"game_id": g["game_id"], "home_off_out": h_off, "home_def_out": h_def,
                     "away_off_out": a_off, "away_def_out": a_def})
    return pl.DataFrame(rows, schema={"game_id": pl.Utf8, "home_off_out": pl.Float64,
                                      "home_def_out": pl.Float64, "away_off_out": pl.Float64,
                                      "away_def_out": pl.Float64})
