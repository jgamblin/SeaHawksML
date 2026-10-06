"""Starting-QB ratings shrunk toward a draft-capital prior.

    rating = (prior_dropbacks * bucket_prior + sum(qb_epa)) / (prior_dropbacks + dropbacks)

History is the QB's dropbacks before kickoff within the last `window_seasons` seasons.
Bucket priors are the average early-career (first two seasons) performance of QBs in
that draft bucket, using only seasons before the game's season (expanding window,
so they never see the season being predicted).
"""

from collections import defaultdict
from dataclasses import dataclass

import polars as pl

DRAFT_BUCKETS = ("round_1", "day_2", "day_3_udfa")
MIN_PRIOR_DROPBACKS = 500  # below this, a bucket falls back to the pooled prior


@dataclass(frozen=True)
class QBParams:
    prior_dropbacks: float = 250.0
    window_seasons: int = 3


def draft_bucket(draft_round: int | None) -> str:
    if draft_round == 1:
        return "round_1"
    if draft_round in (2, 3):
        return "day_2"
    return "day_3_udfa"


def _bucket_priors(qb_rows: list[dict], player_info: dict, seasons: list[int]) -> dict:
    """season -> bucket -> (epa_per_dropback, cpoe) using seasons strictly before it."""
    per_season: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0, 0]))
    for r in qb_rows:
        info = player_info.get(r["qb_id"])
        if info is None or info["rookie_season"] is None or r["season"] - info["rookie_season"] > 1:
            continue
        acc = per_season[r["season"]][draft_bucket(info["draft_round"])]
        acc[0] += r["qb_epa_sum"]
        acc[1] += r["dropbacks"]
        acc[2] += r["cpoe_sum"]
        acc[3] += r["cpoe_n"]
    priors = {}
    running: dict[str, list[float]] = defaultdict(lambda: [0.0, 0, 0.0, 0])
    for season in sorted(set(seasons)):
        pooled = [sum(running[b][i] for b in DRAFT_BUCKETS) for i in range(4)]
        pooled_val = (pooled[0] / pooled[1] if pooled[1] else 0.0,
                      pooled[2] / pooled[3] if pooled[3] else 0.0)
        priors[season] = {
            b: ((running[b][0] / running[b][1], running[b][2] / running[b][3] if running[b][3] else 0.0)
                if running[b][1] >= MIN_PRIOR_DROPBACKS else pooled_val)
            for b in DRAFT_BUCKETS
        }
        for b, acc in per_season.get(season, {}).items():
            for i in range(4):
                running[b][i] += acc[i]
    return priors


def compute_qb_features(
    games: pl.DataFrame,
    qb_games: pl.DataFrame,
    players: pl.DataFrame,
    params: QBParams = QBParams(),
) -> pl.DataFrame:
    """Per game: home/away starter EPA and CPOE ratings and draft bucket."""
    kickoff = {r["game_id"]: r["kickoff_utc"] for r in games.iter_rows(named=True)}
    player_info = {r["gsis_id"]: r for r in players.iter_rows(named=True)}
    qb_rows = [r | {"kickoff_utc": kickoff[r["game_id"]]}
               for r in qb_games.iter_rows(named=True) if r["game_id"] in kickoff]
    history: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(qb_rows, key=lambda x: x["kickoff_utc"]):
        history[r["qb_id"]].append(r)
    team_history: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(qb_rows, key=lambda x: x["kickoff_utc"]):
        team_history[r["team"]].append(r)
    priors = _bucket_priors(qb_rows, player_info, games["season"].to_list())

    def rate(qb_id: str | None, season: int, ko) -> tuple[float, float, str]:
        info = player_info.get(qb_id) if qb_id else None
        bucket = draft_bucket(info["draft_round"] if info else None)
        p_epa, p_cpoe = priors[season][bucket]
        n = epa = c_sum = c_n = 0.0
        for h in history.get(qb_id, []):
            if h["kickoff_utc"] >= ko:
                break
            if h["season"] > season - params.window_seasons:
                n += h["dropbacks"]
                epa += h["qb_epa_sum"]
                c_sum += h["cpoe_sum"]
                c_n += h["cpoe_n"]
        k = params.prior_dropbacks
        return (k * p_epa + epa) / (k + n), (k * p_cpoe + c_sum) / (k + c_n), bucket

    def latest_starter(team: str, ko) -> str | None:
        """QB with the most dropbacks in the team's latest game before kickoff."""
        best: dict | None = None
        for h in reversed(team_history.get(team, [])):
            if h["kickoff_utc"] >= ko:
                continue
            if best is None:
                best = h
            elif h["game_id"] == best["game_id"]:
                if h["dropbacks"] > best["dropbacks"]:
                    best = h
            else:
                break
        return best["qb_id"] if best else None

    rows = []
    for g in games.iter_rows(named=True):
        ko = g["kickoff_utc"]
        # Expected starters are often unannounced far ahead: use the team's last starter.
        home_qb = g["home_qb_id"] or latest_starter(g["home_team"], ko)
        away_qb = g["away_qb_id"] or latest_starter(g["away_team"], ko)
        h_epa, h_cpoe, h_b = rate(home_qb, g["season"], ko)
        a_epa, a_cpoe, a_b = rate(away_qb, g["season"], ko)
        rows.append({"game_id": g["game_id"], "home_qb_epa": h_epa, "away_qb_epa": a_epa,
                     "home_qb_cpoe": h_cpoe, "away_qb_cpoe": a_cpoe,
                     "home_qb_bucket": h_b, "away_qb_bucket": a_b})
    return pl.DataFrame(rows, schema={
        "game_id": pl.Utf8, "home_qb_epa": pl.Float64, "away_qb_epa": pl.Float64,
        "home_qb_cpoe": pl.Float64, "away_qb_cpoe": pl.Float64,
        "home_qb_bucket": pl.Utf8, "away_qb_bucket": pl.Utf8,
    })
