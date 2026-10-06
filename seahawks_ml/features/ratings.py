"""Pre-game offensive/defensive EPA-per-play ratings with a shrunk prior-season anchor.

All values are deviations from the previous season's league average, so they're
comparable across eras and use no same-season league information.

    prior   = prior_regression * (team_prev_season_avg - league_prev_season_avg)
    rating  = (k * prior + sum(current_season_devs)) / (k + n_current_games)

k is `prior_games`, or `prior_games_new_coach` when the team has a new head coach,
so new regimes move off last season's numbers faster.
"""

from collections import defaultdict
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class RatingParams:
    prior_regression: float = 0.6
    prior_games: float = 4.0
    prior_games_new_coach: float = 2.0


def _per_game_epa(team_epa: pl.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    """(game_id, team) -> (offense EPA/play, defense EPA/play allowed)."""
    off = {(r["game_id"], r["team"]): r["epa_sum"] / r["plays"]
           for r in team_epa.iter_rows(named=True) if r["plays"]}
    out = {}
    for r in team_epa.iter_rows(named=True):
        key = (r["game_id"], r["team"])
        allowed = off.get((r["game_id"], r["opponent"]))
        if key in off and allowed is not None:
            out[key] = (off[key], allowed)
    return out


def compute_team_ratings(
    team_games: pl.DataFrame,
    team_epa: pl.DataFrame,
    coach_flags: pl.DataFrame,
    params: RatingParams = RatingParams(),
) -> pl.DataFrame:
    epa = _per_game_epa(team_epa)
    season_of = {r["game_id"]: r["season"] for r in team_games.iter_rows(named=True)}

    league_sum: dict[int, float] = defaultdict(float)
    league_n: dict[int, int] = defaultdict(int)
    team_sums: dict[tuple[str, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    for (game_id, team), (o, d) in epa.items():
        season = season_of.get(game_id)
        if season is None:
            continue
        league_sum[season] += o
        league_n[season] += 1
        s = team_sums[(team, season)]
        s[0] += o
        s[1] += d
        s[2] += 1

    def prior(team: str, season: int) -> tuple[float, float]:
        prev = team_sums.get((team, season - 1))
        if not prev or not league_n.get(season - 1):
            return 0.0, 0.0
        lg = league_sum[season - 1] / league_n[season - 1]
        return (params.prior_regression * (prev[0] / prev[2] - lg),
                params.prior_regression * (prev[1] / prev[2] - lg))

    def center(season: int) -> float:
        n = league_n.get(season - 1)
        return league_sum[season - 1] / n if n else 0.0

    new_coach = {(r["game_id"], r["team"]): r["new_head_coach"] for r in coach_flags.iter_rows(named=True)}
    running: dict[tuple[str, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    rows = []
    for r in team_games.sort("kickoff_utc", "game_id").iter_rows(named=True):
        team, season, game_id = r["team"], r["season"], r["game_id"]
        k = params.prior_games_new_coach if new_coach.get((game_id, team)) else params.prior_games
        p_off, p_def = prior(team, season)
        cur = running[(team, season)]
        rows.append({
            "game_id": game_id,
            "team": team,
            "off_rating": (k * p_off + cur[0]) / (k + cur[2]),
            "def_rating": (k * p_def + cur[1]) / (k + cur[2]),
        })
        if (game_id, team) in epa:
            o, d = epa[(game_id, team)]
            c = center(season)
            cur[0] += o - c
            cur[1] += d - c
            cur[2] += 1
    return pl.DataFrame(rows, schema={"game_id": pl.Utf8, "team": pl.Utf8,
                                      "off_rating": pl.Float64, "def_rating": pl.Float64})
