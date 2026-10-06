"""Pre-game offensive/defensive team ratings with a shrunk prior-season anchor.

The same scheme is applied to several per-game stats (each with an offense side and a
defense "allowed" side):

    total EPA/play   -> off_rating, def_rating
    pass EPA/play    -> pass_off_rating, pass_def_rating     (dropbacks)
    rush EPA/play    -> rush_off_rating, rush_def_rating     (designed runs)
    success rate     -> success_off_rating, success_def_rating

All values are deviations from the previous season's league average of that stat, so
they're comparable across eras and use no same-season league information.

    prior   = prior_regression * (team_prev_season_avg - league_prev_season_avg)
    rating  = (k * prior + sum(current_season_devs)) / (k + n_current_games)

k is `prior_games`, or `prior_games_new_coach` when the team has a new head coach,
so new regimes move off last season's numbers faster.

Opponent adjustment (`opponent_adjust=True`): when a game is added to a team's running
sums, its offensive deviation is reduced by the opponent's *pre-game* defensive rating
for that stat, and its defensive deviation by the opponent's pre-game offensive rating.
Ratings for every game sharing a kickoff time are computed before any of those games
updates the running sums, so simultaneous games never see each other. The prior-season
anchor stays unadjusted.

`extra_stats=False` keeps only total EPA/play; the pass/rush/success columns are 0.0.
A stat whose columns are missing from `team_epa` (e.g. an old cache) is also 0.0.
"""

from collections import defaultdict
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class RatingParams:
    prior_regression: float = 0.6
    prior_games: float = 4.0
    prior_games_new_coach: float = 2.0
    opponent_adjust: bool = False
    extra_stats: bool = True


# output prefix -> (numerator column, denominator column) in team_epa
STATS = {
    "": ("epa_sum", "plays"),
    "pass_": ("pass_epa_sum", "pass_plays"),
    "rush_": ("rush_epa_sum", "rush_plays"),
    "success_": ("success_sum", "plays"),
}
RATING_COLUMNS = [f"{p}{side}_rating" for p in STATS for side in ("off", "def")]

Key = tuple[str, str]  # (game_id, team)


def _per_game(team_epa: pl.DataFrame, num: str, den: str) -> dict[Key, tuple[float, float]]:
    """(game_id, team) -> (offense value per play, defense value per play allowed)."""
    if num not in team_epa.columns or den not in team_epa.columns:
        return {}
    off = {(r["game_id"], r["team"]): r[num] / r[den]
           for r in team_epa.iter_rows(named=True) if r[den] and r[num] is not None}
    out = {}
    for r in team_epa.iter_rows(named=True):
        key = (r["game_id"], r["team"])
        allowed = off.get((r["game_id"], r["opponent"]))
        if key in off and allowed is not None:
            out[key] = (off[key], allowed)
    return out


def _stat_ratings(
    values: dict[Key, tuple[float, float]],
    groups: list[list[dict]],
    season_of: dict[str, int],
    new_coach: dict[Key, int],
    params: RatingParams,
) -> dict[Key, tuple[float, float]]:
    league_sum: dict[int, float] = defaultdict(float)
    league_n: dict[int, int] = defaultdict(int)
    team_sums: dict[tuple[str, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    for (game_id, team), (o, d) in values.items():
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

    running: dict[tuple[str, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    ratings: dict[Key, tuple[float, float]] = {}
    for group in groups:
        # 1) every game at this kickoff gets its pre-game rating first ...
        for r in group:
            key = (r["game_id"], r["team"])
            k = params.prior_games_new_coach if new_coach.get(key) else params.prior_games
            p_off, p_def = prior(r["team"], r["season"])
            cur = running[(r["team"], r["season"])]
            ratings[key] = ((k * p_off + cur[0]) / (k + cur[2]), (k * p_def + cur[1]) / (k + cur[2]))
        # 2) ... then results update the running sums
        for r in group:
            key = (r["game_id"], r["team"])
            if key not in values:
                continue
            o, d = values[key]
            c = center(r["season"])
            o, d = o - c, d - c
            if params.opponent_adjust:
                opp_off, opp_def = ratings.get((r["game_id"], r["opponent"]), (0.0, 0.0))
                o, d = o - opp_def, d - opp_off
            cur = running[(r["team"], r["season"])]
            cur[0] += o
            cur[1] += d
            cur[2] += 1
    return ratings


def compute_team_ratings(
    team_games: pl.DataFrame,
    team_epa: pl.DataFrame,
    coach_flags: pl.DataFrame,
    params: RatingParams = RatingParams(),
) -> pl.DataFrame:
    season_of = {r["game_id"]: r["season"] for r in team_games.iter_rows(named=True)}
    new_coach = {(r["game_id"], r["team"]): r["new_head_coach"] for r in coach_flags.iter_rows(named=True)}
    ordered = team_games.sort("kickoff_utc", "game_id", "team")
    groups = [g.to_dicts() for _, g in ordered.group_by("kickoff_utc", maintain_order=True)]
    rows = ordered.select("game_id", "team").to_dicts()

    for prefix, (num, den) in STATS.items():
        if prefix and not params.extra_stats:
            ratings = {}
        else:
            ratings = _stat_ratings(_per_game(team_epa, num, den), groups, season_of, new_coach, params)
        for row in rows:
            off, dfn = ratings.get((row["game_id"], row["team"]), (0.0, 0.0))
            row[f"{prefix}off_rating"] = off
            row[f"{prefix}def_rating"] = dfn
    schema = {"game_id": pl.Utf8, "team": pl.Utf8, **{c: pl.Float64 for c in RATING_COLUMNS}}
    return pl.DataFrame(rows, schema=schema)
