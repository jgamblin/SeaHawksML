"""Monte Carlo simulation of the rest of the regular season (predictions/season_sim.jsonl).

Completed games keep their actual result (a tie is a tie); every remaining game is an
independent Bernoulli draw with the model's home-win probability (no ties simulated, team
strength not resampled). Standings use SIMPLIFIED tiebreakers, applied in this order:

    win pct -> head-to-head win pct among the tied teams -> division win pct
            -> conference win pct -> coin flip

Head-to-head is computed once over the whole tied group (teams that never met score 0.5);
it is not re-applied when a multi-team tie is partly broken. Per conference the division
winners are seeded 1-4 by the same ordering, then the 3 best other teams are wild cards.
Win pct counts a tie as half a win. Everything is vectorized over simulations.
"""

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from seahawks_ml.config import SEASON_SIM_PATH, TEAM
from seahawks_ml.data import current_season
from seahawks_ml.features.columns import FEATURE_COLUMNS
from seahawks_ml.features.elo import elo_win_prob
from seahawks_ml.pipeline.history import append_record

N_WILD_CARDS = 3
SIM_KEYS = {
    "as_of", "season", "team", "n_sims", "wins_mean", "wins_p10", "wins_p50", "wins_p90", "win_dist",
    "p_playoffs", "p_division", "p_top_seed", "record_now", "model_version", "sim_day",
    "games_remaining",
}
OPTIONAL_KEYS = {"sim_day", "games_remaining"}  # absent from snapshots logged before the field existed
SIM_ONLY_HOURS = (10, 14, 18, 22)  # UTC hours at which a run with nothing else to do attempts the sim
SIM_DAY_OFFSET_HOURS = 10  # a sim day starts at 10:00 UTC, after every US night game has ended


def _pct(points: np.ndarray, games: np.ndarray) -> np.ndarray:
    """Win pct (ties already counted as half a point); 0.5 for a team with no such games."""
    return np.where(games > 0, points / np.maximum(games, 1), 0.5)


def _rank(cand: np.ndarray, group: np.ndarray, keys: dict[str, np.ndarray], h2h_pts: np.ndarray,
          h2h_games: np.ndarray) -> np.ndarray:
    """Rank (0 = best) of each team per simulation, candidates first.

    cand: (S, T) bool; group: (T, T) or (S, T, T) bool, which pairs may be compared head to head.
    keys: pct/div/conf/rand arrays of shape (S, T). Ranks between two candidates are only
    meaningful when they are in the same `group`.
    """
    pct = keys["pct"]
    n = pct.shape[1]
    tied = ((pct[:, :, None] == pct[:, None, :]) & cand[:, :, None] & cand[:, None, :] & group
            & ~np.eye(n, dtype=bool))
    h2h = _pct((h2h_pts * tied).sum(-1), (h2h_games * tied).sum(-1))
    # np.lexsort sorts by the LAST key first; negate so higher is better.
    order = np.lexsort((-keys["rand"], -keys["conf"], -keys["div"], -h2h, -pct, ~cand), axis=-1)
    return np.argsort(order, axis=-1)


def _outcomes(games: pl.DataFrame, p_home_by_game: dict[str, float], n_sims: int,
              rng: np.random.Generator) -> np.ndarray:
    """(S, G) matrix of home points per game: 1 win, 0.5 tie, 0 loss."""
    margin = games["margin"].to_numpy().astype(float)  # null -> nan
    done = ~np.isnan(margin)
    out = np.empty((n_sims, games.height))
    out[:, done] = np.sign(margin[done]) * 0.5 + 0.5
    remaining = games.filter(pl.col("margin").is_null())["game_id"].to_list()
    missing = [g for g in remaining if g not in p_home_by_game]
    if missing:
        print(f"season sim: no probability for games {missing}; using 0.5")
    p = np.array([p_home_by_game.get(g, 0.5) for g in remaining], dtype=float)
    out[:, ~done] = rng.random((n_sims, len(remaining))) < p
    return out


def simulate_season(games: pl.DataFrame, p_home_by_game: dict[str, float], teams: pl.DataFrame, season: int,
                    n_sims: int = 10_000, seed: int = 0, team: str = TEAM, now: datetime | None = None) -> dict:
    """Simulate `season`'s regular season and summarize it for `team`.

    games: schedule with game_id, season, game_type, home_team, away_team, margin (null = not played).
    p_home_by_game: home-win probability for every unplayed regular-season game.
    teams: team, conf, division for every team in those games.
    """
    games = games.filter((pl.col("season") == season) & (pl.col("game_type") == "REG")).sort("game_id")
    names = sorted(set(games["home_team"]) | set(games["away_team"]))
    info = {r["team"]: r for r in teams.iter_rows(named=True)}
    unknown = [t for t in names if t not in info]
    if unknown:
        raise ValueError(f"no conference/division for teams {unknown}")
    if team not in names:
        raise ValueError(f"{team} has no {season} regular-season games")
    conf = info[team]["conf"]
    members = [t for t in names if info[t]["conf"] == conf]  # only the team's conference is ranked
    idx = {t: i for i, t in enumerate(members)}
    n_teams, n_games = len(members), games.height

    rng = np.random.default_rng(seed)
    results = _outcomes(games, p_home_by_game, n_sims, rng)  # (S, G) home points

    home = games["home_team"].to_list()
    away = games["away_team"].to_list()
    same_div = np.array([info[h]["division"] == info[a]["division"] for h, a in zip(home, away)])
    same_conf = np.array([info[h]["conf"] == info[a]["conf"] for h, a in zip(home, away)])
    h_inc, a_inc = np.zeros((n_games, n_teams)), np.zeros((n_games, n_teams))
    pair_h, pair_a = np.zeros((n_games, n_teams * n_teams)), np.zeros((n_games, n_teams * n_teams))
    for g, (h, a) in enumerate(zip(home, away)):
        if h in idx:
            h_inc[g, idx[h]] = 1
        if a in idx:
            a_inc[g, idx[a]] = 1
        if h in idx and a in idx:
            pair_h[g, idx[h] * n_teams + idx[a]] = 1
            pair_a[g, idx[a] * n_teams + idx[h]] = 1

    def standing(mask: np.ndarray) -> np.ndarray:
        r = results[:, mask]
        pts = r @ h_inc[mask] + (1 - r) @ a_inc[mask]
        return _pct(pts, (h_inc[mask] + a_inc[mask]).sum(0))

    every = np.ones(n_games, dtype=bool)
    points = results @ h_inc + (1 - results) @ a_inc  # wins + ties/2, (S, T)
    keys = {"pct": standing(every), "div": standing(same_div), "conf": standing(same_conf),
            "rand": rng.random((n_sims, n_teams))}
    h2h_pts = (results @ pair_h + (1 - results) @ pair_a).reshape(n_sims, n_teams, n_teams)
    h2h_games = (pair_h + pair_a).sum(0).reshape(n_teams, n_teams)

    division = np.array([info[t]["division"] for t in members])
    same_division = division[:, None] == division[None, :]
    all_cand = np.ones((n_sims, n_teams), dtype=bool)
    div_rank = _rank(all_cand, same_division, keys, h2h_pts, h2h_games)
    winner = np.zeros((n_sims, n_teams), dtype=bool)
    rows = np.arange(n_sims)
    for d in np.unique(division):
        cols = np.flatnonzero(division == d)
        winner[rows, cols[np.argmin(div_rank[:, cols], axis=1)]] = True
    anyone = np.ones((n_teams, n_teams), dtype=bool)
    seed_rank = _rank(winner, anyone, keys, h2h_pts, h2h_games)
    wild_rank = _rank(~winner, anyone, keys, h2h_pts, h2h_games)
    wild = ~winner & (wild_rank < N_WILD_CARDS)

    t = idx[team]
    wins = points[:, t]
    buckets = np.floor(wins + 0.5).astype(int)  # a tie counts half a win; x.5 rounds up
    values, counts = np.unique(buckets, return_counts=True)
    p10, p50, p90 = (float(np.quantile(wins, q, method="inverted_cdf")) for q in (0.1, 0.5, 0.9))

    team_games = games.filter(((pl.col("home_team") == team) | (pl.col("away_team") == team))
                              & pl.col("margin").is_not_null())
    sign = np.where(np.array(team_games["home_team"].to_list()) == team, 1, -1)
    own = team_games["margin"].to_numpy() * sign
    record = f"{int((own > 0).sum())}-{int((own < 0).sum())}-{int((own == 0).sum())}"
    return {
        "as_of": (now or datetime.now(UTC)).isoformat(),
        "sim_day": sim_day(now or datetime.now(UTC)).isoformat(),
        "season": season,
        "team": team,
        "n_sims": n_sims,
        "wins_mean": round(float(wins.mean()), 3),
        "wins_p10": p10,
        "wins_p50": p50,
        "wins_p90": p90,
        "win_dist": [{"wins": int(v), "prob": round(float(c) / n_sims, 4)} for v, c in zip(values, counts)],
        "p_playoffs": round(float((winner[:, t] | wild[:, t]).mean()), 4),
        "p_division": round(float(winner[:, t].mean()), 4),
        "p_top_seed": round(float((winner[:, t] & (seed_rank[:, t] == 0)).mean()), 4),
        "record_now": record,
        "games_remaining": games.filter(((pl.col("home_team") == team) | (pl.col("away_team") == team))
                                        & pl.col("margin").is_null()).height,
    }


def remaining_game_probs(frame: pl.DataFrame, model, season: int) -> dict[str, float]:
    """Model home-win probability for each unplayed regular-season game of `season`.

    Rows with missing/non-finite features fall back to the Elo probability (or 0.5).
    """
    rows = frame.filter((pl.col("season") == season) & (pl.col("game_type") == "REG")
                        & pl.col("margin").is_null())
    if not rows.height:
        return {}
    cols = [pl.col(c).cast(pl.Float64) for c in FEATURE_COLUMNS]
    bad = rows.select(pl.any_horizontal(*[c.is_null() | c.is_nan() | c.is_infinite() for c in cols])).to_series()
    probs: dict[str, float] = {}
    good = rows.filter(~bad)
    if good.height:
        p = model.predict(good)["p_win"]
        probs.update({g: float(x) for g, x in zip(good["game_id"].to_list(), p)})
    for g in rows.iter_rows(named=True):
        p = probs.get(g["game_id"])
        if p is not None and math.isfinite(p):
            continue
        elo = g["elo_home_pre"] is not None and g["elo_away_pre"] is not None
        probs[g["game_id"]] = float(elo_win_prob(g["elo_home_pre"] - g["elo_away_pre"], g["neutral"])) if elo else 0.5
        print(f"season sim: {g['game_id']} has incomplete features; using {'Elo' if elo else 0.5}")
    return probs


FINAL_SNAPSHOT_GRACE = timedelta(days=3)  # keep simulating briefly so final results get a snapshot


def season_in_progress(games: pl.DataFrame, now: datetime) -> bool:
    """True from the week before the current season's opener until a few days after its last
    regular-season game."""
    if not games.height:
        return False
    season = current_season(games, now)
    last = games.filter((pl.col("season") == season) & (pl.col("game_type") == "REG"))["kickoff_utc"].max()
    return last is not None and now < last + FINAL_SNAPSHOT_GRACE


def _prob(x) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and 0 <= x <= 1


def validate_sim(record: dict) -> None:
    keys = set(record)
    if not (SIM_KEYS - OPTIONAL_KEYS) <= keys <= SIM_KEYS:
        raise ValueError(f"record keys mismatch: missing {SIM_KEYS - OPTIONAL_KEYS - keys}, extra {keys - SIM_KEYS}")
    for k in ("p_playoffs", "p_division", "p_top_seed"):
        if not _prob(record[k]):
            raise ValueError(f"{k} must be in [0, 1], got {record[k]!r}")
    datetime.fromisoformat(record["as_of"])


def sim_day(now: datetime):
    """The simulation day `now` belongs to: UTC date after shifting back SIM_DAY_OFFSET_HOURS."""
    return (now.astimezone(UTC) - timedelta(hours=SIM_DAY_OFFSET_HOURS)).date()


def snapshot_day(record: dict) -> str:
    """ISO sim day of a snapshot; derived from as_of for snapshots logged without one."""
    return record.get("sim_day") or sim_day(datetime.fromisoformat(record["as_of"])).isoformat()


def has_snapshot_for(log: list[dict], now: datetime) -> bool:
    day = sim_day(now).isoformat()
    return any(snapshot_day(r) == day for r in log)


def append_snapshot(record: dict, path: Path = SEASON_SIM_PATH) -> bool:
    """Append unless a snapshot for the same sim day is already logged. Returns True if appended."""
    from seahawks_ml.pipeline.history import read_history

    validate_sim(record)
    if has_snapshot_for(read_history(path), datetime.fromisoformat(record["as_of"])):
        return False
    append_record(record, path, validate_sim)
    return True
