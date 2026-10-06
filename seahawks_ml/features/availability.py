"""Starter availability from snap shares and the final pre-game injury report.

A "starter" is a non-QB who averaged >= 50% of offense (or defense) snaps in the
games he appeared in among his team's previous `lookback` games (min 2 appearances).
QBs are excluded because the QB features already handle starter changes.
Null before the first season with snap counts, and null for a team-week with no injury
report rows at all (the report is not out yet, so availability is unknown, not zero).

Outputs per team (home_/away_ prefix):
- `off_out` / `def_out`: snap-share-weighted count of starters listed Out or Doubtful.
- `out_<group>` for the six position groups (ol, wrte, rb, dl, lb, db): the same weight
  per group, using offense shares for ol/wrte/rb and defense shares for dl/lb/db. In mode
  "values" each missing starter's weight is snap share x quality:
  - WR/TE, RB: 1 + quality_scale x (shrunk EPA/opportunity - replacement), floored at 0.
    EPA per target + carry over the player's games before kickoff in the game's season and
    the one before, shrunk toward replacement with `prior_opps` opportunities. Replacement
    is the group's mean EPA/opportunity over all seasons before the game's season.
  - OL, DL, LB, DB: lineman_quality[draft bucket] x (1 - w + w x durability), where
    durability is the player's mean share of the team's snaps over its previous 8 games
    (0 for games he missed) and w is `durability_weight`. Draft round is known at draft
    time, so it does not leak.
"""

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field

import polars as pl

from seahawks_ml.config import FIRST_SNAP_SEASON

OUT_STATUSES = {"Out", "Doubtful"}
STARTER_SHARE = 0.5
MIN_APPEARANCES = 2
AVAILABILITY_MODES = ("count", "groups", "values")
GROUPS = ("ol", "wrte", "rb", "dl", "lb", "db")
OFFENSE_GROUPS = frozenset({"ol", "wrte", "rb"})
# Snap-count positions plus the coarser labels nflverse's players table uses (fallback).
POSITION_GROUPS = {
    **dict.fromkeys(("T", "G", "C", "OT", "OG", "OL"), "ol"),
    **dict.fromkeys(("WR", "TE"), "wrte"),
    **dict.fromkeys(("RB", "FB", "HB"), "rb"),
    **dict.fromkeys(("DE", "DT", "NT", "DL"), "dl"),
    **dict.fromkeys(("LB", "OLB", "ILB", "MLB"), "lb"),
    **dict.fromkeys(("CB", "S", "SS", "FS", "DB", "SAF"), "db"),
}
SKILL_STAT_GROUPS = {"wrte": frozenset({"WR", "TE"}), "rb": frozenset({"RB"})}
SKILL_WINDOW_SEASONS = 2  # the game's season so far plus the previous season
REPLACEMENT_FALLBACK = 0.0  # EPA/opportunity when no earlier season has stats
DURABILITY_GAMES = 8


def _default_lineman_quality() -> dict[str, float]:
    return {"round_1": 1.3, "day_2": 1.1, "later": 1.0, "udfa": 0.9}


@dataclass(frozen=True)
class AvailabilityParams:
    """How missing starters are turned into features.

    mode: "count" - the four off/def snap-share columns; "groups" - six per-position-group
    snap-share columns; "values" - the same six groups weighted by player quality.
    prior_opps / quality_scale: skill-player (WR/TE, RB) EPA-per-opportunity shrinkage and scale.
    lineman_quality / durability_weight: OL/DL/LB/DB quality from draft bucket and snap durability.
    """

    mode: str = "count"
    prior_opps: float = 60.0
    quality_scale: float = 4.0
    lineman_quality: dict[str, float] = field(default_factory=_default_lineman_quality)
    durability_weight: float = 0.5

    def __post_init__(self):
        if self.mode not in AVAILABILITY_MODES:
            raise ValueError(f"unknown availability mode {self.mode!r}")


def position_group(position: str | None) -> str | None:
    """Position group of a snap-count (or players-table) position; None for QB/K/P/LS/unknown."""
    return POSITION_GROUPS.get(position) if position else None


def draft_bucket(draft_round: int | None) -> str:
    if draft_round is None:
        return "udfa"
    if draft_round == 1:
        return "round_1"
    return "day_2" if draft_round in (2, 3) else "later"


class _SkillHistory:
    """Per-player EPA/opportunity history indexed for leak-free lookups by kickoff."""

    def __init__(self, player_stats: pl.DataFrame | None, kickoff: dict):
        rows: dict[str, list[tuple]] = defaultdict(list)
        season_totals: dict[str, dict[int, list[float]]] = {g: defaultdict(lambda: [0.0, 0.0])
                                                            for g in SKILL_STAT_GROUPS}
        if player_stats is not None:
            for r in player_stats.iter_rows(named=True):
                ko = kickoff.get(r["game_id"])
                opps = (r["targets"] or 0) + (r["carries"] or 0)
                if ko is None or opps <= 0:
                    continue
                epa = (r["receiving_epa"] or 0.0) + (r["rushing_epa"] or 0.0)
                rows[r["player_id"]].append((ko, r["season"], opps, epa))
                for g, stat_groups in SKILL_STAT_GROUPS.items():
                    if r["position_group"] in stat_groups:
                        season_totals[g][r["season"]][0] += epa
                        season_totals[g][r["season"]][1] += opps
        self._season_totals = season_totals
        self._index: dict[str, tuple[list, list, list[float], list[float]]] = {}
        for pid, hist in rows.items():
            hist.sort(key=lambda h: h[0])
            cum_opps, cum_epa = [0.0], [0.0]
            for _, _, o, e in hist:
                cum_opps.append(cum_opps[-1] + o)
                cum_epa.append(cum_epa[-1] + e)
            self._index[pid] = ([h[0] for h in hist], [h[1] for h in hist], cum_opps, cum_epa)
        self._replacement: dict[tuple[str, int], float] = {}

    def replacement(self, group: str, season: int) -> float:
        key = (group, season)
        if key not in self._replacement:
            epa = opps = 0.0
            for s, (e, o) in self._season_totals[group].items():
                if s < season:
                    epa += e
                    opps += o
            self._replacement[key] = epa / opps if opps else REPLACEMENT_FALLBACK
        return self._replacement[key]

    def quality(self, player: str, group: str, season: int, ko, params: AvailabilityParams) -> float:
        repl = self.replacement(group, season)
        opps = epa = 0.0
        if player in self._index:
            kos, seasons, cum_opps, cum_epa = self._index[player]
            hi = bisect_left(kos, ko)
            lo = min(bisect_left(seasons, season - SKILL_WINDOW_SEASONS + 1), hi)
            opps, epa = cum_opps[hi] - cum_opps[lo], cum_epa[hi] - cum_epa[lo]
        k = params.prior_opps
        shrunk = (k * repl + epa) / (k + opps) if k + opps > 0 else repl
        return max(0.0, 1.0 + params.quality_scale * (shrunk - repl))


def compute_availability(
    games: pl.DataFrame,
    snaps: pl.DataFrame,
    injuries: pl.DataFrame,
    players: pl.DataFrame,
    player_stats: pl.DataFrame | None = None,
    params: AvailabilityParams = AvailabilityParams(),
    lookback: int = 4,
) -> pl.DataFrame:
    pfr_to_gsis = {r["pfr_id"]: r["gsis_id"] for r in players.iter_rows(named=True) if r["pfr_id"]}
    player_info = {r["gsis_id"]: r for r in players.iter_rows(named=True)}
    kickoff = {r["game_id"]: r["kickoff_utc"] for r in games.iter_rows(named=True)}
    values = params.mode == "values"
    skill = _SkillHistory(player_stats, kickoff) if values else None

    # (team, game) -> [(gsis, offense share, defense share, snap position)]
    team_game_snaps: dict[tuple[str, str], list[tuple[str, float, float, str | None]]] = defaultdict(list)
    for r in snaps.filter(pl.col("position").ne_missing("QB")).iter_rows(named=True):
        gsis = pfr_to_gsis.get(r["pfr_player_id"])
        if gsis and r["game_id"] in kickoff:
            team_game_snaps[(r["team"], r["game_id"])].append(
                (gsis, r["offense_pct"] or 0.0, r["defense_pct"] or 0.0, r["position"]))
    team_history: dict[str, list[str]] = defaultdict(list)
    for team, game_id in sorted(team_game_snaps, key=lambda k: kickoff[k[1]]):
        team_history[team].append(game_id)
    team_kickoffs = {t: [kickoff[g] for g in gids] for t, gids in team_history.items()}

    reported = {(r["season"], r["week"], r["team"])
                for r in injuries.select("season", "week", "team").unique().iter_rows(named=True)}
    out_lists: dict[tuple[int, int, str], set[str]] = defaultdict(set)
    for r in injuries.filter(pl.col("report_status").is_in(list(OUT_STATUSES))).iter_rows(named=True):
        out_lists[(r["season"], r["week"], r["team"])].add(r["gsis_id"])

    def durability(team: str, prior: list[str], gsis: str, offense: bool) -> float:
        recent = prior[-DURABILITY_GAMES:]
        total = 0.0
        for gid in recent:
            for pid, off, de, _ in team_game_snaps[(team, gid)]:
                if pid == gsis:
                    total += off if offense else de
                    break
        return total / len(recent)

    def quality(team: str, prior: list[str], gsis: str, group: str, game: dict) -> float:
        if group in SKILL_STAT_GROUPS:
            return skill.quality(gsis, group, game["season"], game["kickoff_utc"], params)
        info = player_info.get(gsis)
        base = params.lineman_quality[draft_bucket(info["draft_round"] if info else None)]
        w = params.durability_weight
        return max(0.0, base * (1 - w + w * durability(team, prior, gsis, group in OFFENSE_GROUPS)))

    def team_out(team: str, game: dict) -> tuple[float, float, dict[str, float]] | None:
        if game["season"] < FIRST_SNAP_SEASON:
            return None
        if (game["season"], game["week"], team) not in reported:
            return None
        prior = team_history.get(team, [])[:bisect_left(team_kickoffs.get(team, []), game["kickoff_utc"])]
        recent = prior[-lookback:]
        if not recent:
            return None
        shares: dict[str, list[list[float]]] = defaultdict(lambda: [[], []])
        latest_pos: dict[str, str] = {}
        for gid in recent:
            for gsis, off, de, pos in team_game_snaps[(team, gid)]:
                shares[gsis][0].append(off)
                shares[gsis][1].append(de)
                if pos:
                    latest_pos[gsis] = pos
        out_set = out_lists.get((game["season"], game["week"], team), set())
        off_out = def_out = 0.0
        groups = dict.fromkeys(GROUPS, 0.0)
        for gsis, (offs, defs) in shares.items():
            if gsis not in out_set or len(offs) < MIN_APPEARANCES:
                continue
            off_avg, def_avg = sum(offs) / len(offs), sum(defs) / len(defs)
            if off_avg >= STARTER_SHARE:
                off_out += off_avg
            if def_avg >= STARTER_SHARE:
                def_out += def_avg
            info = player_info.get(gsis)
            group = position_group(latest_pos.get(gsis)) or position_group(info["position"] if info else None)
            if group is None:
                continue
            share = off_avg if group in OFFENSE_GROUPS else def_avg
            if share >= STARTER_SHARE:
                groups[group] += share * (quality(team, prior, gsis, group, game) if values else 1.0)
        return off_out, def_out, groups

    rows = []
    for g in games.iter_rows(named=True):
        row = {"game_id": g["game_id"]}
        for side in ("home", "away"):
            res = team_out(g[f"{side}_team"], g)
            row[f"{side}_off_out"] = res[0] if res else None
            row[f"{side}_def_out"] = res[1] if res else None
            for grp in GROUPS:
                row[f"{side}_out_{grp}"] = res[2][grp] if res else None
        rows.append(row)
    schema = {"game_id": pl.Utf8, "home_off_out": pl.Float64, "home_def_out": pl.Float64,
              "away_off_out": pl.Float64, "away_def_out": pl.Float64,
              **{f"{side}_out_{grp}": pl.Float64 for side in ("home", "away") for grp in GROUPS}}
    return pl.DataFrame(rows, schema=schema)
