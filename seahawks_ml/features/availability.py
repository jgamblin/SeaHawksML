"""Starter availability from snap shares and the final pre-game injury report.

A "starter" is a non-QB who averaged >= 50% of offense (or defense) snaps in the
games he appeared in among his team's previous `lookback` games (min 2 appearances).
QBs are excluded because the QB features already handle starter changes.
Null before the first season with snap counts, and null for a team-week with no injury
report rows at all (the report is not out yet, so availability is unknown, not zero).

With `questionable`, a starter listed Questionable counts too, weighted by p_miss(practice status):
among Questionable starters with that final practice status (DNP / limited / full / none) in
seasons strictly before the game's, the share who then played 0 snaps (no snap row, or 0 share);
QUESTIONABLE_FALLBACK when a bucket has fewer than MIN_QUESTIONABLE_CASES such cases.

Outputs per team (home_/away_ prefix):
- `off_out` / `def_out`: snap-share-weighted count of starters listed Out or Doubtful (plus
  p_miss x share for Questionable starters with `questionable`).
- In mode "values_contract" each missing starter's group weight is snap share x contract quality:
  0.5 + the percentile of the apy_cap_pct of his latest contract with 0 < year_signed < the game's
  season within his position group's apy_cap_pct distribution over all contracts signed before that
  season (empirical CDF, ties take the mid-rank), so quality is in [0.5, 1.5]. A ratio to the group
  median saturated: minimum-salary deals dominate and apy_cap_pct is rounded to 0.001, so medians
  are ~0.004 and ~40% of starters hit the cap. With no such contract, DEFAULT_CONTRACT_QUALITY by
  draft bucket on the same scale. Only contracts signed in earlier seasons are read, so this is
  known before every game of the season.
- `out_<group>` for the six position groups (ol, wrte, rb, dl, lb, db): the same weight
  per group, using offense shares for ol/wrte/rb and defense shares for dl/lb/db. In the
  modes "values" and "values_pooled" each missing starter's weight is snap share x quality:
  - WR/TE, RB: 1 + quality_scale x (shrunk EPA/opportunity - replacement), floored at 0, so a
    replacement-level player weighs 1 and better ones more. Receiving (EPA per target) and
    rushing (EPA per carry) are kept apart: over the player's games before kickoff in the
    game's season and the one before he has t targets and c carries, and every baseline is
    taken in his own mix, (t x receiving + c x rushing) / (t + c) (the group's average mix
    when he has no history). shrunk = (k x mean + receiving EPA + rushing EPA) / (k + t + c)
    with k = `prior_opps`, shrinking toward the group mean. Mean = the group's EPA per target
    / per carry over all seasons before the game's season; replacement = the 25th percentile
    of player-season EPA per target / per carry over those seasons among player-seasons with
    >= 20 targets / carries (the mean when none qualify).
  - OL, DL, LB, DB: lineman_quality(draft bucket) x (1 - w + w x durability), where
    durability is the player's mean share of the team's snaps over its previous 8 games
    (0 for games he missed) and w is `durability_weight`. Draft round is known at draft
    time, so it does not leak.
"""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from seahawks_ml.config import FIRST_SNAP_SEASON

OUT_STATUSES = {"Out", "Doubtful"}
QUESTIONABLE = "Questionable"
# P(a Questionable starter plays 0 snaps) by practice status, used when earlier seasons have
# fewer than MIN_QUESTIONABLE_CASES Questionable starters in the bucket.
QUESTIONABLE_FALLBACK = {"dnp": 0.6, "limited": 0.25, "full": 0.05, None: 0.2}
MIN_QUESTIONABLE_CASES = 50
STARTER_SHARE = 0.5
MIN_APPEARANCES = 2
AVAILABILITY_MODES = ("count", "groups", "values", "count_diff", "values_pooled", "values_contract")
VALUE_MODES = frozenset({"values", "values_pooled"})  # modes that weight starters by EPA/draft quality
CONTRACT_MODE = "values_contract"  # weights starters by contract quality
POOLED_GROUP_MODES = frozenset({"values_pooled", CONTRACT_MODE})  # off/def diffs pooled from the groups
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
SKILL_STAT_POSITIONS = frozenset().union(*SKILL_STAT_GROUPS.values())
SKILL_WINDOW_SEASONS = 2  # the game's season so far plus the previous season
REPLACEMENT_FALLBACK = 0.0  # EPA/opportunity when no earlier season has stats
REPLACEMENT_PERCENTILE = 25.0  # replacement level: this percentile of earlier player-seasons
REPLACEMENT_MIN_OPPS = 20  # targets (carries) for a player-season to count toward replacement
DURABILITY_GAMES = 8


# OverTheCap contract positions that the snap-count labels don't cover
CONTRACT_POSITION_GROUPS = {**POSITION_GROUPS, **dict.fromkeys(("LT", "LG", "RG", "RT"), "ol"),
                            **dict.fromkeys(("ED", "IDL"), "dl")}
# Contract quality is 0.5 + the player's percentile in his position group's apy_cap_pct, in [0.5, 1.5];
# these are the defaults on that scale for a starter with no contract signed before the season
DEFAULT_CONTRACT_QUALITY = {"round_1": 1.1, "day_2": 0.9, "later": 0.75, "udfa": 0.6}

DRAFT_BUCKETS = ("round_1", "day_2", "later", "udfa")
# (draft bucket, quality) pairs: a tuple so AvailabilityParams stays immutable and hashable
LinemanQuality = tuple[tuple[str, float], ...]
DEFAULT_LINEMAN_QUALITY: LinemanQuality = (("round_1", 1.3), ("day_2", 1.1), ("later", 1.0), ("udfa", 0.9))


@dataclass(frozen=True)
class AvailabilityParams:
    """How missing starters are turned into features.

    mode: "count" - the four off/def snap-share columns; "groups" - six per-position-group
    snap-share columns; "values" - the same six groups weighted by player quality;
    "count_diff" - home minus away of the off/def counts (2 columns); "values_pooled" - quality-weighted
    out, home minus away, pooled over offense (OL+WRTE+RB) and defense (DL+LB+DB) groups (2 columns);
    "values_contract" - the same pooled diffs with each starter weighted by contract quality.
    prior_opps / quality_scale: skill-player (WR/TE, RB) EPA-per-opportunity shrinkage and scale.
    lineman_quality / durability_weight: OL/DL/LB/DB quality from draft bucket and snap durability.
    questionable: also count starters listed Questionable, each weighted by the probability that a
    Questionable starter with his practice status plays 0 snaps (see `questionable_cases`).
    """

    mode: str = "count"
    prior_opps: float = 60.0
    quality_scale: float = 4.0
    lineman_quality: LinemanQuality = DEFAULT_LINEMAN_QUALITY
    durability_weight: float = 0.5
    questionable: bool = False

    def __post_init__(self):
        if self.mode not in AVAILABILITY_MODES:
            raise ValueError(f"unknown availability mode {self.mode!r}")
        # accept a mapping or (bucket, quality) pairs (as JSON lists), store immutable pairs
        lq: Mapping | Iterable = self.lineman_quality
        pairs = tuple((str(k), float(v)) for k, v in (lq.items() if isinstance(lq, Mapping) else lq))
        if sorted(k for k, _ in pairs) != sorted(DRAFT_BUCKETS):
            raise ValueError(f"lineman_quality needs exactly the buckets {DRAFT_BUCKETS}, got {pairs}")
        object.__setattr__(self, "lineman_quality", pairs)

    def lineman_weight(self, bucket: str) -> float:
        return dict(self.lineman_quality)[bucket]

    def to_dict(self) -> dict:
        """JSON-ready form; lineman_quality as an object, which the constructor accepts back."""
        return {**asdict(self), "lineman_quality": dict(self.lineman_quality)}


def position_group(position: str | None) -> str | None:
    """Position group of a snap-count (or players-table) position; None for QB/K/P/LS/unknown.

    Combined snap-count labels such as "C/G" or "DE/L" use the first listed position.
    """
    return POSITION_GROUPS.get(position.split("/")[0]) if position else None


def contract_position_group(position: str | None) -> str | None:
    """Position group of an OverTheCap contract position; None for QB/K/P/LS/unknown."""
    return CONTRACT_POSITION_GROUPS.get(position) if position else None


def draft_bucket(draft_round: int | None) -> str:
    if draft_round is None:
        return "udfa"
    if draft_round == 1:
        return "round_1"
    return "day_2" if draft_round in (2, 3) else "later"


class _SkillHistory:
    """Per-player receiving/rushing history indexed for leak-free lookups by kickoff.

    Receiving (targets, receiving EPA) and rushing (carries, rushing EPA) are kept apart so a
    player is measured against baselines in his own opportunity mix. Only WR/TE/RB rows with a
    target or carry are read.
    """

    def __init__(self, player_stats: pl.DataFrame | None, kickoff: dict):
        rows: dict[str, list[tuple]] = defaultdict(list)
        # group -> season -> player -> [receiving EPA, targets, rushing EPA, carries]
        player_seasons: dict[str, dict[int, dict[str, list[float]]]] = {
            g: defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])) for g in SKILL_STAT_GROUPS}
        if player_stats is not None:
            # only WR/TE/RB rows with a target or carry matter; most of the table is defenders
            skill_rows = player_stats.filter(
                pl.col("position_group").is_in(sorted(SKILL_STAT_POSITIONS))
                & ((pl.col("targets").fill_null(0) + pl.col("carries").fill_null(0)) > 0))
            for r in skill_rows.iter_rows(named=True):
                ko = kickoff.get(r["game_id"])
                t, c = r["targets"] or 0, r["carries"] or 0
                if ko is None:
                    continue
                vals = (r["receiving_epa"] or 0.0, t, r["rushing_epa"] or 0.0, c)
                rows[r["player_id"]].append((ko, r["season"], vals))
                for g, stat_groups in SKILL_STAT_GROUPS.items():
                    if r["position_group"] in stat_groups:
                        tot = player_seasons[g][r["season"]][r["player_id"]]
                        for i, v in enumerate(vals):
                            tot[i] += v
        self._player_seasons = player_seasons
        self._index: dict[str, tuple[list, list, list[tuple]]] = {}
        for pid, hist in rows.items():
            hist.sort(key=lambda h: h[0])
            cum = [(0.0, 0.0, 0.0, 0.0)]
            for _, _, vals in hist:
                cum.append(tuple(a + b for a, b in zip(cum[-1], vals)))
            self._index[pid] = ([h[0] for h in hist], [h[1] for h in hist], cum)
        self._baseline: dict[tuple[str, int], tuple[float, float, float, float, float]] = {}

    def baseline(self, group: str, season: int) -> tuple[float, float, float, float, float]:
        """Group baselines over all seasons before `season`: (mean receiving EPA/target, mean rushing
        EPA/carry, replacement receiving, replacement rushing, targets share of opportunities).

        Replacement is the REPLACEMENT_PERCENTILE of player-season EPA per target (per carry) among
        player-seasons with >= REPLACEMENT_MIN_OPPS targets (carries); the mean when none qualify.
        """
        key = (group, season)
        if key not in self._baseline:
            earlier = [v for s, players in self._player_seasons[group].items() if s < season
                       for v in players.values()]
            e_rec, t, e_rush, c = (sum(v[i] for v in earlier) for i in range(4))
            mean_rec = e_rec / t if t else REPLACEMENT_FALLBACK
            mean_rush = e_rush / c if c else REPLACEMENT_FALLBACK

            def replacement(epa_i: int, n_i: int, mean: float) -> float:
                rates = [v[epa_i] / v[n_i] for v in earlier if v[n_i] >= REPLACEMENT_MIN_OPPS]
                return float(np.percentile(rates, REPLACEMENT_PERCENTILE)) if rates else mean

            self._baseline[key] = (mean_rec, mean_rush, replacement(0, 1, mean_rec),
                                   replacement(2, 3, mean_rush), t / (t + c) if t + c else 0.5)
        return self._baseline[key]

    def quality(self, player: str, group: str, season: int, ko, params: AvailabilityParams) -> float:
        mean_rec, mean_rush, repl_rec, repl_rush, group_mix = self.baseline(group, season)
        e_rec = t = e_rush = c = 0.0
        if player in self._index:
            kos, seasons, cum = self._index[player]
            hi = bisect_left(kos, ko)
            lo = min(bisect_left(seasons, season - SKILL_WINDOW_SEASONS + 1), hi)
            e_rec, t, e_rush, c = (a - b for a, b in zip(cum[hi], cum[lo]))
        mix = t / (t + c) if t + c else group_mix  # no history: the group's average mix
        prior = mix * mean_rec + (1 - mix) * mean_rush
        repl = mix * repl_rec + (1 - mix) * repl_rush
        k = params.prior_opps
        shrunk = (k * prior + e_rec + e_rush) / (k + t + c) if k + t + c > 0 else prior
        return max(0.0, 1.0 + params.quality_scale * (shrunk - repl))


class _ContractQuality:
    """Leak-free contract quality: 0.5 + the percentile of a player's latest contract signed in an
    earlier season within his position group's apy_cap_pct over contracts signed in earlier seasons
    (empirical CDF, ties take the mid-rank); else a draft-bucket default. Always in [0.5, 1.5]."""

    def __init__(self, contracts: pl.DataFrame | None, player_info: dict):
        self._player_info = player_info
        self._players: dict[str, tuple[list[int], list[float]]] = {}
        self._groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self._sorted: dict[tuple[str, int], list[float]] = {}  # (group, season) -> sorted apys
        if contracts is None:
            return
        rows = contracts.filter((pl.col("year_signed") > 0) & pl.col("apy_cap_pct").is_not_null()
                                & pl.col("gsis_id").is_not_null()).sort("year_signed", "apy_cap_pct")
        per_player: dict[str, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))
        for r in rows.iter_rows(named=True):
            years, apys = per_player[r["gsis_id"]]
            years.append(r["year_signed"])
            apys.append(r["apy_cap_pct"])
            group = contract_position_group(r["position"])
            if group:
                self._groups[group].append((r["year_signed"], r["apy_cap_pct"]))
        self._players = dict(per_player)

    def _distribution(self, group: str, season: int) -> list[float]:
        key = (group, season)
        if key not in self._sorted:
            self._sorted[key] = sorted(apy for year, apy in self._groups.get(group, ()) if year < season)
        return self._sorted[key]

    def __call__(self, gsis: str, group: str, season: int) -> float:
        if gsis in self._players:
            years, apys = self._players[gsis]
            i = bisect_left(years, season)  # contracts signed before the season; same year: highest apy
            dist = self._distribution(group, season)
            if i and dist:
                apy = apys[i - 1]
                mid_rank = (bisect_left(dist, apy) + bisect_right(dist, apy)) / 2
                return 0.5 + mid_rank / len(dist)
        info = self._player_info.get(gsis)
        return DEFAULT_CONTRACT_QUALITY[draft_bucket(info["draft_round"] if info else None)]


class _SnapIndex:
    """Per team-game non-QB snap shares (gsis ids) with each team's games ordered by kickoff."""

    def __init__(self, games: pl.DataFrame, snaps: pl.DataFrame, players: pl.DataFrame, lookback: int):
        pfr_to_gsis = {r["pfr_id"]: r["gsis_id"] for r in players.iter_rows(named=True) if r["pfr_id"]}
        self.kickoff = {r["game_id"]: r["kickoff_utc"] for r in games.iter_rows(named=True)}
        self.lookback = lookback
        # (team, game) -> [(gsis, offense share, defense share, snap position)]
        self.team_game: dict[tuple[str, str], list[tuple[str, float, float, str | None]]] = defaultdict(list)
        for r in snaps.filter(pl.col("position").ne_missing("QB")).iter_rows(named=True):
            gsis = pfr_to_gsis.get(r["pfr_player_id"])
            if gsis and r["game_id"] in self.kickoff:
                self.team_game[(r["team"], r["game_id"])].append(
                    (gsis, r["offense_pct"] or 0.0, r["defense_pct"] or 0.0, r["position"]))
        history: dict[str, list[str]] = defaultdict(list)
        for team, game_id in sorted(self.team_game, key=lambda k: self.kickoff[k[1]]):
            history[team].append(game_id)
        self._history = history
        self._kickoffs = {t: [self.kickoff[g] for g in gids] for t, gids in history.items()}

    def prior(self, team: str, ko) -> list[str]:
        """The team's games with snap data kicking off before `ko`, oldest first."""
        return self._history.get(team, [])[:bisect_left(self._kickoffs.get(team, []), ko)]

    def candidates(self, team: str, prior: list[str]) -> dict[str, tuple[float, float, str | None]]:
        """Players with >= MIN_APPEARANCES in the last `lookback` prior games ->
        (mean offense share, mean defense share, latest snap position) over those appearances."""
        shares: dict[str, list[list[float]]] = defaultdict(lambda: [[], []])
        latest_pos: dict[str, str] = {}
        for gid in prior[-self.lookback:]:
            for gsis, off, de, pos in self.team_game[(team, gid)]:
                shares[gsis][0].append(off)
                shares[gsis][1].append(de)
                if pos:
                    latest_pos[gsis] = pos
        return {g: (sum(o) / len(o), sum(d) / len(d), latest_pos.get(g))
                for g, (o, d) in shares.items() if len(o) >= MIN_APPEARANCES}


def _report_lists(injuries: pl.DataFrame):
    """Reported team-weeks (at least one row with a game-status designation; practice-only rows from
    early in the week do not count), Out/Doubtful sets and Questionable {gsis: practice bucket} per team-week."""
    reported = {(r["season"], r["week"], r["team"])
                for r in injuries.filter(pl.col("report_status").is_not_null())
                .select("season", "week", "team").unique().iter_rows(named=True)}
    out_lists: dict[tuple[int, int, str], set[str]] = defaultdict(set)
    q_lists: dict[tuple[int, int, str], dict[str, str | None]] = defaultdict(dict)
    rows = injuries.filter(pl.col("report_status").is_in([*OUT_STATUSES, QUESTIONABLE]))
    for r in rows.iter_rows(named=True):
        key = (r["season"], r["week"], r["team"])
        if r["report_status"] in OUT_STATUSES:
            out_lists[key].add(r["gsis_id"])
        else:
            q_lists[key][r["gsis_id"]] = r.get("practice_status")
    for key, q in q_lists.items():  # listed Out/Doubtful wins over a Questionable row
        for gsis in out_lists.get(key, ()):
            q.pop(gsis, None)
    return reported, out_lists, q_lists


def _questionable_counts(games: pl.DataFrame, index: _SnapIndex, injuries: pl.DataFrame
                         ) -> dict[tuple[int, str | None], list[int]]:
    """(season, practice bucket) -> [Questionable starters, of them with 0 snaps in the game].

    A case is a starter (same rule as the features) listed Questionable, not Out/Doubtful, in a
    team-game with snap data; he missed it when he has no snap row or 0 offense and defense share.
    """
    _, _, q_lists = _report_lists(injuries)
    counts: dict[tuple[int, str | None], list[int]] = defaultdict(lambda: [0, 0])
    for g in games.filter(pl.col("season") >= FIRST_SNAP_SEASON).iter_rows(named=True):
        for team in (g["home_team"], g["away_team"]):
            q = q_lists.get((g["season"], g["week"], team))
            played = index.team_game.get((team, g["game_id"]))
            if not q or not played:
                continue
            cands = index.candidates(team, index.prior(team, g["kickoff_utc"]))
            snapped = {gsis for gsis, off, de, _ in played if off + de > 0}
            for gsis, bucket in q.items():
                c = cands.get(gsis)
                if c is None or max(c[0], c[1]) < STARTER_SHARE:
                    continue
                n = counts[(g["season"], bucket)]
                n[0] += 1
                n[1] += gsis not in snapped
    return counts


def questionable_cases(games: pl.DataFrame, snaps: pl.DataFrame, injuries: pl.DataFrame,
                       players: pl.DataFrame, lookback: int = 4) -> pl.DataFrame:
    """Per season and practice bucket: Questionable starters (`cases`) and how many played 0 snaps (`missed`)."""
    counts = _questionable_counts(games, _SnapIndex(games, snaps, players, lookback), injuries)
    return pl.DataFrame([{"season": s, "practice_status": b, "cases": n, "missed": m}
                         for (s, b), (n, m) in sorted(counts.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))],
                        schema={"season": pl.Int64, "practice_status": pl.Utf8, "cases": pl.Int64,
                                "missed": pl.Int64})


class _MissRates:
    """Leak-free P(0 snaps | Questionable starter, practice bucket) from seasons before the game's."""

    def __init__(self, counts: dict[tuple[int, str | None], list[int]]):
        self._counts = counts
        self._cache: dict[tuple[int, str | None], float] = {}

    def __call__(self, season: int, bucket: str | None) -> float:
        key = (season, bucket)
        if key not in self._cache:
            n = sum(c[0] for (s, b), c in self._counts.items() if s < season and b == bucket)
            m = sum(c[1] for (s, b), c in self._counts.items() if s < season and b == bucket)
            self._cache[key] = m / n if n >= MIN_QUESTIONABLE_CASES else QUESTIONABLE_FALLBACK.get(bucket, 0.2)
        return self._cache[key]


def compute_availability(
    games: pl.DataFrame,
    snaps: pl.DataFrame,
    injuries: pl.DataFrame,
    players: pl.DataFrame,
    player_stats: pl.DataFrame | None = None,
    params: AvailabilityParams = AvailabilityParams(),
    lookback: int = 4,
    contracts: pl.DataFrame | None = None,
) -> pl.DataFrame:
    player_info = {r["gsis_id"]: r for r in players.iter_rows(named=True)}
    contract_quality = _ContractQuality(contracts, player_info) if params.mode == CONTRACT_MODE else None
    index = _SnapIndex(games, snaps, players, lookback)
    kickoff = index.kickoff
    team_game_snaps = index.team_game
    values = params.mode in VALUE_MODES or params.mode == CONTRACT_MODE
    skill = _SkillHistory(player_stats, kickoff) if params.mode in VALUE_MODES else None
    reported, out_lists, q_lists = _report_lists(injuries)
    miss_rate = _MissRates(_questionable_counts(games, index, injuries)) if params.questionable else None

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
        if contract_quality is not None:
            return contract_quality(gsis, group, game["season"])
        if group in SKILL_STAT_GROUPS:
            return skill.quality(gsis, group, game["season"], game["kickoff_utc"], params)
        info = player_info.get(gsis)
        base = params.lineman_weight(draft_bucket(info["draft_round"] if info else None))
        w = params.durability_weight
        return max(0.0, base * (1 - w + w * durability(team, prior, gsis, group in OFFENSE_GROUPS)))

    def team_out(team: str, game: dict) -> tuple[float, float, dict[str, float]] | None:
        if game["season"] < FIRST_SNAP_SEASON:
            return None
        key = (game["season"], game["week"], team)
        if key not in reported:
            return None
        prior = index.prior(team, game["kickoff_utc"])
        if not prior:
            return None
        out_set = out_lists.get(key, set())
        q_set = q_lists.get(key, {}) if params.questionable else {}
        off_out = def_out = 0.0
        groups = dict.fromkeys(GROUPS, 0.0)
        for gsis, (off_avg, def_avg, pos) in index.candidates(team, prior).items():
            if gsis in out_set:
                weight = 1.0
            elif gsis in q_set:
                weight = miss_rate(game["season"], q_set[gsis])
            else:
                continue
            if off_avg >= STARTER_SHARE:
                off_out += weight * off_avg
            if def_avg >= STARTER_SHARE:
                def_out += weight * def_avg
            info = player_info.get(gsis)
            group = position_group(pos) or position_group(info["position"] if info else None)
            if group is None:
                continue
            share = off_avg if group in OFFENSE_GROUPS else def_avg
            if share >= STARTER_SHARE:
                groups[group] += weight * share * (quality(team, prior, gsis, group, game) if values else 1.0)
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
