# New Data Sources — Design

**Date:** 2026-10-12 (built 2026-10-13)
**Status:** Approved by owner ("build all four"); decisions below made by the controller.

Four free additions, each behind a config toggle that defaults OFF (old configs keep their meaning) and each added to the `backtest --tune-features` stage-0 grid so the walk-forward backtest (2012–2023) decides. Improvements under ~0.0003 log-loss are treated as ties and the simpler option wins. All features use only information available before kickoff and are covered by the `RawData.as_of` leakage test.

Baseline: model 0.6277 walk-forward log-loss (Elo 0.6327, Vegas 0.6130); availability mode `count_diff`.

## 1. Special-teams rating
- Ratings use only pass/run plays today. Add per team-game special-teams value from play-by-play plays with `play_type` in {field_goal, punt, kickoff, extra_point}. nflverse EPA is from the possession team's perspective, so a team's ST value in a game = Σ EPA on ST plays where it has possession − Σ EPA on the opponent's ST plays.
- Shrunk prior/blend rating like the other ratings (`st_rating`), feature `st_diff` (home − away).
- Toggle `RatingParams.special_teams` (default False → column 0.0).

## 2. Questionable + practice status
- Today only Out/Doubtful starters count. Add Questionable starters weighted by their probability of not playing given Friday-or-last practice status (DNP / Limited / Full / no practice data).
- Probabilities estimated leak-free from seasons strictly before the game's season: share of Questionable starters in each practice-status bucket who then logged 0 snaps in that game (snap counts, 2013+). Fixed fallbacks when history is thin: DNP 0.6, Limited 0.25, Full 0.05, unknown 0.2.
- Folded into the pooled `count_diff` offense/defense diffs as expected starter weight missing.
- Toggle `AvailabilityParams.questionable` (default False).

## 3. Contract value as player quality
- nflverse `load_contracts()`: for each player and game, the most recent contract with `year_signed` strictly before the game's season → `apy_cap_pct`. Quality is a bounded percentile scale: `0.5 + percentile`, where percentile is the player's `apy_cap_pct` rank within his position group's distribution of contracts with `0 < year_signed < season` (empirical CDF, ties → mid-rank), so quality ∈ [0.5, 1.5]. Leak-free: only earlier-season contracts enter either the player's value or the distribution; sorted arrays are precomputed per (season, group) and queried with `bisect`.
- Why not value ÷ group median (the first version, clipped to [0.2, 5]): minimum-salary deals dominate each group and `apy_cap_pct` is rounded to 0.001, so group medians are ~0.004 and ~40% of starters saturated at the cap — the feature only separated "real contract" from "rookie/minimum". The percentile scale keeps resolution across the whole distribution.
- Missing contract → draft-bucket default on the same scale (round 1 = 1.1, day 2 = 0.9, later = 0.75, undrafted = 0.6). Cached as `data/cache/contracts.parquet`.
- New availability mode `values_contract`: pooled offense/defense diffs (like `values_pooled`) with snap share × contract quality.

## 4. Altitude
- `elevation_m` column in `data/stadiums.csv` (Empower Field 1609 m, Estadio Azteca 2240 m, Phoenix venues ≈ 330–360 m, others from known values).
- Feature `away_altitude_gain_km` = max(0, venue elevation − away team's home elevation) / 1000 (the visiting team climbing; Denver home games ≈ 1.6).
- Toggle `SituationalParams.altitude` (default False → 0.0).
