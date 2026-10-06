# Player Values — Design

**Date:** 2026-10-06
**Status:** Approved by owner ("do it all"); decisions below made by the controller.

Replace the flat "starters out" count with injury features that know *which* players are missing and *how good* they are. Two steps; each is kept only if the walk-forward backtest (2012–2023) improves, and both are then judged on the live league scorecard.

## Baseline (measured 2026-10-06, 2014–2023 walk-forward)
Full model 0.6299 log-loss; without the current availability feature 0.6304; without QB ratings 0.6331. Injuries matter in 42% of games (a team missing ≥1 starter-weight).

## Leakage rules (same as everywhere)
Only information available before kickoff: snap shares and player stats from earlier games, the injury report for the game's own week, and draft position (known at draft time). **Not allowed:** career Approximate Value (`car_av`, `w_av`) from `load_draft_picks` — it is computed as of today and includes future seasons.

## Step 1 — position groups
- Group players by snap-count position: OL (T, G, C), WR/TE (WR, TE), RB (RB, FB), DL (DE, DT, NT), LB (LB), DB (CB, S, SS, FS, DB). QB excluded (QB ratings handle starters); K/P/LS excluded.
- For each team-game: snap-share-weighted starters Out/Doubtful per group (same starter definition as today: ≥50% share over the previous 4 team games, ≥2 appearances, offense share for offensive groups, defense share for defensive groups).
- Features: home-minus-away per group: `out_ol_diff, out_wrte_diff, out_rb_diff, out_dl_diff, out_lb_diff, out_db_diff`. The old four columns stay available for comparison.

## Step 2 — player quality weights
Replace each missing starter's weight (snap share) with `snap_share × quality`, where quality is a shrunk per-player value relative to a replacement-level player at that group:
- **WR/TE and RB:** EPA per opportunity (targets for receiving, carries for rushing; nflverse weekly `player_stats`) over the player's games in the previous 2 seasons up to (not including) the game, shrunk toward the group mean with a prior of `k` opportunities; quality = 1 + scale × (shrunk EPA/opp − group replacement EPA/opp), floored at 0.
- **OL, DL, LB, DB:** no free individual grades. Quality from (a) draft capital bucket (round 1 / rounds 2–3 / later / undrafted, from `load_players().draft_round`) and (b) durability of the starting role (share of the team's snaps over the previous 8 team games). Combined as a small fixed lookup, tuned by backtest.
- Features: same six per-group columns, now quality-weighted ("value out").
- `k`, the scale factor, and the lineman lookup are tuned in the backtest feature stage.

## Configuration and evaluation
- `AvailabilityParams.mode ∈ {"count", "groups", "values"}` in `ProjectConfig` (old configs → "count"). Unused columns are 0.0 so the column list is fixed.
- Backtest stage 0c compares the three modes (plus a small grid for "values"); winner is locked. Holdout stays "seen before"; the live league scorecard is the honest test.
- New cache: `data/cache/player_stats_{season}.parquet` (player_id, game_id, season, week, team, position_group, targets, receiving_epa, carries, rushing_epa) for 2012+ (current season refreshed like other tables).

## Dashboard
- Factor labels for the new columns ("Offensive line value out", …).
- Seahawks prediction records gain nothing new; the "Why" factors and "What changed" notes pick up the new columns automatically.
