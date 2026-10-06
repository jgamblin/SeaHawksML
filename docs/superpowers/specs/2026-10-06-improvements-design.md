# Improvements Round 1 — Design

**Date:** 2026-10-06
**Status:** Approved by owner ("do all of that"); decisions below made by the controller.

Four pieces, built in this order. Each piece ships with tests; nothing changes the append-only rule for prediction logs.

## 1. League-wide live predictions + scorecard

**Why:** The 2024–25 holdout has been seen, so further model changes need fresh, honest evaluation. Seattle alone gives ~17 games/season; the whole league gives ~285.

- New append-only log `predictions/league.jsonl`. One **prediction record per game**, made once, in the same window as the Seahawks `final_injury` run: the first hourly run where `kickoff − 24h ≤ now < kickoff − 6h`. Games already logged are skipped.
- Record keys (type `league_prediction`): `game_id, season, week, predicted_at, kickoff_utc, home_team, away_team, p_home, margin_home, p_vegas_home (nullable), p_elo_home, model_version`. Result records (type `league_result`): `game_id, recorded_at, home_score, away_score, margin_home`.
- The predict workflow runs league predictions in the same job as the Seahawks gate (one data load per run). If neither a Seahawks run nor a league run is due, the job still exits in seconds.
- Forecast weather is fetched for every outdoor game being predicted (Open-Meteo, one request per stadium).
- Dashboard: a "Live this season" card — model vs Vegas vs Elo log-loss, Brier, accuracy on completed league games (ties = 0.5, as everywhere), with game count and a cumulative log-loss-by-week chart. Shows an empty state until games complete.

## 2. Better team ratings

All leak-free (only data before kickoff). Each addition is kept only if the walk-forward backtest (2012–2023) improves; the decision is recorded in `models/backtest.json`.

- **Opponent adjustment:** when a game is added to a team's running ratings, its offensive EPA/play is adjusted by subtracting the opponent's *pre-game* defensive rating (and defense by the opponent's pre-game offensive rating). Toggle: `RatingParams.opponent_adjust` (tuned in stage 0).
- **Pass/rush split:** separate offense/defense EPA-per-play ratings for dropbacks and designed runs (same shrinkage scheme). New feature columns `pass_off_diff, pass_def_diff, rush_off_diff, rush_def_diff`.
- **Success rate:** offense/defense success-rate ratings (nflverse `success`), same scheme. New columns `success_off_diff, success_def_diff`.
- **Fumble luck:** in pbp aggregation, plays with a fumble use that season's league-average EPA for fumble plays instead of their actual EPA, so recovery luck (≈50/50) doesn't move ratings.
- Requires rebuilding the per-season pbp caches.
- **Holdout:** the 2024–25 holdout was already viewed once with the v1 model. After this change it is re-run with `--force` and the JSON gets `"previously_viewed": true`; the dashboard labels it "seen before — reference only" and points to the live scorecard as the honest number.

## 3. Seahawks season simulation

- Simulate the rest of the regular season 10,000 times. Every remaining game's home-win probability comes from the current model (features as of now; unknown future QBs fall back to each team's latest starter; weather = climatology/indoor). Completed games use actual results. Games are simulated independently (team strength not resampled — stated on the dashboard).
- Standings: division and conference from `nflreadpy.load_teams()`. Seeding uses win percentage, then head-to-head, then division win%, then conference win%, then a coin flip — **simplified tiebreakers**, labelled as such. 7 playoff teams per conference (4 division winners seeded 1–4, then 3 wild cards).
- Outputs for SEA: mean/median wins and an 80% range, win-total distribution, P(playoffs), P(division), P(#1 seed). Appended to `predictions/season_sim.jsonl` (one snapshot per day at most, with `as_of`), so the dashboard can chart playoff odds over the season.
- Runs whenever the predict job records anything, and at least once per day during the season.

## 4. Quality of life

- Season log shows **every** 2026 Seahawks game: unpredicted past games show the result with "not predicted", future games show "scheduled".
- Dependabot for GitHub Actions (weekly).
- Failure alerts: on failure of a scheduled workflow, open (or comment on) a GitHub issue labelled `workflow-failure`.
- Offseason keepalive: a monthly workflow that calls the GitHub API to re-enable the Predict, Retrain and Keepalive workflows, which resets the 60-day inactivity timer without commits.

## Out of scope

Team-points/totals model, player-value model (later phases).
