# Seahawks Game Prediction Model — Design

**Date:** 2026-10-06
**Status:** v3 — approved; revised after verifying against real data (see "Revisions after data verification")

## Goal

Predict the outcome of each upcoming Seattle Seahawks game using only free, public data, and publish the prediction plus the model's track record on a static dashboard updated automatically by GitHub Actions.

The project is model-focused. The Vegas line is used only as a **benchmark** to measure the model against — never as a model input.

## Decisions

| Topic | Decision |
|---|---|
| Output (phase 1) | Seahawks win probability, plus predicted point margin with an interval |
| Training target | Home-team point margin (regression) |
| Margin → probability | Empirical conditional margin distribution **or** Gaussian CDF (σ from out-of-fold residuals), optionally ensembled with a binary win/loss classifier — all chosen by backtest |
| Training data | All NFL regular-season and playoff games, not only Seahawks games |
| Model selection | Regularized linear model vs. LightGBM, chosen by walk-forward backtest |
| Market line | Benchmark only (spread converted to probability); not a feature |
| Delivery | Static dashboard on GitHub Pages, rebuilt by scheduled GitHub Actions |
| Compute split | Hyperparameter tuning and full backtests run locally; CI only refits with locked hyperparameters |
| Cost | $0 — no paid APIs, no API keys required |

## Data Sources

| Source | What we use | Coverage |
|---|---|---|
| nflverse via `nflreadpy` | Schedules/results (kickoff time, roof, rest, starting/projected QBs, head coaches, spread line), play-by-play with EPA, injury reports, snap counts, players (draft round, PFR↔GSIS ids) | PBP 1999+, injuries 2009+, snaps 2013+ |
| Open-Meteo archive API | Hourly historical weather at stadium coordinates for kickoff hour | Training |
| Open-Meteo forecast API | Forecast weather at kickoff hour for the upcoming game | Prediction |
| `data/stadiums.csv` (hand-maintained) | Stadium ID → lat/lon, time zone, roof type, surface, elevation; includes international venues | All |

Season ranges:
- Team and QB ratings are computed from 2002 onward (32-team era), so 2009 starts with warmed-up priors.
- Model training rows: **2009+** (first season with injury data), ~4,600 games through 2025.

## Leakage Rule

Every feature for a game uses only information available before that game's kickoff. Concretely:
- Rolling stats include only games strictly before the target game.
- Injury/availability features use the final pre-game injury report, not post-game snap counts for that game.
- Season-level priors use only prior seasons.
- Anything fit on predictions (margin distribution, calibration, σ, ensemble weights) is fit only on data from training seasons of the current fold (see Evaluation).

This is enforced by tests (see Testing).

## Features

All matchup features are expressed as **home minus away** differences (or as home/away pairs where a difference doesn't make sense), so the model is symmetric and doesn't learn artifacts of listing order. Neutral-site games set the home-field indicator to 0.

### Team strength
- **Offensive and defensive EPA/play ratings.** Each team starts the season from its prior-season rating regressed toward league average; current-season games are blended in as they accumulate (weight on current season grows with games played). Shrinkage weights are tuned in backtest.
- **Elo rating**, computed in-house from game results with margin-of-victory adjustment and preseason regression to the mean.

### Regime change
- **`new_head_coach`** (per team, binary): 1 if the team's head coach for this game differs from its head coach in the final game of the previous season. Source: nflverse schedule coach fields.
- Used two ways:
  1. As a model feature (home/away pair).
  2. In the team-strength shrinkage: teams with a new head coach get a separate, tuned prior weight, so their prior-season rating is discounted faster as current-season games accumulate.
- Mid-season coaching changes are not flagged in phase 1.

### Quarterback
- **Starting-QB rating**: EPA/play + CPOE for the expected starter, shrunk toward a prior based on sample size. When a backup starts, the team's rating uses the backup's QB value, not the team's season average.
- **`draft_capital`** seeds the QB prior: `round_1`, `day_2` (rounds 2–3), or `day_3_udfa` (rounds 4–7 and undrafted). Each bucket's prior mean and strength is estimated from historical early-career QB performance, using only seasons in the current training fold. This matters most for rookies and low-sample backups, whose rating is mostly prior. Source: nflverse draft picks joined to rosters.
- `draft_capital` (of each team's expected starter) is also included directly as a categorical feature.
- Expected starter comes from the nflverse schedule QB fields, which are populated with the projected starter for upcoming games too.

### Availability (phase-1 version)
- Count of starters listed **Out** or **Doubtful** on that week's injury report, weighted by snap share. A starter is a non-QB averaging ≥50% of offense or defense snaps over the team's previous 4 games (min. 2 appearances). Reported separately for offense and defense.
- Snap counts exist from 2013; earlier games get 0 plus `availability_known = 0`.

### Situational
- Rest days (difference), post-bye flag, short-week flag (e.g., Thursday after Sunday).
- Away-team travel distance (great-circle miles) and time zones crossed.
- **Body clock**: kickoff time expressed in each team's home time zone (captures West Coast teams at 10am PT, East Coast teams at night in the West).
- Primetime flag, divisional-game flag.
- Home-field indicator, with home-field advantage allowed to drift over time (via recency weighting and a season-trend interaction; 2020 flagged as no-crowd season).

### Season timing
- **`week_number`**: regular-season week (playoff games get a separate `is_playoff` flag).
- **`is_final_regular_week`**: 1 for the last regular-season week (Week 17 through 2020, Week 18 from 2021). Captures teams resting starters or evaluating depth.
- Clinch/elimination status is out of scope for phase 1 (see Future Phases).

### Venue and weather
- Roof type (indoor/retractable-closed games get neutral weather values and an indoor flag), surface type.
- Game-window (kickoff hour + 3) mean temperature and wind, total precipitation, from Open-Meteo. Outdoor games with no cached hours fall back to stadium same-month climatology; `weather_source` records which was used.

Expectation: weather will likely carry little weight for win prediction (it matters more for totals). It's kept because it's cheap and a future score model will use it.

## Models

1. **Baseline: regularized linear regression** (ridge) on margin.
2. **LightGBM regressor** on margin with conservative settings (shallow trees, strong regularization), since the signal-to-noise ratio is low and data is ~4,600 rows.
3. **Recency weighting:** training rows are weighted by an exponential time decay; the half-life is tuned in backtest.

The model with the better walk-forward log-loss is promoted. If they're within noise, the linear model wins (simpler, more stable).

### Margin → win probability

NFL margins are multimodal (clustered at 3, 7, 10, 14), so a smooth Gaussian CDF misprices probability mass near key numbers. Candidate methods, compared in backtest:

- **A. Empirical conditional margin distribution (default).** For a predicted margin `m`, take historical games from the training fold, weight each by a kernel on `|predicted_margin_i − m|`, and use their **actual** margins as the outcome distribution. Then:
  `P(home win) = P(margin > 0) + 0.5 · P(margin = 0)`
  The kernel bandwidth is tuned in backtest. This also gives the margin interval directly.
- **B. Ensemble with a binary classifier.** A regularized logistic regression on the same features, trained on win/loss (ties as 0.5 targets via sample weights), averaged with method A. The ensemble weight is tuned in backtest.

- **C. Gaussian CDF.** `Φ(m/σ)`, σ from out-of-fold residuals. Kept as a candidate because on 2012–2023 real data it scored 0.629 log-loss vs 0.631–0.636 for method A (key numbers matter more for spreads than for win/loss).

The tuned backtest picks among A, C, and an ensemble with B; B is adopted only if it improves walk-forward log-loss beyond noise.

**Optional calibration:** if the backtest shows residual miscalibration, apply Platt scaling. The Platt model is fit **strictly inside each walk-forward fold** — only on out-of-fold predictions for that fold's training seasons (via an inner walk-forward over seasons < Y) — never globally on all out-of-fold predictions.

**Explanations:** per-prediction feature contributions — coefficient × value for the linear model, LightGBM's native `pred_contrib` for the tree model. No SHAP dependency needed.

## Evaluation

- **Walk-forward backtest:** for each season Y in 2012–2023, train on all seasons < Y, predict season Y. All tuning (shrinkage, decay half-life, hyperparameters, kernel bandwidth, ensemble weight) happens inside this loop.
- **Nested fitting for anything fit on predictions:** the empirical margin distribution, ensemble weight, and Platt scaling for fold Y are fit only on out-of-fold predictions generated by an inner walk-forward over seasons < Y.
- **Locked holdout:** seasons 2024–2025 are evaluated once, after model selection, and never used for tuning.
- **Metrics:** log-loss (primary), Brier score, accuracy, margin MAE, calibration curve.
- **Ties:** a tie is scored as an outcome of **0.5**:
  - Log-loss: `−[0.5·log(p) + 0.5·log(1−p)]`
  - Brier: `(p − 0.5)²`
  - Accuracy: ties are excluded from the denominator.
  - Margin MAE: actual margin = 0.
- **Baselines:**
  - Home team always wins, at the historical home win rate
  - Elo only
  - Vegas spread converted to probability with the same margin → probability method (benchmark ceiling; beating it is not expected)
- Metrics are reported on **all games**. Seahawks-only results are shown for interest but are not used for decisions (~17 games/season is too noisy). Metrics include bootstrap confidence intervals.

## Prediction Runs

Each upcoming Seahawks game gets up to three predictions, timed relative to that game's kickoff (so Thursday, Saturday, Monday, and international games work the same as Sunday games):

| Run | When | Purpose |
|---|---|---|
| `midweek` | ~4 days before kickoff (Wednesday for a Sunday game) | Early read with latest depth charts |
| `final_injury` | Day before kickoff (Saturday for a Sunday game) | After the final injury report |
| `gameday` | ~3 hours before kickoff | Final Open-Meteo forecast; wind and precipitation timing shift a lot after Saturday |

Each run:
1. Refreshes data for the current season and fetches the Open-Meteo forecast for kickoff hour.
2. Builds the feature row using the same feature code as training (no separate code path).
3. Loads the current promoted model and predicts margin, win probability, and an 80% margin interval.
4. **Appends** a record to `predictions/history.jsonl`. Records are never overwritten. Each record includes:
   - `game_id`, `run_type` (`midweek` / `final_injury` / `gameday`), `predicted_at` timestamp
   - `is_final_injury_report`: true if the run happened after the final injury report for that game was published
   - Data snapshot info: latest injury-report week seen for SEA, weather source (Open-Meteo doesn't expose a forecast issue time; `predicted_at` serves as the snapshot time)
   - Win probability, predicted margin, margin interval, top feature contributions
   - Model version
5. After the game, a results step appends the actual outcome (a separate `result` record keyed by `game_id`).

**Scoring rule:** the track record scores the **last pre-kickoff prediction** for each game (normally `gameday`). Earlier runs are kept for the trajectory display.

If the forecast isn't available yet (game > 16 days out), the weather feature uses the stadium's historical average for that week and the record is flagged.

## Automation (GitHub Actions)

| Workflow | Schedule | Does |
|---|---|---|
| `predict.yml` | Cron every 15 minutes during the season (Sep–Feb) + manual trigger | A cheap gate step checks the next Seahawks kickoff. If a `midweek`, `final_injury`, or `gameday` run is due and hasn't run yet, it predicts → updates results → builds site → commits → Pages deploys. Otherwise exits in seconds. |
| `retrain.yml` | Tuesday weekly during season + manual trigger | Refresh data → rebuild features → refit model with locked hyperparameters → commit model + metrics |

Notes on the gate:
- GitHub cron can run late or skip runs under load, so each run type has a window (e.g., `gameday` = between 6h and 75 min before kickoff) and whichever scheduled run lands in the window first does it. Records note the actual `predicted_at` time.
- Season-off months: the cron is limited to Sep–Feb, and the gate exits if there's no upcoming game.

### Compute limits

The full walk-forward backtest with hyperparameter search (nested folds × hyperparameter grid × two model families) is too heavy for CI's 6-hour job limit and standard runner memory. So:

- **Locally (offseason or manually):** `backtest` runs the full walk-forward tuning and the holdout evaluation. It writes the chosen hyperparameters, shrinkage weights, decay half-life, kernel bandwidth, and ensemble weight to `models/config.json`, plus backtest metrics to `models/backtest.json`. Both are committed.
- **In CI (during the season):** `retrain` reads the locked `models/config.json` and only refits final model weights on all completed games, plus the empirical margin distribution / calibration it needs (a handful of refits with fixed settings — minutes, not hours). No hyperparameter search runs in CI.
- **Memory:** only the current season's raw data is re-downloaded in CI; completed seasons' derived features are committed as a parquet file and reused. Play-by-play is read lazily with only the needed columns.

Other automation rules:
- Compact caches (per-season pbp aggregates, injuries, snaps, game-hour weather) live in `data/cache/` and are committed, so CI only refreshes the current season and never re-downloads the weather archive.
- Committed: data cache, derived feature table, trained model artifact, `config.json`, metrics JSON, prediction history. The site is built to `public/` and deployed as a GitHub Pages artifact (so `docs/` stays documentation-only).
- If any data fetch fails, the workflow fails without publishing; the dashboard always shows a "last updated" timestamp.

## Dashboard

Static HTML + JSON in `public/`, deployed to GitHub Pages:
1. **Next game:** opponent, kickoff, venue, weather; Seahawks win probability, predicted margin ± interval; Vegas spread-implied probability shown as a benchmark.
2. **Weekly trajectory:** win probability across `midweek` → `final_injury` → `gameday` runs, with a short note on what changed between runs (e.g., injury status changes, forecast shifts).
3. **Why:** top feature contributions for the latest prediction.
4. **Season log:** every 2026 Seahawks game — scored (last pre-kickoff) prediction vs. actual result, with the trajectory available per game.
5. **Model report:** backtest metrics vs. baselines, calibration plot, holdout results.

## Repository Layout

```
pyproject.toml                 # uv-managed; Python 3.12 in CI
data/stadiums.csv
seahawks_ml/
  ingest/      nflverse.py, weather.py, stadiums.py
  features/    ratings.py (EPA + shrinkage), qb.py (incl. draft capital), elo.py,
               coaching.py, situational.py, season_timing.py, availability.py,
               weather.py, build.py
  models/      train.py, margin_dist.py, calibrate.py, backtest.py,
               predict.py, results.py
  pipeline/    gate.py (decides which run, if any, is due)
  site/        build.py, templates/
predictions/history.jsonl
models/        # promoted model artifact, config.json, backtest.json, metrics.json
docs/          # built dashboard (GitHub Pages)
tests/
.github/workflows/predict.yml, retrain.yml
```

Each `features/*.py` module exposes one function taking the games table (plus raw inputs) and returning per-game feature columns, so features are independently testable and the build step just joins them.

Stack: polars (nflreadpy's native format), scikit-learn, lightgbm, httpx for Open-Meteo, Jinja2 for the site, a CDN chart library for the dashboard.

## Testing

- **Leakage tests:** for sampled games, recompute features with all data after that game's kickoff removed; results must be identical to the full-data features. Also check that the margin distribution and calibration for fold Y never see season Y data.
- **Unit tests:** Elo update, rating shrinkage (including new-coach path), QB draft-capital prior, travel distance/time zones, body-clock calculation, availability weighting, final-regular-week detection across the 2021 schedule change, empirical margin → probability (including tie mass), tie handling in metrics.
- **Gate tests:** given a schedule and current time, the gate picks the correct run type (or none) for Sunday, Thursday, Monday, Saturday, and international kickoffs, and doesn't repeat a run already in `history.jsonl`.
- **Schema tests:** feature table has expected columns, types, no unexpected nulls, one row per game; `history.jsonl` records match the schema.
- **Smoke test:** full pipeline (ingest from fixtures → features → train with a fixed config → predict → site build) on a small fixture dataset, run in CI on every push.

## Out of Scope (Phase 1)

- Full player-value model (all positions)
- Score/total predictions
- Clinch/elimination status for late-season motivation
- Mid-season coaching change detection
- Live or historical line snapshots, betting-specific metrics
- Seahawks-specific model or weighting

## Future Phases

- **Phase 2:** player-value ratings (snap share × per-play contribution by position) replacing the simple availability count; clinch/elimination status for Weeks 15–18.
- **Phase 3:** team-score model (points for each team), yielding totals and score distributions where weather should matter more.

## Revisions after data verification (2026-10-06)

Checked against `nflreadpy 0.1.5`, Open-Meteo, and a full real-data dry run before planning:

1. **Margin → probability:** Gaussian added as a tuned candidate alongside the empirical distribution (it scored better in the dry run). The empirical method uses all inner out-of-fold seasons, not 5.
2. **Depth charts dropped:** nflverse changed the depth-chart format in 2025. Starters come from snap shares; expected QB from the schedule.
3. **Snap counts start in 2013**, so availability is unknown for 2009–2012 (flagged, not dropped).
4. **Weather:** 4-hour game window plus climatology fallback; archive fetched in paced two-week windows (Open-Meteo rate limits), 2009+ only.
5. **Caching/Pages:** compact caches committed instead of `actions/cache`; site deployed as a Pages artifact from `public/`.
6. **Dry-run results (2012–2023 walk-forward, partial weather):** model 0.629 log-loss, Elo 0.633, home-always 0.686, Vegas 0.613.

Implementation plans: `docs/superpowers/plans/2026-10-06-plan-{1,2,3}-*.md`.
