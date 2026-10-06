# Seahawks Game Prediction Model — Design

**Date:** 2026-10-06
**Status:** Draft, awaiting review

## Goal

Predict the outcome of each upcoming Seattle Seahawks game using only free, public data, and publish the prediction plus the model's track record on a static dashboard updated automatically by GitHub Actions.

The project is model-focused. The Vegas line is used only as a **benchmark** to measure the model against — never as a model input.

## Decisions

| Topic | Decision |
|---|---|
| Output (phase 1) | Seahawks win probability, plus predicted point margin with an interval |
| Training target | Home-team point margin (regression), converted to win probability via `P(win) = Φ(margin / σ)` |
| Training data | All NFL regular-season and playoff games, not only Seahawks games |
| Model selection | Regularized linear model vs. LightGBM, chosen by walk-forward backtest |
| Market line | Benchmark only (spread converted to probability); not a feature |
| Delivery | Static dashboard on GitHub Pages, rebuilt by scheduled GitHub Actions |
| Cost | $0 — no paid APIs, no API keys required |

## Data Sources

| Source | What we use | Coverage |
|---|---|---|
| nflverse via `nflreadpy` | Schedules/results (kickoff time, roof, surface, rest, starting QBs, spread line), play-by-play with EPA, weekly player stats, injury reports, depth charts, snap counts, rosters | PBP 1999+, injuries 2009+ |
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

This is enforced by tests (see Testing).

## Features

All matchup features are expressed as **home minus away** differences (or as home/away pairs where a difference doesn't make sense), so the model is symmetric and doesn't learn artifacts of listing order. Neutral-site games set the home-field indicator to 0.

### Team strength
- **Offensive and defensive EPA/play ratings.** Each team starts the season from its prior-season rating regressed toward league average; current-season games are blended in as they accumulate (weight on current season grows with games played). Exact shrinkage weights are tuned in backtest.
- **Elo rating**, computed in-house from game results with margin-of-victory adjustment and preseason regression to the mean.

### Quarterback
- **Starting-QB rating**: EPA/play + CPOE for the expected starter, shrunk toward a replacement-level prior based on sample size. When a backup starts, the team's rating uses the backup's QB value, not the team's season average.
- Expected starter comes from nflverse schedule QB fields for historical games and the latest depth chart for the upcoming game.

### Availability (phase-1 version)
- Count of projected starters (from depth charts) listed **Out** or **Doubtful** on the final injury report, weighted by each player's snap share over the previous 4 games. Reported separately for offense and defense.

### Situational
- Rest days (difference), post-bye flag, short-week flag (e.g., Thursday after Sunday).
- Away-team travel distance (great-circle miles) and time zones crossed.
- **Body clock**: kickoff time expressed in each team's home time zone (captures West Coast teams at 10am PT, East Coast teams at night in the West).
- Primetime flag, divisional-game flag.
- Home-field indicator, with home-field advantage allowed to drift over time (via recency weighting and a season-trend interaction; 2020 flagged as no-crowd season).

### Venue and weather
- Roof type (indoor/retractable-closed games get neutral weather values and an indoor flag), surface type.
- Kickoff-hour temperature, wind speed, precipitation from Open-Meteo.

Expectation: weather will likely carry little weight for win prediction (it matters more for totals). It's kept because it's cheap and a future score model will use it.

## Models

1. **Baseline: regularized linear regression** (ridge) on margin.
2. **LightGBM regressor** on margin with conservative settings (shallow trees, strong regularization), since the signal-to-noise ratio is low and data is ~4,600 rows.
3. **Conversion to probability:** σ is estimated from out-of-fold residuals; `P(home win) = Φ(predicted_margin / σ)`. If the backtest shows miscalibration, apply Platt scaling fitted on out-of-fold predictions.
4. **Recency weighting:** training rows are weighted by an exponential time decay; the half-life is tuned in backtest.

The model with the better walk-forward log-loss is promoted. If they're within noise, the linear model wins (simpler, more stable).

**Explanations:** per-prediction feature contributions — coefficient × value for the linear model, LightGBM's native `pred_contrib` for the tree model. No SHAP dependency needed.

## Evaluation

- **Walk-forward backtest:** for each season Y in 2012–2023, train on all seasons < Y, predict season Y. All tuning (shrinkage, decay half-life, hyperparameters) happens inside this loop.
- **Locked holdout:** seasons 2024–2025 are evaluated once, after model selection, and never used for tuning.
- **Metrics:** log-loss (primary), Brier score, accuracy, margin MAE, calibration curve.
- **Baselines:**
  - Home team always wins, at the historical home win rate
  - Elo only
  - Vegas spread converted to probability with the same Φ mapping (benchmark ceiling; beating it is not expected)
- Metrics are reported on **all games**. Seahawks-only results are shown for interest but are not used for decisions (~17 games/season is too noisy). Metrics include bootstrap confidence intervals.

## Prediction Run (weekly)

1. Refresh nflverse data and fetch Open-Meteo forecast for the next Seahawks game's kickoff.
2. Build the feature row using the same feature code as training (no separate code path).
3. Load the current promoted model, predict margin, win probability, and an 80% margin interval.
4. Append the prediction to `predictions/history.jsonl` with a timestamp and the data snapshot date. **This file is append-only**, so the track record reflects what was predicted before kickoff.
5. After games complete, a results step fills in actual outcomes for scored predictions.

## Automation (GitHub Actions)

| Workflow | Schedule | Does |
|---|---|---|
| `predict.yml` | Wed and Sat (after final injury reports) during the season, plus manual trigger | Refresh data → predict next SEA game → update results → build site → commit → Pages deploy |
| `retrain.yml` | Tuesday weekly during season, plus manual trigger | Refresh data → rebuild features → retrain → run backtest → commit model + metrics |

- Raw nflverse/weather downloads are cached with `actions/cache`, not committed.
- Committed: the derived feature table (small parquet), the trained model artifact, metrics JSON, prediction history, and the built site in `docs/`.
- If any data fetch fails, the workflow fails without publishing; the dashboard always shows a "last updated" timestamp.
- If the forecast isn't available yet (game > 16 days out), the weather feature uses the stadium's historical average for that week and the dashboard flags it.

## Dashboard

Static HTML + JSON in `docs/`, served by GitHub Pages:
1. **Next game:** opponent, kickoff, venue, weather; Seahawks win probability, predicted margin ± interval; Vegas spread-implied probability shown as a benchmark.
2. **Why:** top feature contributions for this prediction.
3. **Season log:** every 2026 Seahawks prediction vs. actual result.
4. **Model report:** backtest metrics vs. baselines, calibration plot, holdout results.

## Repository Layout

```
pyproject.toml                 # uv-managed; Python 3.12 in CI
data/stadiums.csv
seahawks_ml/
  ingest/      nflverse.py, weather.py, stadiums.py
  features/    ratings.py (EPA + shrinkage), qb.py, elo.py,
               situational.py, availability.py, weather.py, build.py
  models/      train.py, backtest.py, predict.py, results.py
  site/        build.py, templates/
predictions/history.jsonl
models/        # promoted model artifact + metrics.json
docs/          # built dashboard (GitHub Pages)
tests/
.github/workflows/predict.yml, retrain.yml
```

Each `features/*.py` module exposes one function taking the games table (plus raw inputs) and returning per-game feature columns, so features are independently testable and the build step just joins them.

Stack: polars (nflreadpy's native format), scikit-learn, lightgbm, httpx for Open-Meteo, Jinja2 for the site, a CDN chart library for the dashboard.

## Testing

- **Leakage tests:** for sampled games, recompute features with all data after that game's kickoff removed; results must be identical to the full-data features.
- **Unit tests:** Elo update, rating shrinkage, travel distance/time zones, body-clock calculation, availability weighting, Φ conversion.
- **Schema tests:** feature table has expected columns, types, no unexpected nulls, one row per game.
- **Smoke test:** full pipeline (ingest from fixtures → features → train → predict → site build) on a small fixture dataset, run in CI on every push.

## Out of Scope (Phase 1)

- Full player-value model (all positions)
- Score/total predictions and empirical margin distributions
- Live or historical line snapshots, betting-specific metrics
- Seahawks-specific model or weighting

## Future Phases

- **Phase 2:** player-value ratings (snap share × per-play contribution by position) replacing the simple availability count.
- **Phase 3:** team-score model (points for each team), yielding totals and score distributions where weather should matter more.
