# Seahawks ML

Predicts Seattle Seahawks games (win probability and margin) from free public data —
nflverse play-by-play, injuries, snap counts, rosters, and Open-Meteo weather — and
publishes a dashboard to GitHub Pages via scheduled GitHub Actions.

Design: `docs/superpowers/specs/2026-10-06-seahawks-win-model-design.md`

## Setup

```bash
brew install uv
uv sync
uv run pytest
```

## Local workflow (offseason or whenever you change features/models)

```bash
uv run python -m seahawks_ml.cli features                  # download + cache data, build features
uv run python -m seahawks_ml.cli backtest --tune-features  # tune, lock models/config.json (~15 min)
uv run python -m seahawks_ml.cli holdout                   # score 2024-2025 ONCE
uv run python -m seahawks_ml.cli retrain                   # fit production model
```

The first `features` run fetches ~17 seasons of game-time weather from Open-Meteo in
small paced requests. It can take an hour or more and may pause on rate limits; it saves
after every stadium, so re-running resumes where it stopped. Commit `data/cache/`
afterwards so CI never has to repeat it.

## In season (GitHub Actions)

- `predict.yml` — hourly; predicts ~4 days, ~1 day, and ~3 hours before each kickoff,
  records results, and redeploys the dashboard.
- `retrain.yml` — Tuesdays; refits weights with the locked config (no tuning in CI).
- `ci.yml` — tests on every push.

Dashboard: the Seahawks-themed page is built to `public/index.html`
(`uv run python -m seahawks_ml.cli build-site`).

Manual run: `uv run python -m seahawks_ml.cli predict --run-type midweek`

Data: [nflverse](https://github.com/nflverse), weather by [Open-Meteo](https://open-meteo.com/).
An analytical model, not betting advice.
