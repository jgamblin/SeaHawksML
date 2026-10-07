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

- `predict.yml` — every 15 minutes; predicts ~4 days, ~1 day, and 6h–75min before each kickoff,
  records results, and redeploys the dashboard.
- `retrain.yml` — Tuesdays; refits weights with the locked config (no tuning in CI).
- `ci.yml` — tests on every push.

Dashboard: the Seahawks-themed page is built to `public/index.html`
(`uv run python -m seahawks_ml.cli build-site`).

The dashboard's factor list excludes home-field because it sits in the model baseline.

League-wide scorecard: in the same job, every NFL game gets one prediction logged
about a day before kickoff (`predictions/league.jsonl`, append-only, with the Vegas and
Elo probabilities alongside) and its result once final. The dashboard's "Live this
season" card scores model vs Vegas vs Elo on those fresh, never-edited predictions.
The live Vegas benchmark uses the spread nflverse has about 6-24 hours before kickoff, which
is not necessarily the closing line.

Season simulation: runs once per "sim day" (starting 10:00 UTC), attempted on runs that
already load data or at 10/14/18/22 UTC when nothing else is due. The rest of the regular season is simulated 10,000 times. Completed
games keep their results; each remaining game is an independent coin flip weighted by the
current model's home-win probability (team strength is not resampled). Standings use
simplified tiebreakers (win %, head-to-head, division %, conference %, coin flip) to pick
4 division winners and 3 wild cards per conference. Seattle's projected wins, win-total
distribution, P(playoffs), P(division) and P(#1 seed) are appended to
`predictions/season_sim.jsonl` (at most one snapshot per day) and shown in the dashboard's
"Season outlook" card. Run it by hand with `uv run python -m seahawks_ml.cli simulate`.

Manual run: `uv run python -m seahawks_ml.cli predict --run-type midweek`

## First-time GitHub setup

1. Settings -> Pages -> Source = "GitHub Actions".
2. The `github-pages` environment (Settings -> Environments) must allow deployments from `main`.
3. Settings -> Actions -> General -> Workflow permissions = "Read and write".
4. Publish once: `gh workflow run predict.yml -f publish_only=true`

Site or metrics changes pushed to `main` also republish automatically.

## Offseason

GitHub disables scheduled workflows after 60 days without repo activity. The monthly
Keepalive workflow re-enables Predict, Retrain and itself through the API, which resets that
timer without commits, so nothing needs doing over the offseason. If workflows ever do get
disabled, re-enable them manually in the Actions tab (or `gh workflow enable predict.yml`,
`gh workflow enable retrain.yml` and `gh workflow enable keepalive.yml`).

In late August, run `uv run python -m seahawks_ml.cli retrain` (or the Retrain workflow).

## Maintenance

- A failed scheduled Predict or Retrain run opens a GitHub issue labelled `workflow-failure`
  (or comments on the existing open one) with a link to the failing run.
- Dependabot opens one grouped weekly PR for GitHub Actions updates.

Data: [nflverse](https://github.com/nflverse), weather by [Open-Meteo](https://open-meteo.com/).
An analytical model, not betting advice.
