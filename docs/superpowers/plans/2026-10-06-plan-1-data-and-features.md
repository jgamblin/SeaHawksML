# Plan 1 of 3: Data Ingest and Feature Table

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download free NFL and weather data, cache it compactly, and build one leak-free feature row per NFL game since 2002.

**Architecture:** `seahawks_ml/ingest` downloads nflverse tables (play-by-play is aggregated per season immediately) and Open-Meteo game-hour weather into `data/cache/` (committed). `seahawks_ml/data.RawData` bundles every input and can truncate itself to any moment (`as_of`) for leakage tests. Each `seahawks_ml/features/*` module is a pure function returning per-game columns; `features/build.py` joins them into `data/features/games.parquet`.

**Tech Stack:** Python 3.12, uv, polars, nflreadpy, httpx, scikit-learn, LightGBM, scipy, Jinja2, pytest, GitHub Actions + Pages.

**Spec:** `docs/superpowers/specs/2026-10-06-seahawks-win-model-design.md`

---

## Notes verified against the real data (2026-10-06)

These facts were checked with `nflreadpy 0.1.5` and the live APIs; the code below depends on them:

- `load_schedules()` has `gameday` + `gametime` in **US Eastern** time, `result` = home minus away, `location` in {Home, Neutral}, `roof` in {outdoors, dome, closed, open, None}, and projected `home_qb_id`/`away_qb_id` even for unplayed games. Relocated teams appear as OAK/SD/STL in old seasons → normalized to LV/LAC/LA.
- `load_snap_counts()` only covers **2013+** (2012 returns 0 rows). Availability features are null before 2013 and filled with 0 plus an `availability_known` flag.
- `load_depth_charts()` changed format in 2025 (timestamped `dt`, no `week`). This plan does **not** use depth charts: starters come from snap shares, the expected starting QB from the schedule.
- Snap counts are keyed by PFR id; `load_players()` maps `pfr_id` → `gsis_id` and provides `draft_round`/`rookie_season`.
- Open-Meteo counts any request spanning more than two weeks as multiple calls, and long-range requests quickly hit HTTP 429. Weather is fetched in two-week windows around game dates, paced at 1 request/second, and the cache is saved after each stadium.

## File map

| File | Responsibility |
|---|---|
| `pyproject.toml`, `.python-version`, `.gitignore` | Project/tooling config |
| `seahawks_ml/config.py` | Paths and season constants |
| `seahawks_ml/teams.py` | Team abbreviation normalization |
| `data/stadiums.csv`, `seahawks_ml/stadiums.py` | Stadium coordinates, time zones, roof defaults |
| `seahawks_ml/features/base.py` | Canonical games table + per-team long view |
| `seahawks_ml/ingest/nflverse.py` | nflverse downloads, pbp aggregation, per-season cache |
| `seahawks_ml/ingest/weather.py` | Open-Meteo archive/forecast for game hours |
| `seahawks_ml/data.py` | `RawData` container, `as_of` truncation, `load_raw` |
| `seahawks_ml/features/{elo,coaching,ratings,qb,situational,season_timing,availability,weather}.py` | One feature family each |
| `seahawks_ml/features/columns.py`, `features/build.py` | Model column list; join everything |
| `seahawks_ml/cli.py` | `features` command (grows in Plans 2 and 3) |
| `tests/synthetic.py` | Offline synthetic `RawData` for tests |

---

### Task 1: Project scaffold

**Files:**
- Create/replace: `pyproject.toml`
- Create/replace: `.gitignore`
- Create/replace: `seahawks_ml/config.py`
- Create/replace: `seahawks_ml/teams.py`
- Test: `tests/test_teams.py`

- [ ] **Step 1: Install uv (if missing) and create the package skeleton**

```bash
brew install uv
mkdir -p seahawks_ml/ingest seahawks_ml/features seahawks_ml/models seahawks_ml/pipeline seahawks_ml/site/templates tests data
touch seahawks_ml/__init__.py seahawks_ml/ingest/__init__.py seahawks_ml/features/__init__.py seahawks_ml/models/__init__.py seahawks_ml/pipeline/__init__.py seahawks_ml/site/__init__.py tests/__init__.py
echo 3.12 > .python-version
```

- [ ] **Step 2: Write `pyproject.toml`, `.gitignore`, and `seahawks_ml/config.py`**

`pyproject.toml`:

```toml
[project]
name = "seahawks-ml"
version = "0.1.0"
description = "Seahawks game outcome model built on free public data"
requires-python = ">=3.12"
dependencies = [
    "nflreadpy>=0.1.5",
    "polars>=1.30",
    "pyarrow>=17",
    "numpy>=2.0",
    "scipy>=1.13",
    "scikit-learn>=1.5",
    "lightgbm>=4.5",
    "httpx>=0.27",
    "jinja2>=3.1",
]

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["seahawks_ml"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["network: hits external APIs (run with -m network)"]
addopts = "-m 'not network'"

[tool.ruff]
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "I"]
```

`.gitignore`:

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
public/
```

`seahawks_ml/config.py`:

```python
"""Paths and season constants shared across the project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
STADIUMS_CSV = DATA_DIR / "stadiums.csv"
FEATURES_PATH = DATA_DIR / "features" / "games.parquet"
MODELS_DIR = ROOT / "models"
PREDICTIONS_PATH = ROOT / "predictions" / "history.jsonl"
SITE_DIR = ROOT / "public"

FIRST_RATING_SEASON = 2002  # ratings warm up from here (32-team era)
FIRST_TRAIN_SEASON = 2009  # first season with injury reports
FIRST_SNAP_SEASON = 2013  # first season nflverse has snap counts
BACKTEST_SEASONS = tuple(range(2012, 2024))
HOLDOUT_SEASONS = (2024, 2025)

TEAM = "SEA"
# Relocated franchises are tracked under their current abbreviation.
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
```

- [ ] **Step 3: Install dependencies (creates `uv.lock`)**

```bash
uv sync
```
Expected: resolves and installs nflreadpy, polars, lightgbm, etc. into `.venv`; writes `uv.lock`.

- [ ] **Step 4: Write the failing test**

`tests/test_teams.py`:

```python
import polars as pl

from seahawks_ml.teams import normalize_team


def test_normalize_team_maps_relocated_franchises():
    df = pl.DataFrame({"team": ["OAK", "SD", "STL", "SEA"]})
    out = df.select(normalize_team("team"))["team"].to_list()
    assert out == ["LV", "LAC", "LA", "SEA"]
```

- [ ] **Step 5: Run it to verify it fails**

Run: `uv run pytest tests/test_teams.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.teams'`

- [ ] **Step 6: Write the implementation**

`seahawks_ml/teams.py`:

```python
"""Team abbreviation normalization."""

import polars as pl

from seahawks_ml.config import TEAM_ALIASES


def normalize_team(col: str) -> pl.Expr:
    """Map historical abbreviations (OAK, SD, STL) to the current franchise abbreviation."""
    return pl.col(col).replace(TEAM_ALIASES)
```

- [ ] **Step 7: Run it to verify it passes**

Run: `uv run pytest tests/test_teams.py -v`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitignore seahawks_ml tests
git commit -m "chore: scaffold project and team normalization"
```

---

### Task 2: Stadium reference data

**Files:**
- Create/replace: `data/stadiums.csv`
- Create/replace: `seahawks_ml/stadiums.py`
- Test: `tests/test_stadiums.py`

Weather lookups, travel distance, and time zones all key off nflverse `stadium_id`. Every stadium used since 2002 (including 2026 international venues) is listed; `validate_stadiums` fails loudly when nflverse adds a new one.

- [ ] **Step 1: Write the failing test**

`tests/test_stadiums.py`:

```python
from zoneinfo import ZoneInfo

import pytest

from seahawks_ml.stadiums import load_stadiums, validate_stadiums


def test_load_stadiums_parses_every_row():
    stadiums = load_stadiums()
    sea = stadiums["SEA00"]
    assert sea.name == "Lumen Field"
    assert sea.roof_default == "outdoors"
    assert 47 < sea.lat < 48 and -123 < sea.lon < -122
    for s in stadiums.values():
        ZoneInfo(s.tz)  # every tz must be a valid IANA zone
        assert -90 <= s.lat <= 90 and -180 <= s.lon <= 180


def test_validate_stadiums_reports_missing_ids():
    stadiums = load_stadiums()
    validate_stadiums(["SEA00", "LON02"], stadiums)
    with pytest.raises(ValueError, match="XXX99"):
        validate_stadiums(["SEA00", "XXX99"], stadiums)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_stadiums.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.stadiums'`

- [ ] **Step 3: Write the implementation**

`data/stadiums.csv`:

```csv
stadium_id,name,lat,lon,tz,roof_default
ATL00,Georgia Dome,33.758,-84.401,America/New_York,dome
ATL97,Mercedes-Benz Stadium,33.755,-84.401,America/New_York,retractable
BAL00,M&T Bank Stadium,39.278,-76.623,America/New_York,outdoors
BOS00,Gillette Stadium,42.091,-71.264,America/New_York,outdoors
BRG00,Tiger Stadium (LSU),30.412,-91.184,America/Chicago,outdoors
BUF00,Highmark Stadium,42.774,-78.787,America/New_York,outdoors
BUF01,Rogers Centre,43.641,-79.389,America/Toronto,dome
CAR00,Bank of America Stadium,35.226,-80.853,America/New_York,outdoors
CHI98,Soldier Field,41.862,-87.617,America/Chicago,outdoors
CHI99,Memorial Stadium (Champaign),40.099,-88.236,America/Chicago,outdoors
CIN00,Paycor Stadium,39.095,-84.516,America/New_York,outdoors
CLE00,Huntington Bank Field,41.506,-81.700,America/New_York,outdoors
DAL00,AT&T Stadium,32.748,-97.093,America/Chicago,retractable
DAL99,Texas Stadium,32.840,-96.911,America/Chicago,outdoors
DEN00,Empower Field at Mile High,39.744,-105.020,America/Denver,outdoors
DET00,Ford Field,42.340,-83.046,America/Detroit,dome
FRA00,Deutsche Bank Park,50.069,8.645,Europe/Berlin,outdoors
GER00,Allianz Arena,48.219,11.625,Europe/Berlin,outdoors
GNB00,Lambeau Field,44.501,-88.062,America/Chicago,outdoors
HOU00,NRG Stadium,29.685,-95.411,America/Chicago,retractable
IND00,Lucas Oil Stadium,39.760,-86.164,America/Indiana/Indianapolis,retractable
IND99,RCA Dome,39.764,-86.163,America/Indiana/Indianapolis,dome
JAX00,EverBank Stadium,30.324,-81.637,America/New_York,outdoors
KAN00,GEHA Field at Arrowhead Stadium,39.049,-94.484,America/Chicago,outdoors
LAX01,SoFi Stadium,33.953,-118.339,America/Los_Angeles,dome
LAX97,Dignity Health Sports Park,33.864,-118.261,America/Los_Angeles,outdoors
LAX99,Los Angeles Memorial Coliseum,34.014,-118.288,America/Los_Angeles,outdoors
LON00,Wembley Stadium,51.556,-0.280,Europe/London,outdoors
LON01,Twickenham Stadium,51.456,-0.342,Europe/London,outdoors
LON02,Tottenham Hotspur Stadium,51.604,-0.066,Europe/London,outdoors
MAD01,Santiago Bernabeu,40.453,-3.688,Europe/Madrid,retractable
MEL00,Melbourne Cricket Ground,-37.820,144.983,Australia/Melbourne,outdoors
MEX00,Estadio Azteca,19.303,-99.150,America/Mexico_City,outdoors
MIA00,Hard Rock Stadium,25.958,-80.239,America/New_York,outdoors
MIN00,Hubert H. Humphrey Metrodome,44.974,-93.258,America/Chicago,dome
MIN01,U.S. Bank Stadium,44.974,-93.258,America/Chicago,dome
MIN98,Huntington Bank Stadium (TCF Bank),44.976,-93.225,America/Chicago,outdoors
MUN01,Allianz Arena,48.219,11.625,Europe/Berlin,outdoors
NAS00,Nissan Stadium,36.166,-86.771,America/Chicago,outdoors
NOR00,Caesars Superdome,29.951,-90.081,America/Chicago,dome
NYC00,Giants Stadium,40.812,-74.077,America/New_York,outdoors
NYC01,MetLife Stadium,40.814,-74.074,America/New_York,outdoors
OAK00,Oakland Coliseum,37.751,-122.201,America/Los_Angeles,outdoors
PAR00,Stade de France,48.924,2.360,Europe/Paris,outdoors
PHI00,Lincoln Financial Field,39.901,-75.168,America/New_York,outdoors
PHI99,Veterans Stadium,39.907,-75.171,America/New_York,outdoors
PHO00,State Farm Stadium,33.528,-112.263,America/Phoenix,retractable
PHO99,Sun Devil Stadium,33.426,-111.933,America/Phoenix,outdoors
PIT00,Acrisure Stadium,40.447,-80.016,America/New_York,outdoors
RIO00,Maracana Stadium,-22.912,-43.230,America/Sao_Paulo,outdoors
SAN00,Alamodome,29.417,-98.479,America/Chicago,dome
SAO00,Arena Corinthians,-23.545,-46.474,America/Sao_Paulo,outdoors
SDG00,Qualcomm Stadium,32.783,-117.120,America/Los_Angeles,outdoors
SEA00,Lumen Field,47.595,-122.332,America/Los_Angeles,outdoors
SFO00,Candlestick Park,37.714,-122.386,America/Los_Angeles,outdoors
SFO01,Levi's Stadium,37.403,-121.970,America/Los_Angeles,outdoors
STL00,Edward Jones Dome,38.633,-90.188,America/Chicago,dome
TAM00,Raymond James Stadium,27.976,-82.503,America/New_York,outdoors
VEG00,Allegiant Stadium,36.091,-115.184,America/Los_Angeles,dome
WAS00,Northwest Stadium,38.908,-76.865,America/New_York,outdoors
```

`seahawks_ml/stadiums.py`:

```python
"""Stadium reference data (hand-maintained in data/stadiums.csv)."""

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from seahawks_ml.config import STADIUMS_CSV

ROOF_DEFAULTS = {"outdoors", "dome", "retractable"}


@dataclass(frozen=True)
class Stadium:
    stadium_id: str
    name: str
    lat: float
    lon: float
    tz: str
    roof_default: str


def load_stadiums(path: Path = STADIUMS_CSV) -> dict[str, Stadium]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    stadiums = {}
    for r in rows:
        if r["roof_default"] not in ROOF_DEFAULTS:
            raise ValueError(f"{r['stadium_id']}: bad roof_default {r['roof_default']!r}")
        stadiums[r["stadium_id"]] = Stadium(
            stadium_id=r["stadium_id"],
            name=r["name"],
            lat=float(r["lat"]),
            lon=float(r["lon"]),
            tz=r["tz"],
            roof_default=r["roof_default"],
        )
    return stadiums


def validate_stadiums(stadium_ids: Iterable[str], stadiums: dict[str, Stadium]) -> None:
    """Raise if any stadium_id from the schedule is missing from stadiums.csv."""
    missing = sorted({s for s in stadium_ids if s not in stadiums})
    if missing:
        raise ValueError(f"Add these stadium_ids to data/stadiums.csv: {missing}")
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_stadiums.py -v`
Expected: all PASS.

- [ ] **Step 5: Check every real stadium is covered**

```bash
uv run python -c "import nflreadpy as n, polars as pl; from seahawks_ml.stadiums import load_stadiums, validate_stadiums; s=n.load_schedules(True).filter(pl.col('season')>=2002); validate_stadiums(s['stadium_id'].to_list(), load_stadiums()); print('all stadiums present')"
```
Expected: `all stadiums present`. If it lists missing ids, add rows to `data/stadiums.csv` (lat/lon from the venue, IANA time zone, roof default) and rerun.

- [ ] **Step 6: Commit**

```bash
git add data/stadiums.csv seahawks_ml/stadiums.py tests/test_stadiums.py
git commit -m "feat: stadium reference table"
```

---

### Task 3: Canonical games table

**Files:**
- Create/replace: `seahawks_ml/features/base.py`
- Test: `tests/test_base.py`

`prepare_games` converts nflverse schedules into one typed table (kickoff in UTC, margin = home − away, resolved roof). `team_games` gives the two-rows-per-game view used by Elo, ratings, and coaching.

- [ ] **Step 1: Write the failing test**

`tests/test_base.py`:

```python
from datetime import UTC, datetime

import polars as pl

from seahawks_ml.features.base import kickoff_utc, prepare_games, resolve_roof, team_games
from seahawks_ml.stadiums import load_stadiums


def _schedule_row(**overrides):
    row = {
        "game_id": "2024_01_DEN_SEA", "season": 2024, "week": 1, "game_type": "REG",
        "gameday": "2024-09-08", "gametime": "16:05", "away_team": "DEN", "home_team": "SEA",
        "away_score": 20, "home_score": 26, "result": 6, "location": "Home", "roof": "outdoors",
        "stadium_id": "SEA00", "home_rest": 7, "away_rest": 7, "div_game": 0,
        "home_qb_id": "QB-SEA", "away_qb_id": "QB-DEN", "home_coach": "Mike Macdonald",
        "away_coach": "Sean Payton", "spread_line": 6.5,
    }
    row.update(overrides)
    return row


def test_kickoff_utc_converts_eastern_time():
    # 16:05 ET on 2024-09-08 is EDT (UTC-4) -> 20:05 UTC
    assert kickoff_utc("2024-09-08", "16:05") == datetime(2024, 9, 8, 20, 5, tzinfo=UTC)
    # December is EST (UTC-5)
    assert kickoff_utc("2024-12-15", "13:00") == datetime(2024, 12, 15, 18, 0, tzinfo=UTC)


def test_resolve_roof_falls_back_to_stadium_default():
    stadiums = load_stadiums()
    assert resolve_roof("open", stadiums["PHO00"]) == "open"
    assert resolve_roof(None, stadiums["PHO00"]) == "closed"
    assert resolve_roof(None, stadiums["LAX01"]) == "dome"
    assert resolve_roof(None, stadiums["SEA00"]) == "outdoors"


def test_prepare_games_builds_canonical_table():
    sched = pl.DataFrame([_schedule_row(), _schedule_row(
        game_id="2024_09_SEA_LON", stadium_id="LON02", location="Neutral", roof=None,
        result=None, home_score=None, away_score=None, gameday="2024-11-03", gametime="09:30",
    )])
    games = prepare_games(sched, load_stadiums())
    assert games.height == 2
    first = games.row(0, named=True)
    assert first["margin"] == 6 and first["neutral"] is False and first["div_game"] is False
    second = games.row(1, named=True)
    assert second["neutral"] is True and second["roof"] == "outdoors" and second["margin"] is None


def test_team_games_has_two_rows_per_game():
    games = prepare_games(pl.DataFrame([_schedule_row()]), load_stadiums())
    tg = team_games(games)
    assert tg.height == 2
    sea = tg.filter(pl.col("team") == "SEA").row(0, named=True)
    assert sea["is_home"] and sea["points_for"] == 26 and sea["opponent"] == "DEN"
    den = tg.filter(pl.col("team") == "DEN").row(0, named=True)
    assert not den["is_home"] and den["points_for"] == 20 and den["qb_id"] == "QB-DEN"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.features.base'`

- [ ] **Step 3: Write the implementation**

`seahawks_ml/features/base.py`:

```python
"""Canonical games table and the per-team long view used by most feature modules."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import polars as pl

from seahawks_ml.config import FIRST_RATING_SEASON
from seahawks_ml.stadiums import Stadium, validate_stadiums

ET = ZoneInfo("America/New_York")  # nflverse gametime is US Eastern

GAMES_SCHEMA = {
    "game_id": pl.Utf8,
    "season": pl.Int64,
    "week": pl.Int64,
    "game_type": pl.Utf8,
    "kickoff_utc": pl.Datetime("us", "UTC"),
    "home_team": pl.Utf8,
    "away_team": pl.Utf8,
    "home_score": pl.Int64,
    "away_score": pl.Int64,
    "margin": pl.Int64,  # home score minus away score
    "neutral": pl.Boolean,
    "roof": pl.Utf8,
    "stadium_id": pl.Utf8,
    "home_rest": pl.Int64,
    "away_rest": pl.Int64,
    "div_game": pl.Boolean,
    "home_qb_id": pl.Utf8,
    "away_qb_id": pl.Utf8,
    "home_coach": pl.Utf8,
    "away_coach": pl.Utf8,
    "spread_line": pl.Float64,  # positive = home favored (benchmark only)
}

TEAM_GAMES_SCHEMA = {
    "game_id": pl.Utf8,
    "season": pl.Int64,
    "week": pl.Int64,
    "kickoff_utc": pl.Datetime("us", "UTC"),
    "team": pl.Utf8,
    "opponent": pl.Utf8,
    "is_home": pl.Boolean,
    "points_for": pl.Int64,
    "points_against": pl.Int64,
    "coach": pl.Utf8,
    "qb_id": pl.Utf8,
}


def kickoff_utc(gameday: str, gametime: str) -> datetime:
    local = datetime.strptime(f"{gameday} {gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    return local.astimezone(UTC)


def resolve_roof(roof: str | None, stadium: Stadium) -> str:
    """Use the schedule's roof value; fall back to the stadium default (retractable -> closed)."""
    if roof:
        return roof
    return "closed" if stadium.roof_default == "retractable" else stadium.roof_default


def prepare_games(schedules: pl.DataFrame, stadiums: dict[str, Stadium]) -> pl.DataFrame:
    """Convert team-normalized nflverse schedules into the canonical games table."""
    sched = schedules.filter(pl.col("season") >= FIRST_RATING_SEASON)
    validate_stadiums(sched["stadium_id"].to_list(), stadiums)
    rows = []
    for r in sched.iter_rows(named=True):
        rows.append(
            {
                "game_id": r["game_id"],
                "season": r["season"],
                "week": r["week"],
                "game_type": r["game_type"],
                "kickoff_utc": kickoff_utc(r["gameday"], r["gametime"]),
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "home_score": r["home_score"],
                "away_score": r["away_score"],
                "margin": r["result"],
                "neutral": r["location"] == "Neutral",
                "roof": resolve_roof(r["roof"], stadiums[r["stadium_id"]]),
                "stadium_id": r["stadium_id"],
                "home_rest": r["home_rest"],
                "away_rest": r["away_rest"],
                "div_game": bool(r["div_game"]),
                "home_qb_id": r["home_qb_id"],
                "away_qb_id": r["away_qb_id"],
                "home_coach": r["home_coach"],
                "away_coach": r["away_coach"],
                "spread_line": r["spread_line"],
            }
        )
    return pl.DataFrame(rows, schema=GAMES_SCHEMA).sort("kickoff_utc", "game_id")


def team_games(games: pl.DataFrame) -> pl.DataFrame:
    """Two rows per game, one from each team's perspective."""
    rows = []
    for g in games.iter_rows(named=True):
        for side, other in (("home", "away"), ("away", "home")):
            rows.append(
                {
                    "game_id": g["game_id"],
                    "season": g["season"],
                    "week": g["week"],
                    "kickoff_utc": g["kickoff_utc"],
                    "team": g[f"{side}_team"],
                    "opponent": g[f"{other}_team"],
                    "is_home": side == "home",
                    "points_for": g[f"{side}_score"],
                    "points_against": g[f"{other}_score"],
                    "coach": g[f"{side}_coach"],
                    "qb_id": g[f"{side}_qb_id"],
                }
            )
    return pl.DataFrame(rows, schema=TEAM_GAMES_SCHEMA).sort("kickoff_utc", "game_id", "team")
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_base.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add seahawks_ml/features/base.py tests/test_base.py
git commit -m "feat: canonical games table"
```

---

### Task 4: nflverse ingest and cache

**Files:**
- Create/replace: `seahawks_ml/ingest/nflverse.py`
- Test: `tests/test_ingest_nflverse.py`

Play-by-play is reduced to team-game EPA and QB-game dropback totals right after download, so only small parquet files are cached (`data/cache/*_{season}.parquet`). Completed seasons are read from cache; the current season is refreshed every run. Before a season's first game nflverse raises; that becomes an empty table.

- [ ] **Step 1: Write the failing test**

`tests/test_ingest_nflverse.py`:

```python
import polars as pl

from seahawks_ml.ingest.nflverse import aggregate_qb_games, aggregate_team_epa


def _pbp():
    return pl.DataFrame({
        "game_id": ["g1"] * 6,
        "season": [2015] * 6,
        "posteam": ["STL", "STL", "STL", "SEA", "SEA", None],
        "defteam": ["SEA", "SEA", "SEA", "STL", "STL", None],
        "play_type": ["pass", "run", "punt", "pass", "pass", None],
        "epa": [0.5, -0.1, 2.0, 1.0, None, 0.3],
        "qb_dropback": [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        "passer_player_id": ["QB-STL", None, None, "QB-SEA", "QB-SEA", None],
        "qb_epa": [0.5, -0.1, 2.0, 1.0, None, 0.3],
        "cpoe": [10.0, None, None, None, -5.0, None],
    })


def test_aggregate_team_epa_uses_scrimmage_plays_and_normalizes_teams():
    out = aggregate_team_epa(_pbp())
    rows = {r["team"]: r for r in out.iter_rows(named=True)}
    assert set(rows) == {"LA", "SEA"}  # STL -> LA, null posteam dropped
    assert rows["LA"]["plays"] == 2 and abs(rows["LA"]["epa_sum"] - 0.4) < 1e-9  # punt excluded
    assert rows["LA"]["opponent"] == "SEA"
    assert rows["SEA"]["plays"] == 1  # null epa excluded


def test_aggregate_qb_games_sums_dropbacks():
    out = aggregate_qb_games(_pbp())
    rows = {r["qb_id"]: r for r in out.iter_rows(named=True)}
    assert rows["QB-STL"]["dropbacks"] == 1 and rows["QB-STL"]["team"] == "LA"
    assert rows["QB-STL"]["cpoe_n"] == 1 and rows["QB-STL"]["cpoe_sum"] == 10.0
    assert rows["QB-SEA"]["dropbacks"] == 1  # second SEA dropback has null qb_epa
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_ingest_nflverse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.ingest.nflverse'`

- [ ] **Step 3: Write the implementation**

`seahawks_ml/ingest/nflverse.py`:

```python
"""Download nflverse tables and reduce them to compact, per-season cached parquet files.

Completed seasons are cached in data/cache and committed; the current season is
refreshed on every run. Play-by-play is aggregated immediately so the full
372-column table never has to be kept around.
"""

from collections.abc import Callable
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from seahawks_ml.config import CACHE_DIR, FIRST_SNAP_SEASON, FIRST_TRAIN_SEASON
from seahawks_ml.teams import normalize_team

TEAM_EPA_SCHEMA = {
    "game_id": pl.Utf8, "season": pl.Int64, "team": pl.Utf8, "opponent": pl.Utf8,
    "epa_sum": pl.Float64, "plays": pl.Int64,
}
QB_GAMES_SCHEMA = {
    "game_id": pl.Utf8, "season": pl.Int64, "team": pl.Utf8, "qb_id": pl.Utf8,
    "dropbacks": pl.Int64, "qb_epa_sum": pl.Float64, "cpoe_sum": pl.Float64, "cpoe_n": pl.Int64,
}
INJURIES_SCHEMA = {
    "season": pl.Int64, "week": pl.Int64, "team": pl.Utf8, "gsis_id": pl.Utf8,
    "position": pl.Utf8, "report_status": pl.Utf8,
}
SNAPS_SCHEMA = {
    "game_id": pl.Utf8, "season": pl.Int64, "week": pl.Int64, "team": pl.Utf8,
    "pfr_player_id": pl.Utf8, "position": pl.Utf8, "offense_pct": pl.Float64,
    "defense_pct": pl.Float64,
}
PLAYERS_SCHEMA = {
    "gsis_id": pl.Utf8, "pfr_id": pl.Utf8, "position": pl.Utf8,
    "draft_round": pl.Int64, "rookie_season": pl.Int64,
}


def aggregate_team_epa(pbp: pl.DataFrame) -> pl.DataFrame:
    """Offensive EPA per team-game from scrimmage plays (pass + run)."""
    plays = pbp.filter(
        pl.col("play_type").is_in(["pass", "run"])
        & pl.col("epa").is_not_null()
        & pl.col("posteam").is_not_null()
    )
    return (
        plays.group_by("game_id", "season", "posteam", "defteam")
        .agg(pl.col("epa").sum().alias("epa_sum"), pl.len().alias("plays"))
        .rename({"posteam": "team", "defteam": "opponent"})
        .with_columns(normalize_team("team"), normalize_team("opponent"))
        .cast(TEAM_EPA_SCHEMA)
        .select(list(TEAM_EPA_SCHEMA))
        .sort("game_id", "team")
    )


def aggregate_qb_games(pbp: pl.DataFrame) -> pl.DataFrame:
    """Dropback totals per passer per game."""
    drops = pbp.filter(
        (pl.col("qb_dropback") == 1)
        & pl.col("passer_player_id").is_not_null()
        & pl.col("qb_epa").is_not_null()
    )
    return (
        drops.group_by("game_id", "season", "posteam", "passer_player_id")
        .agg(
            pl.len().alias("dropbacks"),
            pl.col("qb_epa").sum().alias("qb_epa_sum"),
            pl.col("cpoe").sum().alias("cpoe_sum"),
            pl.col("cpoe").count().alias("cpoe_n"),
        )
        .rename({"posteam": "team", "passer_player_id": "qb_id"})
        .with_columns(normalize_team("team"))
        .cast(QB_GAMES_SCHEMA)
        .select(list(QB_GAMES_SCHEMA))
        .sort("game_id", "qb_id")
    )


def _cached(path: Path, fetch: Callable[[], pl.DataFrame], refresh: bool) -> pl.DataFrame:
    if path.exists() and not refresh:
        return pl.read_parquet(path)
    df = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


def _or_empty(fetch: Callable[[], pl.DataFrame], schema: dict) -> pl.DataFrame:
    """nflverse raises before a season's first data exists; treat that as empty."""
    try:
        return fetch()
    except Exception as exc:  # noqa: BLE001 - nflreadpy raises several error types
        print(f"warning: nflverse fetch returned no data ({exc}); using empty table")
        return pl.DataFrame(schema=schema)


def load_schedules() -> pl.DataFrame:
    return nfl.load_schedules(True).with_columns(
        normalize_team("home_team"), normalize_team("away_team")
    )


def load_players() -> pl.DataFrame:
    return (
        nfl.load_players()
        .select(list(PLAYERS_SCHEMA))
        .cast(PLAYERS_SCHEMA)
        .filter(pl.col("gsis_id").is_not_null())
    )


def _fetch_pbp_tables(season: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    try:
        pbp = nfl.load_pbp(season)
    except Exception as exc:  # noqa: BLE001 - nflreadpy raises several error types
        print(f"warning: no play-by-play for {season} yet ({exc})")
        return pl.DataFrame(schema=TEAM_EPA_SCHEMA), pl.DataFrame(schema=QB_GAMES_SCHEMA)
    return aggregate_team_epa(pbp), aggregate_qb_games(pbp)


def _fetch_injuries(season: int) -> pl.DataFrame:
    return (
        nfl.load_injuries(season)
        .with_columns(normalize_team("team"))
        .select(list(INJURIES_SCHEMA))
        .cast(INJURIES_SCHEMA)
    )


def _fetch_snaps(season: int) -> pl.DataFrame:
    return (
        nfl.load_snap_counts(season)
        .with_columns(normalize_team("team"))
        .select(list(SNAPS_SCHEMA))
        .cast(SNAPS_SCHEMA)
    )


def load_season_tables(
    seasons: list[int], current_season: int, cache_dir: Path = CACHE_DIR
) -> dict[str, pl.DataFrame]:
    """Return team_epa, qb_games, injuries and snaps for the given seasons."""
    parts: dict[str, list[pl.DataFrame]] = {k: [] for k in ("team_epa", "qb_games", "injuries", "snaps")}
    for season in seasons:
        refresh = season == current_season
        team_path = cache_dir / f"team_epa_{season}.parquet"
        qb_path = cache_dir / f"qb_games_{season}.parquet"
        if refresh or not (team_path.exists() and qb_path.exists()):
            team_epa, qb_games = _fetch_pbp_tables(season)
            cache_dir.mkdir(parents=True, exist_ok=True)
            team_epa.write_parquet(team_path)
            qb_games.write_parquet(qb_path)
        parts["team_epa"].append(pl.read_parquet(team_path))
        parts["qb_games"].append(pl.read_parquet(qb_path))
        if season >= FIRST_TRAIN_SEASON:
            parts["injuries"].append(_cached(
                cache_dir / f"injuries_{season}.parquet",
                lambda s=season: _or_empty(lambda: _fetch_injuries(s), INJURIES_SCHEMA),
                refresh,
            ))
        if season >= FIRST_SNAP_SEASON:
            parts["snaps"].append(_cached(
                cache_dir / f"snaps_{season}.parquet",
                lambda s=season: _or_empty(lambda: _fetch_snaps(s), SNAPS_SCHEMA),
                refresh,
            ))
    schemas = {"team_epa": TEAM_EPA_SCHEMA, "qb_games": QB_GAMES_SCHEMA,
               "injuries": INJURIES_SCHEMA, "snaps": SNAPS_SCHEMA}
    return {
        k: pl.concat([p.cast(schemas[k]) for p in v]) if v else pl.DataFrame(schema=schemas[k])
        for k, v in parts.items()
    }
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_ingest_nflverse.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add seahawks_ml/ingest/nflverse.py tests/test_ingest_nflverse.py
git commit -m "feat: nflverse ingest with per-season cache"
```

---

### Task 5: Open-Meteo weather ingest

**Files:**
- Create/replace: `seahawks_ml/ingest/weather.py`
- Test: `tests/test_ingest_weather.py`

Only the four hours from each outdoor kickoff are kept. Archive data lags ~5 days, so games newer than 6 days are skipped (they fall back to climatology in features). Requests are two-week windows, paced, with a 65-second back-off on HTTP 429.

- [ ] **Step 1: Write the failing test**

`tests/test_ingest_weather.py`:

```python
from datetime import UTC, datetime

import httpx
import polars as pl
import pytest

from seahawks_ml.ingest.weather import (
    date_windows,
    fetch_archive,
    forecast_for_games,
    game_hours,
    parse_hourly,
    update_archive_cache,
)
from seahawks_ml.stadiums import load_stadiums


def _games(rows):
    return pl.DataFrame(rows, schema={
        "game_id": pl.Utf8, "stadium_id": pl.Utf8, "roof": pl.Utf8,
        "kickoff_utc": pl.Datetime("us", "UTC"),
    }).with_columns(pl.col("kickoff_utc").dt.year().cast(pl.Int64).alias("season"))


def _payload(times):
    return {"hourly": {
        "time": times,
        "temperature_2m": [50.0 + i for i in range(len(times))],
        "wind_speed_10m": [10.0] * len(times),
        "precipitation": [0.1] * len(times),
    }}


def test_parse_hourly_builds_utc_rows():
    df = parse_hourly(_payload(["2024-09-08T20:00", "2024-09-08T21:00"]), "SEA00", "archive")
    assert df["time_utc"][0] == datetime(2024, 9, 8, 20, tzinfo=UTC)
    assert df["temp_f"].to_list() == [50.0, 51.0]
    assert df["source"].unique().to_list() == ["archive"]


def test_game_hours_skips_indoor_games_and_covers_window():
    games = _games([
        {"game_id": "a", "stadium_id": "SEA00", "roof": "outdoors",
         "kickoff_utc": datetime(2024, 9, 8, 20, 5, tzinfo=UTC)},
        {"game_id": "b", "stadium_id": "LAX01", "roof": "dome",
         "kickoff_utc": datetime(2024, 9, 8, 20, 5, tzinfo=UTC)},
    ])
    hours = game_hours(games)
    assert hours["stadium_id"].unique().to_list() == ["SEA00"]
    assert hours["time_utc"].to_list() == [datetime(2024, 9, 8, h, tzinfo=UTC) for h in (20, 21, 22, 23)]


def test_update_archive_cache_fetches_only_missing_hours(tmp_path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        times = [f"2024-09-08T{h:02d}:00" for h in range(24)]
        return httpx.Response(200, json=_payload(times))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    games = _games([{"game_id": "a", "stadium_id": "SEA00", "roof": "outdoors",
                     "kickoff_utc": datetime(2024, 9, 8, 20, 5, tzinfo=UTC)}])
    path = tmp_path / "w.parquet"
    now = datetime(2024, 10, 1, tzinfo=UTC)
    out = update_archive_cache(games, load_stadiums(), now, client=client, cache_path=path, pause=0)
    assert out.height == 4 and len(calls) == 1
    assert calls[0]["start_date"] == "2024-09-08"
    out2 = update_archive_cache(games, load_stadiums(), now, client=client, cache_path=path, pause=0)
    assert out2.height == 4 and len(calls) == 1  # nothing new fetched


def test_update_archive_cache_skips_games_inside_archive_lag(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: pytest.fail("no fetch expected")))
    games = _games([{"game_id": "a", "stadium_id": "SEA00", "roof": "outdoors",
                     "kickoff_utc": datetime(2024, 9, 8, 20, 5, tzinfo=UTC)}])
    out = update_archive_cache(games, load_stadiums(), datetime(2024, 9, 10, tzinfo=UTC),
                               client=client, cache_path=tmp_path / "w.parquet", pause=0)
    assert out.height == 0


def test_date_windows_groups_dates_within_two_weeks():
    from datetime import date
    ds = [date(2024, 9, 8), date(2024, 9, 15), date(2024, 9, 22), date(2024, 10, 20)]
    assert date_windows(ds) == [(date(2024, 9, 8), date(2024, 9, 15)),
                                (date(2024, 9, 22), date(2024, 9, 22)),
                                (date(2024, 10, 20), date(2024, 10, 20))]


def test_update_archive_cache_ignores_seasons_before_training(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: pytest.fail("no fetch expected")))
    games = _games([{"game_id": "a", "stadium_id": "SEA00", "roof": "outdoors",
                     "kickoff_utc": datetime(2005, 9, 8, 20, 5, tzinfo=UTC)}])
    out = update_archive_cache(games, load_stadiums(), datetime(2024, 9, 10, tzinfo=UTC),
                               client=client, cache_path=tmp_path / "w.parquet", pause=0)
    assert out.height == 0


def test_forecast_for_games_keeps_only_game_window():
    def handler(request):
        return httpx.Response(200, json=_payload([f"2024-09-08T{h:02d}:00" for h in range(24)]))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    games = _games([{"game_id": "a", "stadium_id": "SEA00", "roof": "outdoors",
                     "kickoff_utc": datetime(2024, 9, 8, 20, 5, tzinfo=UTC)}])
    out = forecast_for_games(games, load_stadiums(), client=client)
    assert out.height == 4 and out["source"].unique().to_list() == ["forecast"]


@pytest.mark.network
def test_fetch_archive_live():
    with httpx.Client() as client:
        df = fetch_archive(client, load_stadiums()["SEA00"],
                           datetime(2024, 9, 8).date(), datetime(2024, 9, 8).date())
    assert df.height == 24
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_ingest_weather.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.ingest.weather'`

- [ ] **Step 3: Write the implementation**

`seahawks_ml/ingest/weather.py`:

```python
"""Open-Meteo weather for game hours (archive for history, forecast for upcoming games).

Only the hours around each outdoor kickoff are kept, so the cache stays small
enough to commit (data/cache/weather_games.parquet).
"""

import time
from datetime import UTC, date, datetime, timedelta

import httpx
import polars as pl

from seahawks_ml.config import CACHE_DIR, FIRST_TRAIN_SEASON
from seahawks_ml.stadiums import Stadium

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_LAG_DAYS = 6  # archive data trails real time by up to ~5 days
WINDOW_DAYS = 14  # Open-Meteo counts requests spanning > 2 weeks as multiple calls
RATE_LIMIT_SLEEP = 65  # seconds to wait after HTTP 429 (per-minute limit)
GAME_WINDOW_HOURS = 4  # kickoff hour plus the next three
BASE_PARAMS = {
    "hourly": "temperature_2m,wind_speed_10m,precipitation",
    "timezone": "UTC",
    "temperature_unit": "fahrenheit",
    "wind_speed_unit": "mph",
    "precipitation_unit": "inch",
}
WEATHER_SCHEMA = {
    "stadium_id": pl.Utf8,
    "time_utc": pl.Datetime("us", "UTC"),
    "temp_f": pl.Float64,
    "wind_mph": pl.Float64,
    "precip_in": pl.Float64,
    "source": pl.Utf8,  # "archive" or "forecast"
}


def parse_hourly(payload: dict, stadium_id: str, source: str) -> pl.DataFrame:
    h = payload["hourly"]
    n = len(h["time"])
    return pl.DataFrame(
        {
            "stadium_id": [stadium_id] * n,
            "time_utc": [datetime.fromisoformat(t).replace(tzinfo=UTC) for t in h["time"]],
            "temp_f": h["temperature_2m"],
            "wind_mph": h["wind_speed_10m"],
            "precip_in": h["precipitation"],
            "source": [source] * n,
        },
        schema=WEATHER_SCHEMA,
    )


def _get_json(client: httpx.Client, url: str, params: dict, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            resp = client.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            if attempt == retries - 1:
                raise
            limited = isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429
            time.sleep(RATE_LIMIT_SLEEP if limited else 2 ** (attempt + 1))
    raise AssertionError("unreachable")


def date_windows(dates: list[date], max_days: int = WINDOW_DAYS) -> list[tuple[date, date]]:
    """Group sorted dates into [start, end] windows no longer than max_days."""
    windows: list[tuple[date, date]] = []
    for d in sorted(set(dates)):
        if windows and (d - windows[-1][0]).days < max_days:
            windows[-1] = (windows[-1][0], d)
        else:
            windows.append((d, d))
    return windows


def fetch_archive(client: httpx.Client, stadium: Stadium, start: date, end: date) -> pl.DataFrame:
    params = {**BASE_PARAMS, "latitude": stadium.lat, "longitude": stadium.lon,
              "start_date": start.isoformat(), "end_date": end.isoformat()}
    return parse_hourly(_get_json(client, ARCHIVE_URL, params), stadium.stadium_id, "archive")


def fetch_forecast(client: httpx.Client, stadium: Stadium) -> pl.DataFrame:
    params = {**BASE_PARAMS, "latitude": stadium.lat, "longitude": stadium.lon, "forecast_days": 16}
    return parse_hourly(_get_json(client, FORECAST_URL, params), stadium.stadium_id, "forecast")


def game_hours(games: pl.DataFrame) -> pl.DataFrame:
    """(stadium_id, time_utc) for each hour in each outdoor game's window."""
    rows = []
    for g in games.filter(~pl.col("roof").is_in(["dome", "closed"])).iter_rows(named=True):
        start = g["kickoff_utc"].replace(minute=0, second=0, microsecond=0)
        for h in range(GAME_WINDOW_HOURS):
            rows.append({"stadium_id": g["stadium_id"], "time_utc": start + timedelta(hours=h)})
    schema = {"stadium_id": pl.Utf8, "time_utc": pl.Datetime("us", "UTC")}
    return pl.DataFrame(rows, schema=schema).unique().sort("stadium_id", "time_utc")


def update_archive_cache(
    games: pl.DataFrame,
    stadiums: dict[str, Stadium],
    now: datetime,
    client: httpx.Client | None = None,
    cache_path=CACHE_DIR / "weather_games.parquet",
    pause: float = 1.0,
) -> pl.DataFrame:
    """Fetch archive weather for outdoor game hours (2009+) not yet cached; return the cache.

    Requests are two-week windows around game dates, paced by `pause` seconds, and the
    cache is written after each stadium so an interrupted first run can resume.
    """
    cached = pl.read_parquet(cache_path) if cache_path.exists() else pl.DataFrame(schema=WEATHER_SCHEMA)
    cutoff = now - timedelta(days=ARCHIVE_LAG_DAYS)
    eligible = games.filter((pl.col("kickoff_utc") < cutoff) & (pl.col("season") >= FIRST_TRAIN_SEASON))
    needed = game_hours(eligible).join(
        cached.select("stadium_id", "time_utc"), on=["stadium_id", "time_utc"], how="anti"
    )
    if needed.height == 0:
        return cached
    own_client = client is None
    client = client or httpx.Client()
    try:
        for (stadium_id,), hours in needed.sort("stadium_id").group_by(["stadium_id"], maintain_order=True):
            hours_set = set(hours["time_utc"].to_list())
            windows = date_windows([t.date() for t in hours_set])
            print(f"weather archive: {stadium_id} ({len(windows)} requests)")
            parts = []
            for first, last in windows:
                parts.append(fetch_archive(client, stadiums[stadium_id], first, last)
                             .filter(pl.col("time_utc").is_in(list(hours_set))))
                time.sleep(pause)
            cached = pl.concat([cached, *parts]).unique(["stadium_id", "time_utc"]).sort("stadium_id", "time_utc")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cached.write_parquet(cache_path)
    finally:
        if own_client:
            client.close()
    return cached


def forecast_for_games(
    games: pl.DataFrame, stadiums: dict[str, Stadium], client: httpx.Client | None = None
) -> pl.DataFrame:
    """Forecast rows for the game windows of the given (upcoming) games."""
    hours = game_hours(games)
    if hours.height == 0:
        return pl.DataFrame(schema=WEATHER_SCHEMA)
    own_client = client is None
    client = client or httpx.Client()
    try:
        parts = [
            fetch_forecast(client, stadiums[sid]).join(h, on=["stadium_id", "time_utc"], how="inner")
            for (sid,), h in hours.group_by(["stadium_id"])
        ]
    finally:
        if own_client:
            client.close()
    return pl.concat(parts).cast(WEATHER_SCHEMA) if parts else pl.DataFrame(schema=WEATHER_SCHEMA)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_ingest_weather.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the live API check once**

Run: `uv run pytest -m network -v`
Expected: `test_fetch_archive_live PASSED` (24 hourly rows).

- [ ] **Step 6: Commit**

```bash
git add seahawks_ml/ingest/weather.py tests/test_ingest_weather.py
git commit -m "feat: Open-Meteo game-hour weather ingest"
```

---

### Task 6: RawData container and synthetic test data

**Files:**
- Create/replace: `seahawks_ml/data.py`
- Create/replace: `tests/synthetic.py`
- Test: `tests/test_data.py`

`RawData` bundles every input table. `as_of(cutoff)` hides everything not knowable before a kickoff; the leakage tests in Task 15 compare features built from full vs. truncated data. `tests/synthetic.py` builds a small, offline, internally consistent dataset (4 NFC West teams, double round robin) used by most later tests.

- [ ] **Step 1: Write the test helper**

`tests/synthetic.py`:

```python
"""Small, fully synthetic RawData for fast offline tests.

Four NFC West teams play a 6-week double round robin each season. Team strength,
EPA, QB stats, snaps, injuries and weather are random but internally consistent.
"""

import random
from datetime import UTC, datetime, timedelta

import polars as pl

from seahawks_ml.data import RawData
from seahawks_ml.features.base import GAMES_SCHEMA
from seahawks_ml.ingest.nflverse import (
    INJURIES_SCHEMA,
    PLAYERS_SCHEMA,
    QB_GAMES_SCHEMA,
    SNAPS_SCHEMA,
    TEAM_EPA_SCHEMA,
)
from seahawks_ml.ingest.weather import GAME_WINDOW_HOURS, WEATHER_SCHEMA

TEAMS = ["SEA", "SF", "LA", "ARI"]
HOME_STADIUM = {"SEA": "SEA00", "SF": "SFO01", "LA": "LAX01", "ARI": "PHO00"}
ROOF = {"SEA00": "outdoors", "SFO01": "outdoors", "LAX01": "dome", "PHO00": "closed"}
ROUNDS = [[("SEA", "SF"), ("LA", "ARI")], [("SEA", "LA"), ("SF", "ARI")], [("SEA", "ARI"), ("SF", "LA")]]


def _coach(team: str, season: int, seasons) -> str:
    """SF changes head coach after the first season."""
    return f"{team} Coach B" if (team == "SF" and season >= seasons[1]) else f"{team} Coach A"


def make_raw(seasons=(2011, 2012, 2013, 2014), seed: int = 0, unplayed_last_week: bool = False) -> RawData:
    rng = random.Random(seed)
    strength = {t: rng.gauss(0, 4) for t in TEAMS}
    games, team_epa, qb_games, snaps, injuries, weather = [], [], [], [], [], []
    players = []
    for t in TEAMS:
        players.append({"gsis_id": f"{t}-QB1", "pfr_id": f"{t}QB100", "position": "QB",
                        "draft_round": {"SEA": 3, "SF": 1, "LA": 1, "ARI": None}[t], "rookie_season": 2008})
        players.append({"gsis_id": f"{t}-QB2", "pfr_id": f"{t}QB200", "position": "QB",
                        "draft_round": 6, "rookie_season": 2012})
        for i in range(10):
            players.append({"gsis_id": f"{t}-P{i}", "pfr_id": f"{t}P{i:03d}", "position": "WR" if i < 5 else "LB",
                            "draft_round": 2, "rookie_season": 2010})
    last_season = max(seasons)
    for season in seasons:
        for week in range(1, 7):
            pairs = ROUNDS[(week - 1) % 3]
            kickoff = datetime(season, 9, 8, 17, 0, tzinfo=UTC) + timedelta(days=7 * (week - 1))
            for a, b in pairs:
                home, away = (a, b) if week <= 3 else (b, a)
                game_id = f"{season}_{week:02d}_{away}_{home}"
                stadium = HOME_STADIUM[home]
                unplayed = unplayed_last_week and season == last_season and week == 6
                exp = strength[home] - strength[away] + 2
                margin = None if unplayed else round(rng.gauss(exp, 13))
                home_score = None if unplayed else 20 + max(margin, 0)
                away_score = None if unplayed else 20 + max(-margin, 0)
                home_qb = f"{home}-QB2" if (home == "SF" and season == last_season and week >= 4) else f"{home}-QB1"
                away_qb = f"{away}-QB1"
                games.append({
                    "game_id": game_id, "season": season, "week": week, "game_type": "REG",
                    "kickoff_utc": kickoff, "home_team": home, "away_team": away,
                    "home_score": home_score, "away_score": away_score, "margin": margin,
                    "neutral": False, "roof": ROOF[stadium], "stadium_id": stadium,
                    "home_rest": 7, "away_rest": 7 if week > 1 else 10, "div_game": True,
                    "home_qb_id": home_qb, "away_qb_id": away_qb,
                    "home_coach": _coach(home, season, seasons), "away_coach": _coach(away, season, seasons),
                    "spread_line": round(exp * 2) / 2,
                })
                if unplayed:
                    continue
                for team, opp, qb, sign in ((home, away, home_qb, 1), (away, home, away_qb, -1)):
                    epa = sign * margin / 30 + rng.gauss(0, 3)
                    team_epa.append({"game_id": game_id, "season": season, "team": team,
                                     "opponent": opp, "epa_sum": epa, "plays": 60})
                    qb_games.append({"game_id": game_id, "season": season, "team": team, "qb_id": qb,
                                     "dropbacks": 35, "qb_epa_sum": epa * 0.8,
                                     "cpoe_sum": rng.gauss(0, 30), "cpoe_n": 30})
                    if season >= 2013:
                        for i in range(10):
                            snaps.append({"game_id": game_id, "season": season, "week": week,
                                          "team": team, "pfr_player_id": f"{team}P{i:03d}",
                                          "position": "WR" if i < 5 else "LB",
                                          "offense_pct": 0.9 if i < 5 else 0.0,
                                          "defense_pct": 0.0 if i < 5 else 0.9})
                if ROOF[stadium] == "outdoors":
                    for h in range(GAME_WINDOW_HOURS):
                        weather.append({"stadium_id": stadium, "time_utc": kickoff + timedelta(hours=h),
                                        "temp_f": 60.0 - week, "wind_mph": 5.0 + h, "precip_in": 0.0,
                                        "source": "archive"})
            for t in TEAMS:
                if rng.random() < 0.5:
                    injuries.append({"season": season, "week": week, "team": t,
                                     "gsis_id": f"{t}-P{rng.randrange(10)}", "position": "WR",
                                     "report_status": rng.choice(["Out", "Doubtful", "Questionable"])})
    return RawData(
        games=pl.DataFrame(games, schema=GAMES_SCHEMA).sort("kickoff_utc", "game_id"),
        team_epa=pl.DataFrame(team_epa, schema=TEAM_EPA_SCHEMA),
        qb_games=pl.DataFrame(qb_games, schema=QB_GAMES_SCHEMA),
        injuries=pl.DataFrame(injuries, schema=INJURIES_SCHEMA),
        snaps=pl.DataFrame(snaps, schema=SNAPS_SCHEMA),
        players=pl.DataFrame(players, schema=PLAYERS_SCHEMA),
        weather=pl.DataFrame(weather, schema=WEATHER_SCHEMA).unique(["stadium_id", "time_utc"]),
    )
```

- [ ] **Step 2: Write the failing test**

`tests/test_data.py`:

```python
import polars as pl

from tests.synthetic import make_raw


def test_as_of_hides_results_and_stats_from_cutoff_onward():
    raw = make_raw()
    target = raw.games.filter(pl.col("season") == 2013).row(4, named=True)
    cut = raw.as_of(target["kickoff_utc"])
    g = cut.games.filter(pl.col("game_id") == target["game_id"]).row(0, named=True)
    assert g["margin"] is None and g["home_score"] is None
    assert target["game_id"] not in cut.team_epa["game_id"].to_list()
    assert target["game_id"] not in cut.snaps["game_id"].to_list()
    assert cut.games.filter(pl.col("kickoff_utc") < target["kickoff_utc"])["margin"].null_count() == 0
    # the target game's own week of injury reports survives; later weeks do not
    assert cut.injuries.filter((pl.col("season") == 2013) & (pl.col("week") > target["week"])).height == 0
    assert cut.games.height == raw.games.height
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.data'`

- [ ] **Step 4: Write the implementation**

`seahawks_ml/data.py`:

```python
"""RawData: every input table the feature builder needs, plus as-of truncation."""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

import polars as pl

from seahawks_ml.config import FIRST_RATING_SEASON
from seahawks_ml.features.base import prepare_games
from seahawks_ml.stadiums import Stadium


@dataclass(frozen=True)
class RawData:
    games: pl.DataFrame  # canonical games table (features.base.GAMES_SCHEMA)
    team_epa: pl.DataFrame  # ingest.nflverse.TEAM_EPA_SCHEMA
    qb_games: pl.DataFrame  # ingest.nflverse.QB_GAMES_SCHEMA
    injuries: pl.DataFrame  # ingest.nflverse.INJURIES_SCHEMA
    snaps: pl.DataFrame  # ingest.nflverse.SNAPS_SCHEMA
    players: pl.DataFrame  # ingest.nflverse.PLAYERS_SCHEMA
    weather: pl.DataFrame  # ingest.weather.WEATHER_SCHEMA

    def as_of(self, cutoff: datetime) -> "RawData":
        """Drop everything not knowable before `cutoff`.

        Results of games kicking off at/after the cutoff are nulled; per-game stats for
        those games are removed; injury reports are kept only for weeks that have
        started by cutoff + 1 day (so a game's own final report survives). Weather is
        left alone because a forecast stands in for it at prediction time.
        """
        later = pl.col("kickoff_utc") >= cutoff
        games = self.games.with_columns(
            *[pl.when(later).then(None).otherwise(pl.col(c)).alias(c)
              for c in ("home_score", "away_score", "margin")]
        )
        played = set(self.games.filter(~later)["game_id"].to_list())
        week_start = self.games.group_by("season", "week").agg(pl.col("kickoff_utc").min().alias("start"))
        open_weeks = week_start.filter(pl.col("start") < cutoff + timedelta(days=1)).select("season", "week")
        return replace(
            self,
            games=games,
            team_epa=self.team_epa.filter(pl.col("game_id").is_in(list(played))),
            qb_games=self.qb_games.filter(pl.col("game_id").is_in(list(played))),
            snaps=self.snaps.filter(pl.col("game_id").is_in(list(played))),
            injuries=self.injuries.join(open_weeks, on=["season", "week"], how="semi"),
        )


def load_raw(stadiums: dict[str, Stadium], now: datetime) -> RawData:
    """Download/refresh all inputs. Completed seasons come from data/cache."""
    from seahawks_ml.ingest import nflverse, weather

    games = prepare_games(nflverse.load_schedules(), stadiums)
    current = int(games.filter(pl.col("kickoff_utc") <= now)["season"].max())
    seasons = list(range(FIRST_RATING_SEASON, current + 1))
    tables = nflverse.load_season_tables(seasons, current_season=current)
    return RawData(
        games=games,
        team_epa=tables["team_epa"],
        qb_games=tables["qb_games"],
        injuries=tables["injuries"],
        snaps=tables["snaps"],
        players=nflverse.load_players(),
        weather=weather.update_archive_cache(games, stadiums, now),
    )
```

- [ ] **Step 5: Run it to verify it passes**

Run: `uv run pytest tests/test_data.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add seahawks_ml/data.py tests/synthetic.py tests/test_data.py
git commit -m "feat: RawData with as-of truncation; synthetic test data"
```

---

### Task 7: Elo ratings

**Files:**
- Create/replace: `seahawks_ml/features/elo.py`
- Test: `tests/test_elo.py`

In-house Elo (K=20, home field 48 Elo points, one-third regression to the mean each offseason, margin-of-victory multiplier). Used as a feature and as a backtest baseline (`elo_win_prob`).

- [ ] **Step 1: Write the failing test**

`tests/test_elo.py`:

```python
from datetime import UTC, datetime

import polars as pl
import pytest

from seahawks_ml.features.base import GAMES_SCHEMA
from seahawks_ml.features.elo import EloParams, compute_elo, elo_win_prob


def _game(gid, season, day, home, away, margin, neutral=False):
    return {"game_id": gid, "season": season, "week": 1, "game_type": "REG",
            "kickoff_utc": datetime(season, 9, day, 17, tzinfo=UTC), "home_team": home,
            "away_team": away, "home_score": None, "away_score": None, "margin": margin,
            "neutral": neutral, "roof": "outdoors", "stadium_id": "SEA00", "home_rest": 7,
            "away_rest": 7, "div_game": False, "home_qb_id": None, "away_qb_id": None,
            "home_coach": None, "away_coach": None, "spread_line": None}


def test_elo_win_prob_includes_home_field():
    assert elo_win_prob(0.0, neutral=True) == pytest.approx(0.5)
    assert elo_win_prob(0.0, neutral=False) > 0.5


def test_compute_elo_updates_after_result_and_is_zero_sum():
    games = pl.DataFrame([
        _game("g1", 2020, 1, "SEA", "SF", 14),
        _game("g2", 2020, 8, "SF", "SEA", None),
    ], schema=GAMES_SCHEMA)
    out = {r["game_id"]: r for r in compute_elo(games).iter_rows(named=True)}
    assert out["g1"]["elo_home_pre"] == 1500 and out["g1"]["elo_away_pre"] == 1500
    sea_after, sf_after = out["g2"]["elo_away_pre"], out["g2"]["elo_home_pre"]
    assert sea_after > 1500 > sf_after
    assert sea_after + sf_after == pytest.approx(3000)


def test_compute_elo_regresses_toward_mean_between_seasons():
    games = pl.DataFrame([
        _game("g1", 2020, 1, "SEA", "SF", 30),
        _game("g2", 2021, 1, "SEA", "SF", None),
    ], schema=GAMES_SCHEMA)
    kept = compute_elo(games, EloParams(revert=0.0)).row(1, named=True)["elo_home_pre"]
    reverted = compute_elo(games, EloParams(revert=1 / 3)).row(1, named=True)["elo_home_pre"]
    assert reverted - 1500 == pytest.approx((kept - 1500) * (2 / 3))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_elo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.features.elo'`

- [ ] **Step 3: Write the implementation**

`seahawks_ml/features/elo.py`:

```python
"""In-house Elo with margin-of-victory multiplier and preseason regression."""

import math
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0
    hfa: float = 48.0  # Elo points (~1.7 game points)
    revert: float = 1 / 3  # share pulled back to the mean each offseason
    base: float = 1500.0


def elo_win_prob(elo_diff: float, neutral: bool, params: EloParams = EloParams()) -> float:
    """Home win probability from home-minus-away Elo."""
    diff = elo_diff + (0.0 if neutral else params.hfa)
    return 1.0 / (1.0 + 10 ** (-diff / 400))


def _mov_multiplier(margin: int, winner_elo_diff: float) -> float:
    if margin == 0:
        return 1.0
    return math.log(abs(margin) + 1) * 2.2 / (winner_elo_diff * 0.001 + 2.2)


def compute_elo(games: pl.DataFrame, params: EloParams = EloParams()) -> pl.DataFrame:
    """Pre-game Elo for both teams. Games without a result don't update ratings."""
    ratings: dict[str, float] = {}
    rating_season: dict[str, int] = {}
    out = []
    for g in games.sort("kickoff_utc", "game_id").iter_rows(named=True):
        for team in (g["home_team"], g["away_team"]):
            if team not in ratings:
                ratings[team], rating_season[team] = params.base, g["season"]
            elif rating_season[team] != g["season"]:
                ratings[team] = params.base + (ratings[team] - params.base) * (1 - params.revert)
                rating_season[team] = g["season"]
        home, away = ratings[g["home_team"]], ratings[g["away_team"]]
        out.append({"game_id": g["game_id"], "elo_home_pre": home, "elo_away_pre": away})
        margin = g["margin"]
        if margin is None:
            continue
        diff = home - away + (0.0 if g["neutral"] else params.hfa)
        expected = 1.0 / (1.0 + 10 ** (-diff / 400))
        actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        winner_diff = diff if margin > 0 else -diff
        delta = params.k * _mov_multiplier(margin, winner_diff) * (actual - expected)
        ratings[g["home_team"]] += delta
        ratings[g["away_team"]] -= delta
    return pl.DataFrame(out, schema={"game_id": pl.Utf8, "elo_home_pre": pl.Float64, "elo_away_pre": pl.Float64})
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_elo.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add seahawks_ml/features/elo.py tests/test_elo.py
git commit -m "feat: Elo ratings"
```

---

### Task 8: New-head-coach flag

**Files:**
- Create/replace: `seahawks_ml/features/coaching.py`
- Test: `tests/test_coaching.py`

Spec: 1 when a team's coach differs from its coach in the final game of last season; constant within the season (mid-season changes are out of scope).

- [ ] **Step 1: Write the failing test**

`tests/test_coaching.py`:

```python
import polars as pl

from seahawks_ml.features.base import team_games
from seahawks_ml.features.coaching import new_head_coach
from tests.synthetic import make_raw


def test_new_head_coach_flags_first_season_after_change_only():
    raw = make_raw(seasons=(2011, 2012, 2013))
    flags = new_head_coach(team_games(raw.games)).join(
        team_games(raw.games).select("game_id", "team", "season"), on=["game_id", "team"]
    )
    by = flags.group_by("team", "season").agg(pl.col("new_head_coach").max()).sort("team", "season")
    lookup = {(r["team"], r["season"]): r["new_head_coach"] for r in by.iter_rows(named=True)}
    assert lookup[("SF", 2011)] == 0  # no previous season in data
    assert lookup[("SF", 2012)] == 1  # Coach A -> Coach B
    assert lookup[("SF", 2013)] == 0
    assert lookup[("SEA", 2012)] == 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_coaching.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.features.coaching'`

- [ ] **Step 3: Write the implementation**

`seahawks_ml/features/coaching.py`:

```python
"""Regime-change flag: did the team change head coach since last season's final game?"""

import polars as pl


def new_head_coach(team_games: pl.DataFrame) -> pl.DataFrame:
    """Per (game_id, team): 1 if the team's first coach this season differs from the coach of
    its final game last season. Constant within a season (mid-season changes not flagged)."""
    first: dict[tuple[str, int], str | None] = {}
    last: dict[tuple[str, int], str | None] = {}
    for r in team_games.sort("kickoff_utc", "game_id").iter_rows(named=True):
        key = (r["team"], r["season"])
        first.setdefault(key, r["coach"])
        last[key] = r["coach"]
    rows = []
    for r in team_games.iter_rows(named=True):
        prev = last.get((r["team"], r["season"] - 1))
        cur = first[(r["team"], r["season"])]
        flag = int(prev is not None and cur is not None and cur != prev)
        rows.append({"game_id": r["game_id"], "team": r["team"], "new_head_coach": flag})
    return pl.DataFrame(rows, schema={"game_id": pl.Utf8, "team": pl.Utf8, "new_head_coach": pl.Int64})
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_coaching.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add seahawks_ml/features/coaching.py tests/test_coaching.py
git commit -m "feat: new head coach flag"
```

---

### Task 9: Team EPA ratings with shrinkage

**Files:**
- Create/replace: `seahawks_ml/features/ratings.py`
- Test: `tests/test_ratings.py`

Offense/defense EPA per play, anchored on last season shrunk toward the league, then blended with current-season games. New-coach teams use a smaller prior weight so they move off last season faster. Values are deviations from the *previous* season's league average, so no same-season league information leaks in.

- [ ] **Step 1: Write the failing test**

`tests/test_ratings.py`:

```python
from datetime import UTC, datetime

import polars as pl
import pytest

from seahawks_ml.features.base import TEAM_GAMES_SCHEMA
from seahawks_ml.features.ratings import RatingParams, compute_team_ratings


def _tg(game_id, season, day, team, opp):
    return {"game_id": game_id, "season": season, "week": 1,
            "kickoff_utc": datetime(season, 9, day, 17, tzinfo=UTC), "team": team, "opponent": opp,
            "is_home": True, "points_for": 0, "points_against": 0, "coach": "c", "qb_id": None}


def _setup():
    tg = pl.DataFrame([
        _tg("a", 2020, 1, "SEA", "SF"), _tg("a", 2020, 1, "SF", "SEA"),
        _tg("b", 2021, 1, "SEA", "SF"), _tg("b", 2021, 1, "SF", "SEA"),
        _tg("c", 2021, 8, "SEA", "SF"), _tg("c", 2021, 8, "SF", "SEA"),
    ], schema=TEAM_GAMES_SCHEMA)
    epa = pl.DataFrame({
        "game_id": ["a", "a", "b", "b"], "season": [2020, 2020, 2021, 2021],
        "team": ["SEA", "SF", "SEA", "SF"], "opponent": ["SF", "SEA", "SF", "SEA"],
        "epa_sum": [12.0, -6.0, 30.0, 0.0], "plays": [60, 60, 60, 60],
    })
    return tg, epa


def _flags(tg, sf_new=0):
    return tg.select("game_id", "team").with_columns(
        pl.when((pl.col("team") == "SF") & (pl.col("game_id") != "a")).then(sf_new).otherwise(0)
        .alias("new_head_coach"))


def test_ratings_use_shrunk_prior_then_blend_current_season():
    tg, epa = _setup()
    out = compute_team_ratings(tg, epa, _flags(tg), RatingParams(0.5, 4.0, 2.0))
    r = {(x["game_id"], x["team"]): x for x in out.iter_rows(named=True)}
    # 2020 has no prior season -> zero
    assert r[("a", "SEA")]["off_rating"] == 0.0
    # 2020: SEA off 0.2, SF off -0.1, league 0.05. SEA prior off = 0.5*(0.2-0.05) = 0.075
    assert r[("b", "SEA")]["off_rating"] == pytest.approx(0.075)
    assert r[("b", "SEA")]["def_rating"] == pytest.approx(0.5 * (-0.1 - 0.05))
    # game c blends one 2021 game: SEA off 0.5, centered on 2020 league 0.05 -> 0.45
    assert r[("c", "SEA")]["off_rating"] == pytest.approx((4 * 0.075 + 0.45) / 5)


def test_new_head_coach_discounts_prior_faster():
    tg, epa = _setup()
    params = RatingParams(0.5, 4.0, 1.0)
    same = compute_team_ratings(tg, epa, _flags(tg, 0), params)
    new = compute_team_ratings(tg, epa, _flags(tg, 1), params)

    def get(df):
        return df.filter((pl.col("game_id") == "c") & (pl.col("team") == "SF"))["off_rating"][0]

    # SF prior off = 0.5*(-0.1-0.05) = -0.075; 2021 game dev = 0.0-0.05 = -0.05
    assert get(same) == pytest.approx((4 * -0.075 - 0.05) / 5)
    assert get(new) == pytest.approx((1 * -0.075 - 0.05) / 2)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_ratings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.features.ratings'`

- [ ] **Step 3: Write the implementation**

`seahawks_ml/features/ratings.py`:

```python
"""Pre-game offensive/defensive EPA-per-play ratings with a shrunk prior-season anchor.

All values are deviations from the previous season's league average, so they're
comparable across eras and use no same-season league information.

    prior   = prior_regression * (team_prev_season_avg - league_prev_season_avg)
    rating  = (k * prior + sum(current_season_devs)) / (k + n_current_games)

k is `prior_games`, or `prior_games_new_coach` when the team has a new head coach,
so new regimes move off last season's numbers faster.
"""

from collections import defaultdict
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class RatingParams:
    prior_regression: float = 0.6
    prior_games: float = 4.0
    prior_games_new_coach: float = 2.0


def _per_game_epa(team_epa: pl.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    """(game_id, team) -> (offense EPA/play, defense EPA/play allowed)."""
    off = {(r["game_id"], r["team"]): r["epa_sum"] / r["plays"]
           for r in team_epa.iter_rows(named=True) if r["plays"]}
    out = {}
    for r in team_epa.iter_rows(named=True):
        key = (r["game_id"], r["team"])
        allowed = off.get((r["game_id"], r["opponent"]))
        if key in off and allowed is not None:
            out[key] = (off[key], allowed)
    return out


def compute_team_ratings(
    team_games: pl.DataFrame,
    team_epa: pl.DataFrame,
    coach_flags: pl.DataFrame,
    params: RatingParams = RatingParams(),
) -> pl.DataFrame:
    epa = _per_game_epa(team_epa)
    season_of = {r["game_id"]: r["season"] for r in team_games.iter_rows(named=True)}

    league_sum: dict[int, float] = defaultdict(float)
    league_n: dict[int, int] = defaultdict(int)
    team_sums: dict[tuple[str, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    for (game_id, team), (o, d) in epa.items():
        season = season_of.get(game_id)
        if season is None:
            continue
        league_sum[season] += o
        league_n[season] += 1
        s = team_sums[(team, season)]
        s[0] += o
        s[1] += d
        s[2] += 1

    def prior(team: str, season: int) -> tuple[float, float]:
        prev = team_sums.get((team, season - 1))
        if not prev or not league_n.get(season - 1):
            return 0.0, 0.0
        lg = league_sum[season - 1] / league_n[season - 1]
        return (params.prior_regression * (prev[0] / prev[2] - lg),
                params.prior_regression * (prev[1] / prev[2] - lg))

    def center(season: int) -> float:
        n = league_n.get(season - 1)
        return league_sum[season - 1] / n if n else 0.0

    new_coach = {(r["game_id"], r["team"]): r["new_head_coach"] for r in coach_flags.iter_rows(named=True)}
    running: dict[tuple[str, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    rows = []
    for r in team_games.sort("kickoff_utc", "game_id").iter_rows(named=True):
        team, season, game_id = r["team"], r["season"], r["game_id"]
        k = params.prior_games_new_coach if new_coach.get((game_id, team)) else params.prior_games
        p_off, p_def = prior(team, season)
        cur = running[(team, season)]
        rows.append({
            "game_id": game_id,
            "team": team,
            "off_rating": (k * p_off + cur[0]) / (k + cur[2]),
            "def_rating": (k * p_def + cur[1]) / (k + cur[2]),
        })
        if (game_id, team) in epa:
            o, d = epa[(game_id, team)]
            c = center(season)
            cur[0] += o - c
            cur[1] += d - c
            cur[2] += 1
    return pl.DataFrame(rows, schema={"game_id": pl.Utf8, "team": pl.Utf8,
                                      "off_rating": pl.Float64, "def_rating": pl.Float64})
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_ratings.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add seahawks_ml/features/ratings.py tests/test_ratings.py
git commit -m "feat: team EPA ratings with new-coach shrinkage"
```

---

### Task 10: Starting-QB ratings with draft-capital prior

**Files:**
- Create/replace: `seahawks_ml/features/qb.py`
- Test: `tests/test_qb.py`

Each starter's EPA/dropback and CPOE over his last three seasons (before kickoff), shrunk toward the average early-career performance of his draft bucket (`round_1`, `day_2`, `day_3_udfa`). Bucket priors use only seasons before the game's season.

- [ ] **Step 1: Write the failing test**

`tests/test_qb.py`:

```python
from datetime import UTC, datetime

import polars as pl
import pytest

from seahawks_ml.features.base import GAMES_SCHEMA
from seahawks_ml.features.qb import QBParams, compute_qb_features, draft_bucket
from seahawks_ml.ingest.nflverse import PLAYERS_SCHEMA, QB_GAMES_SCHEMA


def test_draft_bucket():
    assert draft_bucket(1) == "round_1"
    assert draft_bucket(2) == draft_bucket(3) == "day_2"
    assert draft_bucket(5) == draft_bucket(None) == "day_3_udfa"


def _game(gid, season, day, home_qb, away_qb):
    return {"game_id": gid, "season": season, "week": 1, "game_type": "REG",
            "kickoff_utc": datetime(season, 9, day, 17, tzinfo=UTC), "home_team": "SEA",
            "away_team": "SF", "home_score": 0, "away_score": 0, "margin": 0, "neutral": False,
            "roof": "outdoors", "stadium_id": "SEA00", "home_rest": 7, "away_rest": 7,
            "div_game": True, "home_qb_id": home_qb, "away_qb_id": away_qb, "home_coach": None,
            "away_coach": None, "spread_line": None}


def _setup():
    games = pl.DataFrame([
        _game("g1", 2020, 1, "VET", "ROOK1"),
        _game("g2", 2021, 1, "VET", "ROOK2"),
        _game("g3", 2021, 8, "VET", "ROOK2"),
    ], schema=GAMES_SCHEMA)
    qb = pl.DataFrame([
        {"game_id": "g1", "season": 2020, "team": "SEA", "qb_id": "VET", "dropbacks": 40,
         "qb_epa_sum": 8.0, "cpoe_sum": 40.0, "cpoe_n": 30},
        # 600 early-career round-1 dropbacks at +0.1/db in 2020 sets the 2021 round_1 prior
        {"game_id": "g1", "season": 2020, "team": "SF", "qb_id": "ROOK1", "dropbacks": 600,
         "qb_epa_sum": 60.0, "cpoe_sum": 0.0, "cpoe_n": 500},
        {"game_id": "g2", "season": 2021, "team": "SF", "qb_id": "ROOK2", "dropbacks": 50,
         "qb_epa_sum": -10.0, "cpoe_sum": 0.0, "cpoe_n": 40},
    ], schema=QB_GAMES_SCHEMA)
    players = pl.DataFrame([
        {"gsis_id": "VET", "pfr_id": None, "position": "QB", "draft_round": 2, "rookie_season": 2010},
        {"gsis_id": "ROOK1", "pfr_id": None, "position": "QB", "draft_round": 1, "rookie_season": 2020},
        {"gsis_id": "ROOK2", "pfr_id": None, "position": "QB", "draft_round": 1, "rookie_season": 2021},
    ], schema=PLAYERS_SCHEMA)
    return games, qb, players


def test_rookie_rating_starts_at_bucket_prior_then_updates():
    games, qb, players = _setup()
    out = {r["game_id"]: r for r in compute_qb_features(games, qb, players, QBParams(250, 3)).iter_rows(named=True)}
    assert out["g2"]["away_qb_bucket"] == "round_1"
    assert out["g2"]["away_qb_epa"] == pytest.approx(0.1)  # pure prior: no dropbacks yet
    assert out["g3"]["away_qb_epa"] == pytest.approx((250 * 0.1 - 10.0) / 300)


def test_rating_uses_only_games_before_kickoff():
    games, qb, players = _setup()
    out = {r["game_id"]: r for r in compute_qb_features(games, qb, players, QBParams(250, 3)).iter_rows(named=True)}
    # in 2020 nothing is known: prior defaults to 0.0 and no history before g1
    assert out["g1"]["home_qb_epa"] == 0.0
    # VET's 2021 games see the 40 dropbacks from 2020, shrunk to the day_2 prior (pooled 0.1)
    assert out["g2"]["home_qb_epa"] == pytest.approx((250 * 0.1 + 8.0) / 290)
    assert out["g2"]["home_qb_bucket"] == "day_2"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_qb.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.features.qb'`

- [ ] **Step 3: Write the implementation**

`seahawks_ml/features/qb.py`:

```python
"""Starting-QB ratings shrunk toward a draft-capital prior.

    rating = (prior_dropbacks * bucket_prior + sum(qb_epa)) / (prior_dropbacks + dropbacks)

History is the QB's dropbacks before kickoff within the last `window_seasons` seasons.
Bucket priors are the average early-career (first two seasons) performance of QBs in
that draft bucket, using only seasons before the game's season (expanding window,
so they never see the season being predicted).
"""

from collections import defaultdict
from dataclasses import dataclass

import polars as pl

DRAFT_BUCKETS = ("round_1", "day_2", "day_3_udfa")
MIN_PRIOR_DROPBACKS = 500  # below this, a bucket falls back to the pooled prior


@dataclass(frozen=True)
class QBParams:
    prior_dropbacks: float = 250.0
    window_seasons: int = 3


def draft_bucket(draft_round: int | None) -> str:
    if draft_round == 1:
        return "round_1"
    if draft_round in (2, 3):
        return "day_2"
    return "day_3_udfa"


def _bucket_priors(qb_rows: list[dict], player_info: dict, seasons: list[int]) -> dict:
    """season -> bucket -> (epa_per_dropback, cpoe) using seasons strictly before it."""
    per_season: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0, 0]))
    for r in qb_rows:
        info = player_info.get(r["qb_id"])
        if info is None or info["rookie_season"] is None or r["season"] - info["rookie_season"] > 1:
            continue
        acc = per_season[r["season"]][draft_bucket(info["draft_round"])]
        acc[0] += r["qb_epa_sum"]
        acc[1] += r["dropbacks"]
        acc[2] += r["cpoe_sum"]
        acc[3] += r["cpoe_n"]
    priors = {}
    running: dict[str, list[float]] = defaultdict(lambda: [0.0, 0, 0.0, 0])
    for season in sorted(set(seasons)):
        pooled = [sum(running[b][i] for b in DRAFT_BUCKETS) for i in range(4)]
        pooled_val = (pooled[0] / pooled[1] if pooled[1] else 0.0,
                      pooled[2] / pooled[3] if pooled[3] else 0.0)
        priors[season] = {
            b: ((running[b][0] / running[b][1], running[b][2] / running[b][3] if running[b][3] else 0.0)
                if running[b][1] >= MIN_PRIOR_DROPBACKS else pooled_val)
            for b in DRAFT_BUCKETS
        }
        for b, acc in per_season.get(season, {}).items():
            for i in range(4):
                running[b][i] += acc[i]
    return priors


def compute_qb_features(
    games: pl.DataFrame,
    qb_games: pl.DataFrame,
    players: pl.DataFrame,
    params: QBParams = QBParams(),
) -> pl.DataFrame:
    """Per game: home/away starter EPA and CPOE ratings and draft bucket."""
    kickoff = {r["game_id"]: r["kickoff_utc"] for r in games.iter_rows(named=True)}
    player_info = {r["gsis_id"]: r for r in players.iter_rows(named=True)}
    qb_rows = [r | {"kickoff_utc": kickoff[r["game_id"]]}
               for r in qb_games.iter_rows(named=True) if r["game_id"] in kickoff]
    history: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(qb_rows, key=lambda x: x["kickoff_utc"]):
        history[r["qb_id"]].append(r)
    priors = _bucket_priors(qb_rows, player_info, games["season"].to_list())

    def rate(qb_id: str | None, season: int, ko) -> tuple[float, float, str]:
        info = player_info.get(qb_id) if qb_id else None
        bucket = draft_bucket(info["draft_round"] if info else None)
        p_epa, p_cpoe = priors[season][bucket]
        n = epa = c_sum = c_n = 0.0
        for h in history.get(qb_id, []):
            if h["kickoff_utc"] >= ko:
                break
            if h["season"] > season - params.window_seasons:
                n += h["dropbacks"]
                epa += h["qb_epa_sum"]
                c_sum += h["cpoe_sum"]
                c_n += h["cpoe_n"]
        k = params.prior_dropbacks
        return (k * p_epa + epa) / (k + n), (k * p_cpoe + c_sum) / (k + c_n), bucket

    rows = []
    for g in games.iter_rows(named=True):
        h_epa, h_cpoe, h_b = rate(g["home_qb_id"], g["season"], g["kickoff_utc"])
        a_epa, a_cpoe, a_b = rate(g["away_qb_id"], g["season"], g["kickoff_utc"])
        rows.append({"game_id": g["game_id"], "home_qb_epa": h_epa, "away_qb_epa": a_epa,
                     "home_qb_cpoe": h_cpoe, "away_qb_cpoe": a_cpoe,
                     "home_qb_bucket": h_b, "away_qb_bucket": a_b})
    return pl.DataFrame(rows, schema={
        "game_id": pl.Utf8, "home_qb_epa": pl.Float64, "away_qb_epa": pl.Float64,
        "home_qb_cpoe": pl.Float64, "away_qb_cpoe": pl.Float64,
        "home_qb_bucket": pl.Utf8, "away_qb_bucket": pl.Utf8,
    })
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_qb.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add seahawks_ml/features/qb.py tests/test_qb.py
git commit -m "feat: QB ratings with draft-capital prior"
```

---

### Task 11: Situational features

**Files:**
- Create/replace: `seahawks_ml/features/situational.py`
- Test: `tests/test_situational.py`

Rest, bye/short-week flags, travel miles, time-zone shift, body clock (kickoff hour in each team's home time zone), primetime, division game, home field with a season trend, and the 2020 no-crowd flag. A team's home base each season is its most-used non-neutral home stadium (handles relocations like STL → LA).

- [ ] **Step 1: Write the failing test**

`tests/test_situational.py`:

```python
from datetime import UTC, datetime

import polars as pl
import pytest

from seahawks_ml.features.base import GAMES_SCHEMA
from seahawks_ml.features.situational import (
    compute_situational,
    haversine_miles,
    team_home_stadiums,
)
from seahawks_ml.stadiums import load_stadiums


def _game(gid, home, away, stadium, ko, neutral=False, home_rest=7, away_rest=7, season=2024):
    return {"game_id": gid, "season": season, "week": 1, "game_type": "REG", "kickoff_utc": ko,
            "home_team": home, "away_team": away, "home_score": None, "away_score": None,
            "margin": None, "neutral": neutral, "roof": "outdoors", "stadium_id": stadium,
            "home_rest": home_rest, "away_rest": away_rest, "div_game": True, "home_qb_id": None,
            "away_qb_id": None, "home_coach": None, "away_coach": None, "spread_line": None}


def _games():
    return pl.DataFrame([
        # Seattle at Miami, 1pm ET (10am PT body clock for SEA)
        _game("mia", "MIA", "SEA", "MIA00", datetime(2024, 10, 6, 17, 0, tzinfo=UTC), away_rest=14),
        _game("sea", "SEA", "MIA", "SEA00", datetime(2024, 10, 13, 20, 5, tzinfo=UTC), home_rest=4),
        # London neutral-site game, SEA listed as home
        _game("lon", "SEA", "MIA", "LON02", datetime(2024, 10, 20, 13, 30, tzinfo=UTC), neutral=True),
        # Sunday night game in Seattle
        _game("snf", "SEA", "MIA", "SEA00", datetime(2024, 10, 28, 0, 20, tzinfo=UTC)),
    ], schema=GAMES_SCHEMA)


def test_haversine_seattle_to_miami():
    assert haversine_miles(47.595, -122.332, 25.958, -80.239) == pytest.approx(2720, rel=0.02)


def test_team_home_stadiums_ignores_neutral_sites():
    home = team_home_stadiums(_games())
    assert home[("SEA", 2024)] == "SEA00" and home[("MIA", 2024)] == "MIA00"


def test_compute_situational_values():
    out = {r["game_id"]: r for r in compute_situational(_games(), load_stadiums()).iter_rows(named=True)}
    mia = out["mia"]
    assert mia["away_body_clock"] == pytest.approx(10.0)  # 1pm ET = 10am PT
    assert mia["home_body_clock"] == pytest.approx(13.0)
    assert mia["away_tz_shift"] == pytest.approx(3.0) and mia["home_tz_shift"] == 0.0
    assert mia["away_post_bye"] == 1 and mia["rest_diff"] == -7
    assert mia["travel_diff"] == pytest.approx(-2720, rel=0.02)
    assert out["sea"]["home_short_week"] == 1
    lon = out["lon"]
    assert lon["home_field"] == 0 and lon["hfa_trend"] == 0.0
    assert lon["home_tz_shift"] == pytest.approx(8.0)  # BST vs PDT
    assert out["snf"]["primetime"] == 1 and out["sea"]["primetime"] == 0
    assert out["sea"]["hfa_trend"] == pytest.approx(0.9)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_situational.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.features.situational'`

- [ ] **Step 3: Write the implementation**

`seahawks_ml/features/situational.py`:

```python
"""Rest, travel, time zones, body clock, primetime and home-field features."""

import math
from collections import Counter
from zoneinfo import ZoneInfo

import polars as pl

from seahawks_ml.features.base import ET
from seahawks_ml.stadiums import Stadium

HFA_TREND_BASE_SEASON = 2015
NO_CROWD_SEASON = 2020


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def team_home_stadiums(games: pl.DataFrame) -> dict[tuple[str, int], str]:
    """(team, season) -> most common non-neutral home stadium, filled from nearest season."""
    counts: dict[tuple[str, int], Counter] = {}
    for g in games.filter(~pl.col("neutral")).iter_rows(named=True):
        counts.setdefault((g["home_team"], g["season"]), Counter())[g["stadium_id"]] += 1
    home = {k: c.most_common(1)[0][0] for k, c in counts.items()}
    teams = set(games["home_team"].to_list()) | set(games["away_team"].to_list())
    seasons = sorted(set(games["season"].to_list()))
    for team in teams:
        known = sorted(s for (t, s) in home if t == team)
        for season in seasons:
            if (team, season) not in home and known:
                nearest = min(known, key=lambda s: (abs(s - season), -s))
                home[(team, season)] = home[(team, nearest)]
    return home


def _offset_hours(tz: str, when) -> float:
    return when.astimezone(ZoneInfo(tz)).utcoffset().total_seconds() / 3600


def compute_situational(games: pl.DataFrame, stadiums: dict[str, Stadium]) -> pl.DataFrame:
    home_of = team_home_stadiums(games)
    rows = []
    for g in games.iter_rows(named=True):
        venue = stadiums[g["stadium_id"]]
        ko = g["kickoff_utc"]
        venue_offset = _offset_hours(venue.tz, ko)
        row = {"game_id": g["game_id"]}
        for side in ("home", "away"):
            base = stadiums[home_of[(g[f"{side}_team"], g["season"])]]
            rest = g[f"{side}_rest"] if g[f"{side}_rest"] is not None else 7
            local = ko.astimezone(ZoneInfo(base.tz))
            row[f"{side}_rest"] = rest
            row[f"{side}_post_bye"] = int(rest >= 13)
            row[f"{side}_short_week"] = int(rest <= 5)
            row[f"{side}_travel_miles"] = haversine_miles(base.lat, base.lon, venue.lat, venue.lon)
            row[f"{side}_tz_shift"] = venue_offset - _offset_hours(base.tz, ko)
            row[f"{side}_body_clock"] = local.hour + local.minute / 60
        home_field = 0 if g["neutral"] else 1
        et = ko.astimezone(ET)
        rows.append(row | {
            "home_field": home_field,
            "hfa_trend": home_field * (g["season"] - HFA_TREND_BASE_SEASON) / 10,
            "no_crowd": int(g["season"] == NO_CROWD_SEASON),
            "rest_diff": row["home_rest"] - row["away_rest"],
            "travel_diff": row["home_travel_miles"] - row["away_travel_miles"],
            "primetime": int(et.hour >= 19),
            "div_game": int(bool(g["div_game"])),
        })
    out = pl.DataFrame(rows)
    return out.drop("home_rest", "away_rest", "home_travel_miles", "away_travel_miles")
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_situational.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add seahawks_ml/features/situational.py tests/test_situational.py
git commit -m "feat: rest, travel, time-zone and body-clock features"
```

---

### Task 12: Season-timing features

**Files:**
- Create/replace: `seahawks_ml/features/season_timing.py`
- Test: `tests/test_season_timing.py`

`week_number`, `is_final_regular_week` (Week 17 through 2020, Week 18 from 2021, derived from each season's schedule), `is_playoff`.

- [ ] **Step 1: Write the failing test**

`tests/test_season_timing.py`:

```python
import polars as pl

from seahawks_ml.features.season_timing import compute_season_timing


def test_final_regular_week_tracks_17_and_18_game_eras():
    games = pl.DataFrame({
        "game_id": ["a", "b", "c", "d", "e"],
        "season": [2020, 2020, 2020, 2021, 2021],
        "week": [16, 17, 18, 17, 18],
        "game_type": ["REG", "REG", "WC", "REG", "REG"],
    })
    out = {r["game_id"]: r for r in compute_season_timing(games).iter_rows(named=True)}
    assert out["b"]["is_final_regular_week"] == 1 and out["a"]["is_final_regular_week"] == 0
    assert out["c"]["is_final_regular_week"] == 0 and out["c"]["is_playoff"] == 1
    assert out["e"]["is_final_regular_week"] == 1 and out["d"]["is_final_regular_week"] == 0
    assert out["e"]["week_number"] == 18
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_season_timing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.features.season_timing'`

- [ ] **Step 3: Write the implementation**

`seahawks_ml/features/season_timing.py`:

```python
"""Week number, final-regular-week and playoff flags."""

import polars as pl


def compute_season_timing(games: pl.DataFrame) -> pl.DataFrame:
    final_week = (
        games.filter(pl.col("game_type") == "REG")
        .group_by("season")
        .agg(pl.col("week").max().alias("final_week"))
    )
    return (
        games.join(final_week, on="season", how="left")
        .select(
            "game_id",
            pl.col("week").alias("week_number"),
            ((pl.col("game_type") == "REG") & (pl.col("week") == pl.col("final_week")))
            .cast(pl.Int64).alias("is_final_regular_week"),
            (pl.col("game_type") != "REG").cast(pl.Int64).alias("is_playoff"),
        )
    )
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_season_timing.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add seahawks_ml/features/season_timing.py tests/test_season_timing.py
git commit -m "feat: week number and final-week flags"
```

---

### Task 13: Starter availability

**Files:**
- Create/replace: `seahawks_ml/features/availability.py`
- Test: `tests/test_availability.py`

Snap-share-weighted count of starters (≥50% of offense or defense snaps over the previous four team games, at least two appearances) listed Out or Doubtful on that week's injury report. QBs are excluded (the QB features handle starter changes).

- [ ] **Step 1: Write the failing test**

`tests/test_availability.py`:

```python
import polars as pl
import pytest

from seahawks_ml.features.availability import compute_availability
from seahawks_ml.ingest.nflverse import INJURIES_SCHEMA
from tests.synthetic import make_raw


def test_availability_is_null_before_snap_era():
    raw = make_raw()
    out = compute_availability(raw.games, raw.snaps, raw.injuries, raw.players)
    early = out.join(raw.games.select("game_id", "season"), on="game_id").filter(pl.col("season") < 2013)
    assert early["home_off_out"].null_count() == early.height


def test_availability_weights_out_starters_by_snap_share():
    raw = make_raw()
    game = raw.games.filter((pl.col("season") == 2014) & (pl.col("week") == 5)).row(0, named=True)
    team = game["home_team"]
    base = {"season": 2014, "week": 5, "team": team}
    injuries = pl.DataFrame([
        base | {"gsis_id": f"{team}-P0", "position": "WR", "report_status": "Out"},
        base | {"gsis_id": f"{team}-P7", "position": "LB", "report_status": "Doubtful"},
        base | {"gsis_id": f"{team}-P8", "position": "LB", "report_status": "Questionable"},
    ], schema=INJURIES_SCHEMA)
    out = compute_availability(raw.games, raw.snaps, injuries, raw.players)
    row = out.filter(pl.col("game_id") == game["game_id"]).row(0, named=True)
    assert row["home_off_out"] == pytest.approx(0.9)
    assert row["home_def_out"] == pytest.approx(0.9)
    assert row["away_off_out"] == 0.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_availability.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.features.availability'`

- [ ] **Step 3: Write the implementation**

`seahawks_ml/features/availability.py`:

```python
"""Starter availability from snap shares and the final pre-game injury report.

A "starter" is a non-QB who averaged >= 50% of offense (or defense) snaps in the
games he appeared in among his team's previous `lookback` games (min 2 appearances).
The feature is the snap-share-weighted count of starters listed Out or Doubtful.
QBs are excluded because the QB features already handle starter changes.
Null before the first season with snap counts.
"""

from collections import defaultdict

import polars as pl

from seahawks_ml.config import FIRST_SNAP_SEASON

OUT_STATUSES = {"Out", "Doubtful"}
STARTER_SHARE = 0.5
MIN_APPEARANCES = 2


def compute_availability(
    games: pl.DataFrame,
    snaps: pl.DataFrame,
    injuries: pl.DataFrame,
    players: pl.DataFrame,
    lookback: int = 4,
) -> pl.DataFrame:
    pfr_to_gsis = {r["pfr_id"]: r["gsis_id"] for r in players.iter_rows(named=True) if r["pfr_id"]}
    kickoff = {r["game_id"]: r["kickoff_utc"] for r in games.iter_rows(named=True)}

    team_game_snaps: dict[tuple[str, str], list[tuple[str, float, float]]] = defaultdict(list)
    for r in snaps.filter(pl.col("position") != "QB").iter_rows(named=True):
        gsis = pfr_to_gsis.get(r["pfr_player_id"])
        if gsis and r["game_id"] in kickoff:
            team_game_snaps[(r["team"], r["game_id"])].append(
                (gsis, r["offense_pct"] or 0.0, r["defense_pct"] or 0.0))
    team_history: dict[str, list[str]] = defaultdict(list)
    for team, game_id in sorted(team_game_snaps, key=lambda k: kickoff[k[1]]):
        team_history[team].append(game_id)

    out_lists: dict[tuple[int, int, str], set[str]] = defaultdict(set)
    for r in injuries.filter(pl.col("report_status").is_in(list(OUT_STATUSES))).iter_rows(named=True):
        out_lists[(r["season"], r["week"], r["team"])].add(r["gsis_id"])

    def team_out(team: str, game: dict) -> tuple[float | None, float | None]:
        if game["season"] < FIRST_SNAP_SEASON:
            return None, None
        prior = [gid for gid in team_history.get(team, []) if kickoff[gid] < game["kickoff_utc"]]
        recent = prior[-lookback:]
        if not recent:
            return None, None
        shares: dict[str, list[list[float]]] = defaultdict(lambda: [[], []])
        for gid in recent:
            for gsis, off, de in team_game_snaps[(team, gid)]:
                shares[gsis][0].append(off)
                shares[gsis][1].append(de)
        out_set = out_lists.get((game["season"], game["week"], team), set())
        off_out = def_out = 0.0
        for gsis, (offs, defs) in shares.items():
            if gsis not in out_set or len(offs) < MIN_APPEARANCES:
                continue
            off_avg, def_avg = sum(offs) / len(offs), sum(defs) / len(defs)
            if off_avg >= STARTER_SHARE:
                off_out += off_avg
            if def_avg >= STARTER_SHARE:
                def_out += def_avg
        return off_out, def_out

    rows = []
    for g in games.iter_rows(named=True):
        h_off, h_def = team_out(g["home_team"], g)
        a_off, a_def = team_out(g["away_team"], g)
        rows.append({"game_id": g["game_id"], "home_off_out": h_off, "home_def_out": h_def,
                     "away_off_out": a_off, "away_def_out": a_def})
    return pl.DataFrame(rows, schema={"game_id": pl.Utf8, "home_off_out": pl.Float64,
                                      "home_def_out": pl.Float64, "away_off_out": pl.Float64,
                                      "away_def_out": pl.Float64})
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_availability.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add seahawks_ml/features/availability.py tests/test_availability.py
git commit -m "feat: snap-weighted starter availability"
```

---

### Task 14: Weather features

**Files:**
- Create/replace: `seahawks_ml/features/weather.py`
- Test: `tests/test_weather_features.py`

Game-window mean temperature and wind and total precipitation. Indoor games get fixed neutral values. Outdoor games without cached hours (recent games inside the archive lag, or a failed fetch) fall back to the stadium's same-month climatology. `weather_source` records which was used.

- [ ] **Step 1: Write the failing test**

`tests/test_weather_features.py`:

```python
import polars as pl
import pytest

from seahawks_ml.features.weather import compute_weather
from tests.synthetic import make_raw


def test_indoor_games_get_neutral_weather():
    raw = make_raw()
    out = compute_weather(raw.games, raw.weather).join(raw.games.select("game_id", "roof"), on="game_id")
    indoor = out.filter(pl.col("roof").is_in(["dome", "closed"]))
    assert indoor.height > 0
    assert indoor["weather_source"].unique().to_list() == ["indoor"]
    assert indoor["temp_f"].unique().to_list() == [70.0]


def test_outdoor_games_average_game_window():
    raw = make_raw()
    g = raw.games.filter(pl.col("roof") == "outdoors").row(0, named=True)
    row = compute_weather(raw.games, raw.weather).filter(pl.col("game_id") == g["game_id"]).row(0, named=True)
    assert row["weather_source"] == "archive"
    assert row["temp_f"] == pytest.approx(60.0 - g["week"])
    assert row["wind_mph"] == pytest.approx(6.5)  # mean of 5,6,7,8


def test_missing_outdoor_weather_falls_back_to_climatology():
    raw = make_raw()
    g = raw.games.filter(pl.col("roof") == "outdoors").row(0, named=True)
    weather = raw.weather.filter(
        ~((pl.col("stadium_id") == g["stadium_id"]) & (pl.col("time_utc") >= g["kickoff_utc"])
          & (pl.col("time_utc") < g["kickoff_utc"] + pl.duration(hours=4)))
    )
    row = compute_weather(raw.games, weather).filter(pl.col("game_id") == g["game_id"]).row(0, named=True)
    assert row["weather_source"] == "climatology"
    assert 50 < row["temp_f"] < 60
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_weather_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.features.weather'`

- [ ] **Step 3: Write the implementation**

`seahawks_ml/features/weather.py`:

```python
"""Game-window weather features (mean temp/wind, total precipitation over 4 hours)."""

from collections import defaultdict
from datetime import timedelta

import polars as pl

from seahawks_ml.ingest.weather import GAME_WINDOW_HOURS

INDOOR_ROOFS = {"dome", "closed"}
INDOOR_VALUES = {"temp_f": 70.0, "wind_mph": 0.0, "precip_in": 0.0}


def compute_weather(games: pl.DataFrame, weather: pl.DataFrame) -> pl.DataFrame:
    """weather_source is one of indoor / archive / forecast / climatology."""
    by_hour = {(r["stadium_id"], r["time_utc"]): r for r in weather.iter_rows(named=True)}
    clim: dict[tuple[str, int], list[list[float]]] = defaultdict(lambda: [[], [], []])
    league_clim: dict[int, list[list[float]]] = defaultdict(lambda: [[], [], []])
    for r in weather.filter(pl.col("source") == "archive").iter_rows(named=True):
        for i, c in enumerate(("temp_f", "wind_mph", "precip_in")):
            if r[c] is not None:
                clim[(r["stadium_id"], r["time_utc"].month)][i].append(r[c])
                league_clim[r["time_utc"].month][i].append(r[c])

    def climatology(stadium_id: str, month: int) -> dict:
        vals = clim.get((stadium_id, month)) or league_clim.get(month)
        if not vals or not vals[0]:
            return dict(INDOOR_VALUES)
        temp, wind, precip = (sum(v) / len(v) if v else 0.0 for v in vals)
        return {"temp_f": temp, "wind_mph": wind, "precip_in": precip * GAME_WINDOW_HOURS}

    rows = []
    for g in games.iter_rows(named=True):
        base = {"game_id": g["game_id"]}
        if g["roof"] in INDOOR_ROOFS:
            rows.append(base | INDOOR_VALUES | {"is_indoor": 1, "weather_source": "indoor"})
            continue
        start = g["kickoff_utc"].replace(minute=0, second=0, microsecond=0)
        hours = [by_hour.get((g["stadium_id"], start + timedelta(hours=h))) for h in range(GAME_WINDOW_HOURS)]
        hours = [h for h in hours if h and h["temp_f"] is not None]
        if hours:
            vals = {
                "temp_f": sum(h["temp_f"] for h in hours) / len(hours),
                "wind_mph": sum(h["wind_mph"] or 0.0 for h in hours) / len(hours),
                "precip_in": sum(h["precip_in"] or 0.0 for h in hours) * GAME_WINDOW_HOURS / len(hours),
            }
            source = hours[0]["source"]
        else:
            vals, source = climatology(g["stadium_id"], g["kickoff_utc"].month), "climatology"
        rows.append(base | vals | {"is_indoor": 0, "weather_source": source})
    return pl.DataFrame(rows, schema={"game_id": pl.Utf8, "temp_f": pl.Float64, "wind_mph": pl.Float64,
                                      "precip_in": pl.Float64, "is_indoor": pl.Int64,
                                      "weather_source": pl.Utf8})
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_weather_features.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add seahawks_ml/features/weather.py tests/test_weather_features.py
git commit -m "feat: game-window weather features"
```

---

### Task 15: Feature table builder and leakage tests

**Files:**
- Create/replace: `seahawks_ml/features/columns.py`
- Create/replace: `seahawks_ml/features/build.py`
- Test: `tests/test_build.py`

`FEATURE_COLUMNS` is the single source of truth for model inputs. The leakage test rebuilds features from `raw.as_of(kickoff)` and asserts every feature for that game is unchanged.

- [ ] **Step 1: Write the failing test**

`tests/test_build.py`:

```python
import polars as pl
import pytest

from seahawks_ml.features.build import build_features
from seahawks_ml.features.columns import FEATURE_COLUMNS, ID_COLUMNS
from seahawks_ml.stadiums import load_stadiums
from tests.synthetic import make_raw


@pytest.fixture(scope="module")
def stadiums():
    return load_stadiums()


def test_build_features_schema(stadiums):
    raw = make_raw()
    frame = build_features(raw, stadiums)
    assert frame.columns == ID_COLUMNS + FEATURE_COLUMNS
    assert frame.height == raw.games.height
    assert frame["game_id"].n_unique() == frame.height
    for col in FEATURE_COLUMNS:
        assert frame[col].null_count() == 0, col
        assert frame[col].dtype.is_numeric(), col


def test_build_features_rows_for_unplayed_games(stadiums):
    raw = make_raw(unplayed_last_week=True)
    frame = build_features(raw, stadiums)
    future = frame.filter(pl.col("margin").is_null())
    assert future.height == 2
    assert future["elo_diff"].null_count() == 0


@pytest.mark.parametrize("row_index", [10, 30, 45])
def test_no_leakage_features_match_as_of_kickoff(stadiums, row_index):
    """Features for a game must be identical whether or not later data exists."""
    raw = make_raw()
    target = raw.games.row(row_index, named=True)
    full = build_features(raw, stadiums).filter(pl.col("game_id") == target["game_id"])
    cut = build_features(raw.as_of(target["kickoff_utc"]), stadiums).filter(
        pl.col("game_id") == target["game_id"])
    for col in FEATURE_COLUMNS:
        assert full[col][0] == pytest.approx(cut[col][0]), col
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_build.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seahawks_ml.features.build'`

- [ ] **Step 3: Write the implementation**

`seahawks_ml/features/columns.py`:

```python
"""The model's input columns. Everything is home-minus-away or a home/away pair."""

FEATURE_COLUMNS = [
    # home field
    "home_field", "hfa_trend", "no_crowd",
    # team strength
    "elo_diff", "off_rating_diff", "def_rating_diff",
    # quarterback
    "qb_epa_diff", "qb_cpoe_diff",
    "home_qb_round_1", "home_qb_day_2", "away_qb_round_1", "away_qb_day_2",
    # regime change
    "home_new_coach", "away_new_coach",
    # availability
    "home_off_out", "home_def_out", "away_off_out", "away_def_out", "availability_known",
    # situational
    "rest_diff", "home_post_bye", "away_post_bye", "home_short_week", "away_short_week",
    "travel_diff", "home_tz_shift", "away_tz_shift", "home_body_clock", "away_body_clock",
    "primetime", "div_game",
    # season timing
    "week_number", "is_final_regular_week", "is_playoff",
    # venue and weather
    "is_indoor", "temp_f", "wind_mph", "precip_in",
]

ID_COLUMNS = [
    "game_id", "season", "week", "game_type", "kickoff_utc", "home_team", "away_team",
    "neutral", "margin", "spread_line", "elo_home_pre", "elo_away_pre", "weather_source",
]
```

`seahawks_ml/features/build.py`:

```python
"""Join every feature module into one row per game."""

import polars as pl

from seahawks_ml.data import RawData
from seahawks_ml.features.availability import compute_availability
from seahawks_ml.features.base import team_games
from seahawks_ml.features.coaching import new_head_coach
from seahawks_ml.features.columns import FEATURE_COLUMNS, ID_COLUMNS
from seahawks_ml.features.elo import compute_elo
from seahawks_ml.features.qb import QBParams, compute_qb_features
from seahawks_ml.features.ratings import RatingParams, compute_team_ratings
from seahawks_ml.features.season_timing import compute_season_timing
from seahawks_ml.features.situational import compute_situational
from seahawks_ml.features.weather import compute_weather
from seahawks_ml.stadiums import Stadium


def _home_away(games: pl.DataFrame, per_team: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Turn a (game_id, team, cols...) frame into home_<col>/away_<col> per game."""
    out = games.select("game_id", "home_team", "away_team")
    for side in ("home", "away"):
        renamed = per_team.select(
            "game_id", pl.col("team").alias(f"{side}_team"), *[pl.col(c).alias(f"{side}_{c}") for c in cols]
        )
        out = out.join(renamed, on=["game_id", f"{side}_team"], how="left")
    return out.drop("home_team", "away_team")


def build_features(
    raw: RawData,
    stadiums: dict[str, Stadium],
    rating_params: RatingParams = RatingParams(),
    qb_params: QBParams = QBParams(),
) -> pl.DataFrame:
    games = raw.games
    tg = team_games(games)
    coach = new_head_coach(tg)
    ratings = compute_team_ratings(tg, raw.team_epa, coach, rating_params)

    frame = (
        games.select("game_id", "season", "week", "game_type", "kickoff_utc", "home_team",
                     "away_team", "neutral", "margin", "spread_line")
        .join(compute_elo(games), on="game_id")
        .join(_home_away(games, ratings, ["off_rating", "def_rating"]), on="game_id")
        .join(_home_away(games, coach, ["new_head_coach"]), on="game_id")
        .join(compute_qb_features(games, raw.qb_games, raw.players, qb_params), on="game_id")
        .join(compute_situational(games, stadiums), on="game_id")
        .join(compute_season_timing(games), on="game_id")
        .join(compute_availability(games, raw.snaps, raw.injuries, raw.players), on="game_id")
        .join(compute_weather(games, raw.weather), on="game_id")
    )
    avail = ["home_off_out", "home_def_out", "away_off_out", "away_def_out"]
    frame = frame.with_columns(
        (pl.col("elo_home_pre") - pl.col("elo_away_pre")).alias("elo_diff"),
        (pl.col("home_off_rating") - pl.col("away_off_rating")).alias("off_rating_diff"),
        (pl.col("home_def_rating") - pl.col("away_def_rating")).alias("def_rating_diff"),
        (pl.col("home_qb_epa") - pl.col("away_qb_epa")).alias("qb_epa_diff"),
        (pl.col("home_qb_cpoe") - pl.col("away_qb_cpoe")).alias("qb_cpoe_diff"),
        *[(pl.col(f"{s}_qb_bucket") == b).cast(pl.Int64).alias(f"{s}_qb_{b}")
          for s in ("home", "away") for b in ("round_1", "day_2")],
        pl.col("home_new_head_coach").alias("home_new_coach"),
        pl.col("away_new_head_coach").alias("away_new_coach"),
        pl.col("home_off_out").is_not_null().cast(pl.Int64).alias("availability_known"),
        *[pl.col(c).fill_null(0.0) for c in avail],
    )
    return frame.select(ID_COLUMNS + FEATURE_COLUMNS).sort("kickoff_utc", "game_id")
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_build.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add seahawks_ml/features/columns.py seahawks_ml/features/build.py tests/test_build.py
git commit -m "feat: feature table builder with leakage tests"
```

---

### Task 16: `features` command and first real build

**Files:**
- Create/replace: `seahawks_ml/cli.py`

Adds the CLI entry point. The first real run downloads 2002–present nflverse data (~1 minute) and the weather archive (slow the first time, see below).

- [ ] **Step 1: Write the file**

`seahawks_ml/cli.py`:

```python
"""Command-line entry point: python -m seahawks_ml.cli <command>."""

import argparse
from datetime import UTC, datetime

from seahawks_ml.config import FEATURES_PATH


def _now(args) -> datetime:
    return datetime.fromisoformat(args.now) if getattr(args, "now", None) else datetime.now(UTC)


def cmd_features(args) -> None:
    from seahawks_ml.data import load_raw
    from seahawks_ml.features.build import build_features
    from seahawks_ml.stadiums import load_stadiums

    stadiums = load_stadiums()
    raw = load_raw(stadiums, _now(args))
    frame = build_features(raw, stadiums)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(FEATURES_PATH)
    print(f"wrote {frame.height} games to {FEATURES_PATH}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="seahawks_ml")
    parser.add_argument("--now", help="override current time (ISO 8601, for testing)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("features", help="refresh data and rebuild the feature table").set_defaults(func=cmd_features)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests pass (1 network test deselected).

- [ ] **Step 3: First real build**

```bash
uv run python -m seahawks_ml.cli features
```
Expected: `weather archive: <stadium> (<n> requests)` lines, then `wrote 6771 games to .../data/features/games.parquet` (the count grows as nflverse adds games).

The weather archive is roughly 3,000 small requests. Open-Meteo's free tier allows about 5,000 calls an hour and 10,000 a day, so the first run takes about an hour, longer if it hits HTTP 429 and backs off. If it is interrupted or hits a daily limit, run the same command again later: completed stadiums are cached and skipped.

- [ ] **Step 4: Sanity-check the upcoming Seahawks game row**

```bash
uv run python -c "import polars as pl; f=pl.read_parquet('data/features/games.parquet'); print(f.filter(pl.col('margin').is_null() & ((pl.col('home_team')=='SEA')|(pl.col('away_team')=='SEA'))).head(1).to_dicts())"
```
Expected: one dict with no nulls in any feature column, `availability_known` = 1, and a plausible `weather_source` (`archive`/`climatology`/`indoor`).

- [ ] **Step 5: Commit code and the data cache**

```bash
git add seahawks_ml/cli.py data/cache data/features
git commit -m "feat: features CLI; cache nflverse aggregates and game-hour weather"
```

---
