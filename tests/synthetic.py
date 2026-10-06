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


def make_feature_frame(seasons=range(2009, 2016), games_per_season=120, seed=0) -> pl.DataFrame:
    """Model-ready feature frame where elo_diff and qb_epa_diff drive the margin."""
    import numpy as np

    from seahawks_ml.features.columns import FEATURE_COLUMNS

    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for i in range(games_per_season):
            feats = {c: 0.0 for c in FEATURE_COLUMNS}
            feats["elo_diff"] = rng.normal(0, 80)
            feats["qb_epa_diff"] = rng.normal(0, 0.1)
            feats["home_field"] = 1.0
            margin = round(feats["elo_diff"] / 25 + 30 * feats["qb_epa_diff"] + 1.5 + rng.normal(0, 13))
            rows.append({"game_id": f"{season}_{i:03d}", "season": season, "margin": margin,
                         "home_team": "SEA" if i % 8 == 0 else "SF", "away_team": "LA",
                         "neutral": False, "elo_home_pre": 1500 + feats["elo_diff"],
                         "elo_away_pre": 1500.0,
                         "spread_line": round(2 * (feats["elo_diff"] / 25 + 1.5)) / 2, **feats})
    return pl.DataFrame(rows)
