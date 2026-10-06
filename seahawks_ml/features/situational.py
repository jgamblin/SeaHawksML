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
