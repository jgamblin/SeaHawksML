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
