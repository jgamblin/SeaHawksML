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
