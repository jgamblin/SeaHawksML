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


def test_every_stadium_has_an_elevation():
    stadiums = load_stadiums()
    assert stadiums["DEN00"].elevation_m == 1609
    assert stadiums["MEX00"].elevation_m == 2240
    assert 300 < stadiums["PHO00"].elevation_m < 370
    for s in stadiums.values():
        assert isinstance(s.elevation_m, float) and -10 <= s.elevation_m <= 2500, s.stadium_id


@pytest.mark.parametrize("value", ["", "high", "nan", None])
def test_load_stadiums_rejects_missing_or_bad_elevation(tmp_path, value):
    header = "stadium_id,name,lat,lon,tz,roof_default"
    row = "SEA00,Lumen Field,47.595,-122.332,America/Los_Angeles,outdoors"
    if value is not None:
        header, row = header + ",elevation_m", row + f",{value}"
    path = tmp_path / "s.csv"
    path.write_text(f"{header}\n{row}\n")
    with pytest.raises(ValueError, match="elevation_m"):
        load_stadiums(path)
