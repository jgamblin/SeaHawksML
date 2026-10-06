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
