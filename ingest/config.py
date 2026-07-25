"""
Static configuration for football-data.co.uk ingestion.

This file exists so that when a bookmaker gets added/dropped or a new
stat column shows up (which happens most seasons), the fix is a one-line
edit here -- never a schema change and never a change to load.py's logic.
"""

SOURCE_NAME = "football-data.co.uk"
BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Must already exist in competition_aliases (seeded in schema.sql).
LEAGUE_CODES = ["E0", "E1", "E2", "E3"]

# --- core match fields --------------------------------------------------
# Older seasons use short-form column names for the same fields
# (notes.txt: "FTHG and HG", "FTAG and AG", "FTR and Res"). First
# present column wins.
CORE_FIELD_ALIASES = {
    "home_goals":    ["FTHG", "HG"],
    "away_goals":    ["FTAG", "AG"],
    "home_goals_ht": ["HTHG"],
    "away_goals_ht": ["HTAG"],
}

# --- per-team match stats (-> match_team_stats EAV table) ---------------
# (home_column, away_column, stat_name). Add a row here for any new stat
# column football-data.co.uk introduces; no schema change needed.
STAT_COLUMNS = [
    ("HS",   "AS",   "shots"),
    ("HST",  "AST",  "shots_on_target"),
    ("HC",   "AC",   "corners"),
    ("HF",   "AF",   "fouls"),
    ("HY",   "AY",   "yellow_cards"),
    ("HR",   "AR",   "red_cards"),
    ("HO",   "AO",   "offsides"),
    ("HHW",  "AHW",  "hit_woodwork"),
    ("HBP",  "ABP",  "booking_points"),
    ("HFKC", "AFKC", "free_kicks_conceded"),
]

# --- match odds (-> match_odds EAV table) --------------------------------
# column prefix -> bookmaker label. Pre-closing 1X2 (H/D/A) odds only for
# v1, matching the model's covariate scope -- add prefixes for closing
# odds / over-under / Asian handicap later if those become covariates;
# still no schema change, just new rows.
ODDS_BOOKMAKER_PREFIXES = {
    "B365": "bet365",
    "BW":   "bet_and_win",
    "IW":   "interwetten",
    "PS":   "pinnacle",
    "P":    "pinnacle",   # pre-~2019 column name for the same bookmaker
    "WH":   "william_hill",
    "VC":   "vc_bet",
    "Max":  "market_max",
    "Avg":  "market_avg",
    "1XB":  "1xbet",
    "BMGM": "betmgm",
    "BV":   "betvictor",
}