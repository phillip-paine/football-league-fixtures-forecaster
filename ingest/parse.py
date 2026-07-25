"""
Parse a single football-data.co.uk CSV into normalized row dicts.

Handles the two things that make these files annoying to work with over
a 30-year span: column names that drift across eras, and inconsistent
text encoding (team/referee names with accents).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CORE_FIELD_ALIASES, ODDS_BOOKMAKER_PREFIXES, STAT_COLUMNS


def _read_csv_any_encoding(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"could not decode {path} as utf-8 or cp1252")


def _first_present(row: dict, candidates: list[str]) -> Any:
    for col in candidates:
        if col in row and pd.notna(row[col]):
            return row[col]
    return None


def _int_or_none(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return int(value)


def _parse_date(raw: str) -> str:
    """football-data.co.uk mixes dd/mm/yy and dd/mm/yyyy across eras."""
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"unrecognized date format: {raw!r}")


def _extract_stats(row: dict) -> list[tuple[str, str, float]]:
    """Returns [(side, stat_name, value), ...] for side in ('home','away')."""
    out = []
    for home_col, away_col, stat_name in STAT_COLUMNS:
        if home_col in row and pd.notna(row[home_col]):
            out.append(("home", stat_name, float(row[home_col])))
        if away_col in row and pd.notna(row[away_col]):
            out.append(("away", stat_name, float(row[away_col])))
    return out


def _extract_odds(row: dict) -> list[tuple[str, str, float]]:
    """Returns [(bookmaker, outcome, odds), ...]."""
    out = []
    for prefix, bookmaker in ODDS_BOOKMAKER_PREFIXES.items():
        for outcome in ("H", "D", "A"):
            col = f"{prefix}{outcome}"
            if col in row and pd.notna(row[col]):
                out.append((bookmaker, outcome, float(row[col])))
    return out


def parse_matches(path: Path, league_code: str) -> list[dict]:
    df = _read_csv_any_encoding(path)
    df = df.dropna(how="all")  # some seasons' files carry blank trailing rows

    matches = []
    for _, raw in df.iterrows():
        row = raw.to_dict()
        if pd.isna(row.get("HomeTeam")) or pd.isna(row.get("AwayTeam")):
            continue  # blank/footer row

        home_goals = _first_present(row, CORE_FIELD_ALIASES["home_goals"])
        away_goals = _first_present(row, CORE_FIELD_ALIASES["away_goals"])

        matches.append({
            "league_code":   league_code,
            "match_date":    _parse_date(str(row["Date"]).strip()),
            "kickoff_time":  str(row["Time"]).strip() if pd.notna(row.get("Time")) else None,
            "home_team_raw": str(row["HomeTeam"]).strip(),
            "away_team_raw": str(row["AwayTeam"]).strip(),
            "home_goals":    _int_or_none(home_goals),
            "away_goals":    _int_or_none(away_goals),
            "home_goals_ht": _int_or_none(_first_present(row, CORE_FIELD_ALIASES["home_goals_ht"])),
            "away_goals_ht": _int_or_none(_first_present(row, CORE_FIELD_ALIASES["away_goals_ht"])),
            "referee":       str(row["Referee"]).strip() if pd.notna(row.get("Referee")) else None,
            "attendance":    _int_or_none(row.get("Attendance")),
            "status":        "played" if home_goals is not None and away_goals is not None else "scheduled",
            "stats":         _extract_stats(row),
            "odds":          _extract_odds(row),
        })
    return matches