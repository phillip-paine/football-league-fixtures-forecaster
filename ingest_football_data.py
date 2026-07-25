"""
Ingest football-data.co.uk results/odds/stats into the forecast engine DB.

Usage:
    # one-time historical build
    poetry run python ingest_football_data.py --db forecast.db --from-year 1993 --to-year 2025

    # weekly loop (current season only, force re-download)
    poetry run python ingest_football_data.py --db forecast.db --from-year 2025 --force

Safe to re-run: this is exactly what the weekly loop will call twice a
week, unattended, against the current season's files.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ingest.config import LEAGUE_CODES, SOURCE_NAME
from ingest.download import fetch_csv
from ingest.load import (
    ensure_team_season,
    get_competition_id,
    get_or_create_season,
    get_or_create_team,
    upsert_match,
)
from ingest.parse import parse_matches


def ingest_one(conn: sqlite3.Connection, start_year: int, league_code: str,
               cache_dir: Path, force: bool) -> int:
    csv_path = fetch_csv(start_year, league_code, cache_dir, force=force)
    matches = parse_matches(csv_path, league_code)

    season_id = get_or_create_season(conn, start_year)
    competition_id = get_competition_id(conn, league_code)

    for match in matches:
        home_id = get_or_create_team(conn, match["home_team_raw"])
        away_id = get_or_create_team(conn, match["away_team_raw"])
        ensure_team_season(conn, home_id, season_id, competition_id)
        ensure_team_season(conn, away_id, season_id, competition_id)
        upsert_match(conn, match, season_id, competition_id, home_id, away_id)

    return len(matches)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", required=True, help="path to the SQLite database")
    parser.add_argument("--from-year", type=int, required=True, help="first season start year, e.g. 1993")
    parser.add_argument("--to-year", type=int, default=None, help="last season start year (default: --from-year)")
    parser.add_argument("--leagues", nargs="+", default=LEAGUE_CODES, help="league codes to ingest, e.g. E0 E1")
    parser.add_argument("--cache-dir", default="data/raw/football-data-co-uk", help="local CSV cache directory")
    parser.add_argument("--force", action="store_true", help="re-download even if a cached copy exists")
    parser.add_argument("--sleep", type=float, default=1.0, help="seconds between HTTP requests")
    args = parser.parse_args()

    to_year = args.to_year if args.to_year is not None else args.from_year
    cache_dir = Path(args.cache_dir)

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON")

    run_id = conn.execute(
        "INSERT INTO ingestion_runs (source, started_at, status) VALUES (?, ?, 'running')",
        (SOURCE_NAME, datetime.now(timezone.utc).isoformat()),
    ).lastrowid
    conn.commit()

    total_rows = 0
    try:
        for start_year in range(args.from_year, to_year + 1):
            for league_code in args.leagues:
                print(f"Ingesting {league_code} {start_year}-{start_year + 1}...")
                try:
                    n = ingest_one(conn, start_year, league_code, cache_dir, args.force)
                    total_rows += n
                    conn.commit()
                    print(f"  {n} matches")
                except Exception as exc:  # one bad season/league shouldn't kill the whole run
                    conn.rollback()
                    print(f"  FAILED: {exc}", file=sys.stderr)
                time.sleep(args.sleep)

        conn.execute(
            "UPDATE ingestion_runs SET finished_at=?, rows_written=?, status='success' WHERE run_id=?",
            (datetime.now(timezone.utc).isoformat(), total_rows, run_id),
        )
        conn.commit()
    except Exception:
        conn.execute(
            "UPDATE ingestion_runs SET finished_at=?, rows_written=?, status='failed' WHERE run_id=?",
            (datetime.now(timezone.utc).isoformat(), total_rows, run_id),
        )
        conn.commit()
        raise
    finally:
        conn.close()

    print(f"\nDone. {total_rows} matches processed.")


if __name__ == "__main__":
    main()