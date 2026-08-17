"""
One-time full-season fixture pre-load, from fixturedownload.com.

Why this exists: football-data.co.uk (the twice-weekly ingestion source)
never has more than a rolling 1-2 weeks of fixtures at a time, and
football-data.org's `matches` endpoint is, as of writing, still on last
season's fixture list for these competitions. fixturedownload.com
already has all four divisions' full 2026/27 fixture lists as clean
per-division CSV exports -- a static, one-time gap-filler, not a second
automated ingestion source.

Usage (run once, before the weekly loop's next run):
    poetry run python load_full_season_fixtures.py --db data/forecast.db \
        --fixtures-dir data/raw/fixture-download

Point --fixtures-dir at a folder containing one manually-downloaded CSV
per division -- either fixturedownload.com's own default filenames
(e.g. "epl-20262027.csv", "efl-championship-20262027.csv"; see
ingest/fixturedownload_config.py's DOWNLOAD_FILENAMES) or a manually
renamed "<slug>.csv" (e.g. "epl-2026.csv") -- this is the recommended
path, since fixturedownload.com appears to be behind bot protection that
blocks a plain `requests.get` (see fetch_csv_rows below), and since this
is genuinely a once-a-year operation, downloading four files by hand in
a browser is simpler than fighting that. data/raw/ is already
gitignored, so these files never get committed.

If --fixtures-dir is omitted, falls back to fetching over HTTP directly
(fetch_csv_rows) -- kept in case that route starts working reliably in
the future, but not the expected path right now.

Rows land as status='scheduled', goals NULL. The existing weekly
football-data.co.uk ingestion then updates these same rows in place to
status='played' with real scores as matchdays happen -- via
ingest.load.upsert_match's source-agnostic (home_team_id, away_team_id,
season_id) fallback lookup (see ingest/load.py), not by matching
external IDs across sources; the two sources' IDs are never expected to
line up.

NOT the twice-weekly automated path: this is supervised and
interactive (team-name resolution prompts on the terminal via
ingest.team_matching), meant to be run once per season. Safe to
re-run (existing rows get skipped past ensure_team_season/team_aliases
lookups, or updated in place by upsert_match) but there's no reason to,
outside of correcting a mistake.
"""
from __future__ import annotations

import argparse
import csv
import io
import sqlite3
from datetime import datetime
from pathlib import Path

import requests

from ingest.fixturedownload_config import SOURCE_NAME, BASE_URL, DIVISIONS, DOWNLOAD_FILENAMES
from ingest.load import ensure_team_season, get_or_create_season, upsert_match
from ingest.team_matching import resolve_team_interactive


def ensure_competition_aliases(conn: sqlite3.Connection) -> None:
    """One-time seed: map each fixturedownload.com slug to our
    competition_id, so get_competition_id-style lookups would work for
    this source too (not currently used here since we already have
    competition_id from DIVISIONS, but keeps competition_aliases
    consistent for any future code that expects every source to be
    represented there)."""
    conn.executemany(
        "INSERT OR IGNORE INTO competition_aliases (competition_id, source, alias_code) VALUES (?, ?, ?)",
        [(competition_id, SOURCE_NAME, slug) for slug, competition_id in DIVISIONS.items()],
    )


REQUIRED_COLUMNS = {"Date", "Home Team", "Away Team"}


def _validate_columns(rows: list[dict], slug: str, where: str) -> None:
    if not rows:
        raise ValueError(f"no rows found for {slug!r} ({where}) -- empty CSV?")
    missing = REQUIRED_COLUMNS - set(rows[0].keys())
    if missing:
        raise ValueError(
            f"{slug!r} CSV ({where}) is missing expected column(s) {missing} -- "
            f"got columns: {list(rows[0].keys())}. Wrong file, or fixturedownload.com "
            "changed its export format?"
        )


def read_local_csv_rows(slug: str, fixtures_dir: str) -> list[dict]:
    # Prefer fixturedownload.com's own default download filename for this
    # slug (what you actually get from a real manual download); fall back
    # to "<slug>.csv" for anyone who renamed files by hand instead.
    filename = DOWNLOAD_FILENAMES.get(slug, f"{slug}.csv")
    path = Path(fixtures_dir) / filename
    if not path.exists():
        fallback = Path(fixtures_dir) / f"{slug}.csv"
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(
                f"expected a CSV at {path} (or {fallback}) for division {slug!r} -- "
                "check the file is in --fixtures-dir under one of those two names"
            )
    text = path.read_text(encoding="utf-8-sig")  # fixturedownload's CSVs carry a BOM
    rows = list(csv.DictReader(io.StringIO(text)))
    _validate_columns(rows, slug, where=str(path))
    return rows


def fetch_csv_rows(slug: str) -> list[dict]:
    url = f"{BASE_URL}/{slug}"
    headers = {
        # fixturedownload.com is Cloudflare-fronted and appears to respond
        # to requests' default User-Agent ("python-requests/x.x") with an
        # HTML challenge/interstitial page instead of the real CSV, even
        # though the same URL works fine from a browser. A realistic
        # browser UA is enough to get the real response in testing.
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/csv,text/plain,*/*",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    text = resp.content.decode("utf-8-sig")  # fixturedownload's CSVs carry a BOM

    # Defensive check: raise_for_status() only catches 4xx/5xx -- a bot
    # challenge page comes back as a normal 200 with an HTML body, so it
    # would otherwise sail through and get fed to csv.DictReader, either
    # crashing confusingly downstream (missing "Date"/"Home Team" columns)
    # or, worse, silently producing zero/garbage rows.
    looks_like_html = "text/html" in content_type or text.lstrip()[:15].lower().startswith(("<!doctype", "<html"))
    if looks_like_html:
        raise ValueError(
            f"expected CSV from {url} but got what looks like an HTML page "
            f"(Content-Type: {content_type!r}). This usually means bot "
            "protection served a challenge page instead of the real file -- "
            "try downloading the file manually in a browser instead and use "
            "--fixtures-dir."
        )

    rows = list(csv.DictReader(io.StringIO(text)))
    _validate_columns(rows, slug, where=url)
    return rows


def get_csv_rows(slug: str, fixtures_dir: str | None) -> list[dict]:
    if fixtures_dir:
        return read_local_csv_rows(slug, fixtures_dir)
    return fetch_csv_rows(slug)


def _parse_date(raw: str) -> tuple[str, str]:
    """'21/08/2026 20:00' -> ('2026-08-21', '20:00')."""
    dt = datetime.strptime(raw.strip(), "%d/%m/%Y %H:%M")
    return dt.date().isoformat(), dt.strftime("%H:%M")


def load_division(
        conn: sqlite3.Connection,
        slug: str,
        competition_id: int,
        season_start_year: int,
        fixtures_dir: str | None,
) -> int:
    rows = get_csv_rows(slug, fixtures_dir)
    season_id = get_or_create_season(conn, season_start_year)

    n = 0
    for row in rows:
        match_date, kickoff_time = _parse_date(row["Date"])
        home_raw = row["Home Team"].strip()
        away_raw = row["Away Team"].strip()

        home_id = resolve_team_interactive(conn, SOURCE_NAME, home_raw)
        away_id = resolve_team_interactive(conn, SOURCE_NAME, away_raw)
        ensure_team_season(conn, home_id, season_id, competition_id)
        ensure_team_season(conn, away_id, season_id, competition_id)

        match = {
            "league_code":   slug,
            "match_date":    match_date,
            "kickoff_time":  kickoff_time,
            "home_team_raw": home_raw,
            "away_team_raw": away_raw,
            "home_goals":    None,
            "away_goals":    None,
            "home_goals_ht": None,
            "away_goals_ht": None,
            "status":        "scheduled",
            "referee":       None,
            "attendance":    None,
            "stats":         [],
            "odds":          [],
        }
        upsert_match(conn, match, season_id, competition_id, home_id, away_id, source=SOURCE_NAME)
        n += 1

    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", required=True, help="path to the SQLite database")
    parser.add_argument("--season-start-year", type=int, default=2026)
    parser.add_argument(
        "--divisions", nargs="+", default=list(DIVISIONS.keys()),
        help="fixturedownload.com slugs to load, default: all four divisions",
    )
    parser.add_argument(
        "--fixtures-dir", default=None,
        help="folder of manually-downloaded CSVs, one per division, named "
             "<slug>.csv (e.g. epl-2026.csv). Recommended -- see module docstring. "
             "If omitted, fetches over HTTP instead.",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON")
    ensure_competition_aliases(conn)
    conn.commit()

    total = 0
    for slug in args.divisions:
        if slug not in DIVISIONS:
            raise ValueError(f"unknown division slug: {slug!r} -- must be one of {list(DIVISIONS)}")
        competition_id = DIVISIONS[slug]
        print(f"\n=== {slug} ===")
        n = load_division(conn, slug, competition_id, args.season_start_year, args.fixtures_dir)
        conn.commit()
        print(f"  {n} fixtures loaded")
        total += n

    conn.close()
    print(f"\nDone. {total} fixtures loaded across {len(args.divisions)} division(s).")


if __name__ == "__main__":
    main()