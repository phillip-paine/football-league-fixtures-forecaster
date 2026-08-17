"""
Sanity checks for the full-season fixture preload (load_full_season_fixtures.py).

Run this once, right after the preload script finishes, before trusting
the DB further. Not a CI gate like scripts/sanity_check_predictions.py --
this is a one-time diagnostic for a one-time operation, meant to be read
by a human, not to fail-close a pipeline.

Usage:
    poetry run python scripts/check_fixture_preload.py --db data/forecast.db --season-start-year 2026
"""
from __future__ import annotations

import argparse
import sqlite3

# n_teams -> expected total fixtures in a standard home-and-away
# single round-robin-twice league season (n * (n-1)).
EXPECTED_TEAMS = {
    "Premier League": 20,
    "Championship":   24,
    "League One":     24,
    "League Two":     24,
}


def check_division(conn: sqlite3.Connection, competition_name: str, competition_id: int, season_id: int) -> list[str]:
    problems = []

    n_teams_expected = EXPECTED_TEAMS[competition_name]
    n_fixtures_expected = n_teams_expected * (n_teams_expected - 1)

    rows = conn.execute(
        """SELECT match_id, home_team_id, away_team_id, match_date, status, home_goals, away_goals
           FROM matches WHERE competition_id = ? AND season_id = ?""",
        (competition_id, season_id),
    ).fetchall()

    # --- 1. total fixture count -------------------------------------------------
    if len(rows) != n_fixtures_expected:
        problems.append(
            f"[{competition_name}] expected {n_fixtures_expected} fixtures "
            f"({n_teams_expected} teams), found {len(rows)}"
        )

    # --- 2. duplicate (home, away) pairs ----------------------------------------
    pair_counts: dict[tuple[int, int], int] = {}
    for _, home_id, away_id, *_ in rows:
        pair_counts[(home_id, away_id)] = pair_counts.get((home_id, away_id), 0) + 1
    dupes = {pair: n for pair, n in pair_counts.items() if n > 1}
    if dupes:
        problems.append(f"[{competition_name}] duplicate (home_team_id, away_team_id) pairs: {dupes}")

    # --- 3. every team plays n_teams-1 home and n_teams-1 away -------------------
    home_counts: dict[int, int] = {}
    away_counts: dict[int, int] = {}
    for _, home_id, away_id, *_ in rows:
        home_counts[home_id] = home_counts.get(home_id, 0) + 1
        away_counts[away_id] = away_counts.get(away_id, 0) + 1

    all_team_ids = set(home_counts) | set(away_counts)
    if len(all_team_ids) != n_teams_expected:
        problems.append(
            f"[{competition_name}] expected {n_teams_expected} distinct teams, "
            f"found {len(all_team_ids)}"
        )
    for team_id in all_team_ids:
        h, a = home_counts.get(team_id, 0), away_counts.get(team_id, 0)
        if h != n_teams_expected - 1 or a != n_teams_expected - 1:
            name = conn.execute(
                "SELECT canonical_name FROM teams WHERE team_id = ?", (team_id,)
            ).fetchone()
            problems.append(
                f"[{competition_name}] team_id={team_id} ({name[0] if name else '?'}) "
                f"has {h} home / {a} away fixtures, expected {n_teams_expected - 1} each"
            )

    # --- 4. status / goals sanity -------------------------------------------------
    bad_status = [r for r in rows if r[4] != "scheduled" or r[5] is not None or r[6] is not None]
    if bad_status:
        problems.append(
            f"[{competition_name}] {len(bad_status)} row(s) not status='scheduled' with null goals "
            f"(unexpected for a preload -- e.g. match_ids: {[r[0] for r in bad_status[:5]]})"
        )

    # --- 5. date range sanity -------------------------------------------------
    dates = sorted(r[3] for r in rows)
    if dates and not (dates[0].startswith("2026-08") or dates[0].startswith("2026-07")):
        problems.append(f"[{competition_name}] earliest match_date looks off: {dates[0]}")
    if dates and dates[-1] > "2027-07-01":
        problems.append(f"[{competition_name}] latest match_date looks off: {dates[-1]}")

    return problems


def list_newly_created_teams(conn: sqlite3.Connection, source: str) -> list[tuple[int, str]]:
    """Teams whose *only* alias is this source -- i.e. teams that didn't
    already exist under any other source before this preload ran. Worth
    a manual glance: expected for genuine promotions, unexpected
    otherwise."""
    rows = conn.execute(
        """
        SELECT t.team_id, t.canonical_name
        FROM teams t
        WHERE t.team_id IN (
            SELECT team_id FROM team_aliases WHERE source = ?
        )
        AND t.team_id NOT IN (
            SELECT team_id FROM team_aliases WHERE source != ?
        )
        """,
        (source, source),
    ).fetchall()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", required=True)
    parser.add_argument("--season-start-year", type=int, default=2026)
    parser.add_argument("--source", default="fixturedownload.com")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON")

    season_row = conn.execute(
        "SELECT season_id FROM seasons WHERE start_year = ?", (args.season_start_year,)
    ).fetchone()
    if season_row is None:
        raise SystemExit(f"no season row for start_year={args.season_start_year} -- did the preload run?")
    season_id = season_row[0]

    all_problems: list[str] = []
    for name, competition_id in conn.execute("SELECT name, competition_id FROM competitions").fetchall():
        if name not in EXPECTED_TEAMS:
            continue
        all_problems.extend(check_division(conn, name, competition_id, season_id))

    print("=== Fixture preload sanity check ===\n")
    if all_problems:
        print(f"{len(all_problems)} problem(s) found:\n")
        for p in all_problems:
            print(f"  - {p}")
    else:
        print("All structural checks passed (fixture counts, no duplicate pairs, "
              "home/away balance, status/goals, date ranges).")

    new_teams = list_newly_created_teams(conn, args.source)
    print(f"\n{len(new_teams)} team(s) created for the first time by this preload:")
    for team_id, name in new_teams:
        print(f"  - team_id={team_id}: {name!r}  <- review spelling / confirm this is a genuine new club")

    conn.close()


if __name__ == "__main__":
    main()