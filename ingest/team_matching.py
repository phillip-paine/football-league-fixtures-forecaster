"""
Interactive team-name resolution for onboarding a new data source.

get_or_create_team() (ingest/load.py, used by the football-data.co.uk
path) creates a new team row the first time it sees an alias it doesn't
recognise -- correct for a source whose naming is already trusted via
seeded aliases, wrong for a first-time source like fixturedownload.com,
whose naming ("Man Utd", "Spurs", "Nott'm Forest") won't have any
team_aliases rows yet and would otherwise silently create ~90 duplicate
teams alongside the real ones on first run.

This module is for supervised, one-time use (e.g. the full-season
fixture preload) -- not for the twice-weekly automated loop, which
should keep using get_or_create_team() as-is once a source's aliases
are established.
"""
from __future__ import annotations

import difflib
import sqlite3


def _known_names(conn: sqlite3.Connection) -> dict[str, int]:
    """
    lowercased canonical_name/alias_name -> team_id, across every team
    and every source seen so far -- the widest net available for fuzzy
    matching a new source's naming against what's already in the DB.
    canonical_name wins on any collision (checked first, so a later
    alias with the same lowercased spelling can't overwrite it).
    """
    names: dict[str, int] = {}
    for team_id, name in conn.execute("SELECT team_id, canonical_name FROM teams"):
        names[name.lower()] = team_id
    for team_id, name in conn.execute("SELECT team_id, alias_name FROM team_aliases"):
        names.setdefault(name.lower(), team_id)
    return names


def resolve_team_interactive(conn: sqlite3.Connection, source: str, raw_name: str) -> int:
    """
    Resolve raw_name (as spelled by `source`) to a team_id.

    - Instant if this (source, raw_name) pair has already been resolved
      (a team_aliases row already exists) -- makes re-running the preload
      script idempotent and silent on a second run.
    - Otherwise: exact case-insensitive match against every known
      canonical_name/alias_name; failing that, up to 3 difflib
      close-match suggestions with a terminal prompt; failing that,
      offers to create a brand-new team (correct for a club with no
      history in the DB at all, e.g. a side newly promoted from the
      National League into League Two).
    - Always writes a team_aliases row for (source, raw_name) once
      resolved, so this prompt never repeats for the same raw name.
    """
    existing = conn.execute(
        "SELECT team_id FROM team_aliases WHERE source = ? AND alias_name = ?",
        (source, raw_name),
    ).fetchone()
    if existing:
        return existing[0]

    known = _known_names(conn)
    match = known.get(raw_name.lower())

    if match is None:
        close = difflib.get_close_matches(raw_name.lower(), known.keys(), n=3, cutoff=0.6)
        if close:
            print(f"\n'{raw_name}' -- possible matches:")
            for i, cand in enumerate(close, 1):
                print(f"  {i}) {cand}  (team_id={known[cand]})")
            print(f"  n) none of these -- create '{raw_name}' as a new team")
            choice = input("choice: ").strip().lower()
            if choice.isdigit() and 1 <= int(choice) <= len(close):
                match = close[int(choice) - 1]
        else:
            print(f"\n'{raw_name}' -- no close match found in the DB.")
            confirm = input(f"create '{raw_name}' as a new team? [y/N]: ").strip().lower()
            if confirm != "y":
                raise ValueError(
                    f"unresolved team name: {raw_name!r} (from source {source!r}) -- "
                    "declined to create; re-run once you know what this should map to"
                )

    if match is not None:
        team_id = known[match]
    else:
        cur = conn.execute("INSERT INTO teams (canonical_name) VALUES (?)", (raw_name,))
        team_id = cur.lastrowid
        print(f"created new team: {raw_name!r} (team_id={team_id})")

    conn.execute(
        "INSERT INTO team_aliases (team_id, source, alias_name) VALUES (?, ?, ?)",
        (team_id, source, raw_name),
    )
    return team_id