"""
Load parsed football-data.co.uk rows into the SQLite schema.

Every function here is idempotent: re-running against the same data
updates rows in place rather than duplicating them. That's what lets the
weekly loop just re-download and re-run the whole thing twice a week
with no special-casing for "have I seen this before."

CHANGED for the full-season fixture preload (see
load_full_season_fixtures.py): upsert_match() now takes an optional
`source` and, when a source-scoped external_ids lookup misses, falls
back to a source-agnostic (home_team_id, away_team_id, season_id)
lookup before deciding to INSERT. This is what lets a fixture
pre-loaded from one source (e.g. fixturedownload.com) get updated in
place -- not duplicated -- once football-data.co.uk's own weekly CSV
later has the real result for the same match.
"""
from __future__ import annotations

import sqlite3

from .config import SOURCE_NAME


def get_or_create_team(conn: sqlite3.Connection, alias_name: str) -> int:
    row = conn.execute(
        "SELECT team_id FROM team_aliases WHERE source = ? AND alias_name = ?",
        (SOURCE_NAME, alias_name),
    ).fetchone()
    if row:
        return row[0]

    cur = conn.execute("INSERT INTO teams (canonical_name) VALUES (?)", (alias_name,))
    team_id = cur.lastrowid
    conn.execute(
        "INSERT INTO team_aliases (team_id, source, alias_name) VALUES (?, ?, ?)",
        (team_id, SOURCE_NAME, alias_name),
    )
    return team_id


def get_or_create_season(conn: sqlite3.Connection, start_year: int) -> int:
    row = conn.execute(
        "SELECT season_id FROM seasons WHERE start_year = ?", (start_year,)
    ).fetchone()
    if row:
        return row[0]
    label = f"{start_year}-{start_year + 1}"
    cur = conn.execute(
        "INSERT INTO seasons (start_year, label) VALUES (?, ?)", (start_year, label)
    )
    return cur.lastrowid


def get_competition_id(conn: sqlite3.Connection, league_code: str, source: str = SOURCE_NAME) -> int:
    row = conn.execute(
        "SELECT competition_id FROM competition_aliases WHERE source = ? AND alias_code = ?",
        (source, league_code),
    ).fetchone()
    if row is None:
        raise ValueError(
            f"no competition mapped for {source}/{league_code} -- "
            "add it to competition_aliases before ingesting this league"
        )
    return row[0]


def ensure_team_season(conn: sqlite3.Connection, team_id: int, season_id: int, competition_id: int) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO team_season (team_id, season_id, competition_id) VALUES (?, ?, ?)",
        (team_id, season_id, competition_id),
    )


def _match_external_id(league_code: str, match: dict) -> str:
    """
    football-data.co.uk gives no stable match ID, so we build one from
    fields that don't change between runs: division, date, and the raw
    (unresolved) team-name strings as the source itself spells them.

    Reused as-is for other sources (e.g. fixturedownload.com) -- the
    shape (league_code|date|home_raw|away_raw) works for any source
    that supplies those four fields on the match dict, and since
    external_ids is keyed on (source, entity_type, external_id), two
    sources building an identically-shaped string never collide with
    each other.
    """
    return f"{league_code}|{match['match_date']}|{match['home_team_raw']}|{match['away_team_raw']}"


def find_match_by_teams_season(
        conn: sqlite3.Connection,
        home_team_id: int,
        away_team_id: int,
        season_id: int,
) -> int | None:
    """
    Source-agnostic identity lookup: a given home/away pairing happens
    at most once per season in a normal English top-4-division single
    round-robin-twice fixture list (league fixtures only -- this isn't
    meant to and doesn't need to handle cup replays), so this is a safe
    fallback once the source-scoped external_ids lookup in upsert_match
    has already missed.

    This is what makes a fixture pre-loaded from one source (e.g. a
    full-season preload from fixturedownload.com, status='scheduled')
    get UPDATEd in place -- rather than duplicated -- once a different
    source (football-data.co.uk's weekly CSV) later has the real
    result for that same match.
    """
    row = conn.execute(
        "SELECT match_id FROM matches WHERE home_team_id = ? AND away_team_id = ? AND season_id = ?",
        (home_team_id, away_team_id, season_id),
    ).fetchone()
    return row[0] if row else None


def upsert_match(
        conn: sqlite3.Connection,
        match: dict,
        season_id: int,
        competition_id: int,
        home_team_id: int,
        away_team_id: int,
        source: str = SOURCE_NAME,
) -> int:
    ext_id = _match_external_id(match["league_code"], match)

    existing = conn.execute(
        "SELECT entity_id FROM external_ids WHERE source = ? AND entity_type = 'match' AND external_id = ?",
        (source, ext_id),
    ).fetchone()

    match_id_from_fallback = None
    if not existing:
        # Not seen yet under *this* source's own external_ids -- but it
        # might already exist under a different source (e.g. this is
        # football-data.co.uk's first real result for a match that
        # fixturedownload.com pre-loaded as a scheduled fixture weeks
        # earlier). Check before assuming it's genuinely new.
        match_id_from_fallback = find_match_by_teams_season(
            conn, home_team_id, away_team_id, season_id
        )

    fields = (
        season_id, competition_id, match["match_date"], match["kickoff_time"],
        home_team_id, away_team_id, match["home_goals"], match["away_goals"],
        match["home_goals_ht"], match["away_goals_ht"], match["status"],
        match["referee"], match["attendance"],
    )

    if existing or match_id_from_fallback:
        match_id = existing[0] if existing else match_id_from_fallback
        conn.execute(
            """UPDATE matches SET season_id=?, competition_id=?, match_date=?, kickoff_time=?,
               home_team_id=?, away_team_id=?, home_goals=?, away_goals=?,
               home_goals_ht=?, away_goals_ht=?, status=?, referee=?, attendance=?
               WHERE match_id = ?""",
            fields + (match_id,),
            )
        # Found via the fallback, not this source's own external_ids -- so
        # this source has never recorded an ID for this match before.
        # Record one now, so *this source's own* future runs hit the fast
        # source-scoped lookup above instead of the fallback every time.
        conn.execute(
            "INSERT OR IGNORE INTO external_ids (entity_type, entity_id, source, external_id) "
            "VALUES ('match', ?, ?, ?)",
            (match_id, source, ext_id),
        )
    else:
        cur = conn.execute(
            """INSERT INTO matches (season_id, competition_id, match_date, kickoff_time,
               home_team_id, away_team_id, home_goals, away_goals,
               home_goals_ht, away_goals_ht, status, referee, attendance)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            fields,
        )
        match_id = cur.lastrowid
        conn.execute(
            "INSERT INTO external_ids (entity_type, entity_id, source, external_id) VALUES ('match', ?, ?, ?)",
            (match_id, source, ext_id),
        )

    # Odds firm up and stats simply don't exist until full-time, so both
    # can legitimately change between two runs against the same match.
    # Replacing in full is simpler and just as correct as diffing --
    # for a source that always genuinely has odds/stats (football-data.co.uk).
    #
    # GUARD: a source that structurally never carries odds/stats (the
    # fixture preload) supplies empty lists here. Without this guard, an
    # accidental re-run of that preload mid-season would DELETE real
    # odds/stats that football-data.co.uk already populated for the same
    # match, then insert nothing back. Only touch these tables on a fresh
    # insert (nothing to lose) or when the incoming data actually has
    # something in it.
    is_new_row = not (existing or match_id_from_fallback)
    if is_new_row or match["odds"] or match["stats"]:
        conn.execute("DELETE FROM match_odds WHERE match_id = ?", (match_id,))
        conn.execute("DELETE FROM match_team_stats WHERE match_id = ?", (match_id,))

        conn.executemany(
            "INSERT INTO match_odds (match_id, bookmaker, outcome, odds) VALUES (?,?,?,?)",
            [(match_id, bookmaker, outcome, odds) for bookmaker, outcome, odds in match["odds"]],
        )
        conn.executemany(
            "INSERT INTO match_team_stats (match_id, team_id, stat_name, stat_value) VALUES (?,?,?,?)",
            [
                (match_id, home_team_id if side == "home" else away_team_id, stat_name, value)
                for side, stat_name, value in match["stats"]
            ],
        )

    return match_id