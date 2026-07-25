"""
Sanity checks that cross-reference tables the way the model design
assumes, but that the schema's foreign keys can't enforce by
themselves.

`matches.competition_id` already tells the extraction layer which
division a given fixture belongs to — that alone is enough to build a
MatchDataset. `team_season` is a separate, denormalized record of the
same fact (which competition a team played in for a season), used
elsewhere for a team's own promotion/relegation history. The two
should always agree; check_team_season_consistency() is how to catch
it if they don't, rather than the model silently pooling a team
toward the wrong division's prior.
"""

from __future__ import annotations

import sqlite3

_QUERY = """
SELECT
    m.match_id,
    m.season_id,
    m.competition_id  AS match_competition_id,
    m.home_team_id,
    ht.competition_id AS home_team_season_competition_id,
    m.away_team_id,
    at.competition_id AS away_team_season_competition_id
FROM matches m
LEFT JOIN team_season ht
    ON ht.team_id = m.home_team_id AND ht.season_id = m.season_id
LEFT JOIN team_season at
    ON at.team_id = m.away_team_id AND at.season_id = m.season_id
WHERE ht.competition_id IS NULL
   OR at.competition_id IS NULL
   OR ht.competition_id != m.competition_id
   OR at.competition_id != m.competition_id
"""


def check_team_season_consistency(conn: sqlite3.Connection) -> list[dict]:
    """Returns one dict per mismatch (or missing team_season row).
    Empty list = clean. Not raised as an exception because a missing
    team_season row can legitimately lag behind matches ingestion
    (e.g. fixtures loaded before team_season is backfilled for a new
    season) — the caller decides whether that's fatal for their run.
    """
    rows = conn.execute(_QUERY).fetchall()
    return [dict(r) for r in rows]