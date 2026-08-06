"""
Promoted/relegated status per (team_id, season_id) — derived from
`team_season`.

This is the data-sourcing half of the promoted/relegated covariate
(model/dixon_coles.py owns the other half: consuming these flags as
additive attack/defense terms). Per model_quick_reference.md's own
rule, sourcing new covariate data is features/'s job, not model/'s —
model/ has zero DB dependency by design.

ASSUMPTIONS — verify against your actual schema.sql before trusting
this against the real DB:
  - `team_season` has (team_id, season_id, competition_id) — matching
    build_log.md's description of the table.
  - `competitions.tier` exists and is comparable across seasons (lower
    tier = higher division), matching indices.build_division_index's
    own use of `ORDER BY tier`.
  - `season_id` increases monotonically with time (assigned in
    ingestion order). If your `seasons` table orders differently (e.g.
    a non-sequential PK), sort `rows` below by whatever column actually
    reflects chronological order before the comparison loop.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionStatus:
    is_promoted: bool
    is_relegated: bool


NOT_MOVED = PromotionStatus(is_promoted=False, is_relegated=False)


def build_promotion_status(
        conn: sqlite3.Connection,
) -> dict[tuple[int, int], PromotionStatus]:
    """For every (team_id, season_id) in `team_season`, determine whether
    the team was promoted into, or relegated into, the division it
    played in that season — by comparing tier against that same team's
    most recent *previous* season in `team_season` (not necessarily
    season_id - 1; a team can have gaps in this DB's coverage).

    A team's first-ever season on record gets is_promoted=is_relegated=
    False — there's no prior tier to compare against, and that's the
    right behaviour: a brand-new-to-the-DB team should be handled by
    ordinary thin-data shrinkage on attack/defense, not mislabeled as
    promoted.

    Returns a plain dict, not a class, so it can be passed around and
    reused across load_training_matches/load_fixtures calls the same
    way IndexMap instances are (see feature_extraction_handoff.md —
    same "build once, reuse across a run" pattern).
    """
    rows = conn.execute(
        """
        SELECT ts.team_id, ts.season_id, c.tier
        FROM team_season ts
        JOIN competitions c ON c.competition_id = ts.competition_id
        ORDER BY ts.team_id, ts.season_id
        """
    ).fetchall()

    status: dict[tuple[int, int], PromotionStatus] = {}
    prev_tier_by_team: dict[int, int] = {}

    for r in rows:
        team_id, season_id, tier = r["team_id"], r["season_id"], r["tier"]
        prev_tier = prev_tier_by_team.get(team_id)
        if prev_tier is None:
            status[(team_id, season_id)] = NOT_MOVED
        else:
            status[(team_id, season_id)] = PromotionStatus(
                is_promoted=tier < prev_tier,
                is_relegated=tier > prev_tier,
            )
        prev_tier_by_team[team_id] = tier

    return status


def lookup(
        status: dict[tuple[int, int], PromotionStatus],
        team_id: int,
        season_id: int,
) -> PromotionStatus:
    """Defensive lookup — a missing (team_id, season_id) key means
    team_season hasn't caught up with matches yet (feature_extraction_
    handoff.md notes this can legitimately lag ingestion). Treat as
    "not moved" rather than raising, consistent with how
    check_team_season_consistency() treats the same situation as a
    reportable warning, not a hard failure."""
    return status.get((team_id, season_id), NOT_MOVED)
