"""
Index maps: team_id / competition_id <-> contiguous 0-based indices
that PyMC uses for categorical/hierarchical dimensions.

Deliberately built from the dimension tables (`teams`, `competitions`),
never from `matches`. That's what makes the same index space usable
for both fitting (historical matches) and prediction (next weekend's
fixtures): a team's index doesn't shift depending on which slice of
matches happened to be queried, and a team with almost no matches yet
(newly promoted, or new to the DB) still gets a stable slot — the
hierarchical prior on their division is what carries them, not a
larger training-set index.

Foreign keys on `matches.home_team_id` / `.away_team_id` /
`.competition_id` guarantee every id seen in a match already exists in
these dimension tables, so building the index here and mapping
match-level ids onto it later can never hit an unknown id — a
KeyError there means the FK pragma was off, not that this layer needs
a defensive fallback.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class IndexMap:
    """Bidirectional mapping between database ids and 0-based array indices."""

    id_to_idx: dict[int, int]
    idx_to_id: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.idx_to_id)

    def to_idx(self, ids) -> list[int]:
        """Map an iterable of ids to indices. Raises KeyError with the
        offending id if one isn't in the map (see module docstring for
        why that should only happen if FK enforcement was off)."""
        try:
            return [self.id_to_idx[i] for i in ids]
        except KeyError as e:
            raise KeyError(
                f"id {e.args[0]} not found in this IndexMap — this means a "
                f"match referenced a team/competition id that isn't in the "
                f"dimension table. Check that PRAGMA foreign_keys = ON was "
                f"set on the connection that wrote it."
            ) from e


def build_team_index(conn: sqlite3.Connection) -> IndexMap:
    rows = conn.execute("SELECT team_id FROM teams ORDER BY team_id").fetchall()
    ids = tuple(r["team_id"] for r in rows)
    return IndexMap(id_to_idx={tid: i for i, tid in enumerate(ids)}, idx_to_id=ids)


def build_division_index(conn: sqlite3.Connection) -> IndexMap:
    """Division = competition here (each competition row is already one
    tier: Premier League/tier 1, Championship/tier 2, ...). Ordered by
    tier so index 0 is the top flight — purely cosmetic (helps when
    eyeballing fitted division-level priors) and not relied on
    elsewhere."""
    rows = conn.execute(
        "SELECT competition_id FROM competitions ORDER BY tier, competition_id"
    ).fetchall()
    ids = tuple(r["competition_id"] for r in rows)
    return IndexMap(id_to_idx={cid: i for i, cid in enumerate(ids)}, idx_to_id=ids)