#!/usr/bin/env python3
"""
No pytest in this environment, so this is a plain assert-based script:
    python tests/test_extraction.py

Builds a throwaway sqlite DB from the real schema.sql, seeds a small
scenario that mirrors the ingestion tests described in build_log.md
(a promoted team — here Luton, Championship -> Premier League — plus
ordinary top-flight matches), and exercises every public function in
`features` against it. Deleted at the end regardless of pass/fail.
"""

from __future__ import annotations

import datetime as dt
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import (  # noqa: E402
    build_division_index,
    build_team_index,
    check_team_season_consistency,
    connect,
    decay_weights,
    load_fixtures,
    load_training_matches,
)

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "schema.sql")

PASS = 0
FAIL = 0


def check(label: str, cond: bool) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}")


def build_test_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())

    conn.executemany(
        "INSERT INTO teams (team_id, canonical_name) VALUES (?, ?)",
        [(1, "Arsenal"), (2, "Chelsea"), (3, "Luton Town"), (4, "Burnley")],
    )

    conn.executemany(
        "INSERT INTO seasons (season_id, start_year, label) VALUES (?, ?, ?)",
        [(1, 2022, "2022-2023"), (2, 2023, "2023-2024")],
    )

    # Season 1: Luton & Burnley in the Championship, Arsenal & Chelsea in the PL.
    # Season 2: Luton promoted to the PL; Burnley stay down. This is the
    # promoted-team scenario the model hierarchy has to handle without a
    # hard reset on Luton's rating.
    conn.executemany(
        "INSERT INTO team_season (team_id, season_id, competition_id) VALUES (?, ?, ?)",
        [
            (1, 1, 1), (2, 1, 1),   # Arsenal, Chelsea -> PL, season 1
            (3, 1, 2), (4, 1, 2),   # Luton, Burnley -> Championship, season 1
            (1, 2, 1), (2, 2, 1),   # Arsenal, Chelsea -> PL, season 2
            (3, 2, 1),              # Luton -> PL, season 2 (promoted)
            (4, 2, 2),              # Burnley -> Championship, season 2 (stayed down)
        ],
    )

    matches = [
        # match_id, season, competition, date, home, away, hg, ag, status
        (1, 1, 2, "2022-10-01", 3, 4, 2, 1, "played"),   # Luton 2-1 Burnley (Championship)
        (2, 1, 2, "2023-02-01", 4, 3, 0, 0, "played"),   # Burnley 0-0 Luton (Championship)
        (3, 1, 1, "2022-10-01", 1, 2, 3, 1, "played"),   # Arsenal 3-1 Chelsea (PL)
        (4, 1, 1, "2023-03-01", 2, 1, 1, 1, "played"),   # Chelsea 1-1 Arsenal (PL)
        (5, 2, 1, "2023-08-15", 3, 1, 1, 2, "played"),   # Luton 1-2 Arsenal (PL, promoted)
        (6, 2, 1, "2023-12-01", 2, 3, 2, 0, "played"),   # Chelsea 2-0 Luton (PL)
        (7, 2, 1, "2024-01-15", 3, 2, None, None, "scheduled"),  # Luton v Chelsea, not yet played
        (8, 2, 1, "2024-01-16", 1, 3, 0, 1, "played"),   # Arsenal 0-1 Luton, played but held out of training
    ]
    conn.executemany(
        """INSERT INTO matches
           (match_id, season_id, competition_id, match_date, home_team_id,
            away_team_id, home_goals, away_goals, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        matches,
    )
    conn.commit()
    conn.close()


def run() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        build_test_db(db_path)

        # --- unit test: decay_weights in isolation -------------------
        print("decay_weights:")
        as_of = dt.date(2024, 1, 15)
        half_life = 153
        dates = [as_of - dt.timedelta(days=half_life),
                 as_of - dt.timedelta(days=2 * half_life)]
        w = decay_weights(dates, as_of, half_life)
        check("weight at exactly one half-life ~= 0.5", abs(w[0] - 0.5) < 1e-9)
        check("weight at two half-lives ~= 0.25", abs(w[1] - 0.25) < 1e-9)

        raised = False
        try:
            decay_weights([as_of], as_of, half_life)  # age == 0
        except ValueError:
            raised = True
        check("raises on non-positive age (walk-forward leak guard)", raised)

        # --- read-only connection, as production code will use it ----
        conn = connect(db_path, read_only=True)

        print("indices:")
        team_idx = build_team_index(conn)
        div_idx = build_division_index(conn)
        check("team index covers all 4 teams (global, not match-filtered)",
              len(team_idx) == 4)
        check("division index covers all 4 seeded competitions",
              len(div_idx) == 4)
        check("division index orders PL (tier 1) first",
              div_idx.idx_to_id[0] == 1)
        check("division index orders Championship (tier 2) second",
              div_idx.idx_to_id[1] == 2)

        print("load_training_matches (as_of=2024-01-15, half_life=180):")
        train = load_training_matches(
            conn, as_of_date="2024-01-15", half_life_days=180,
            team_index=team_idx, division_index=div_idx,
        )
        check("training set excludes the scheduled match (id 7)",
              7 not in train.match_id.tolist())
        check("training set excludes the match on/after cutoff (id 8, 2024-01-16)",
              8 not in train.match_id.tolist())
        check("training set includes all 6 prior played matches",
              train.n_matches == 6)
        check("training set is fully played",
              bool(train.is_played.all()))
        check("decay weights are present and in (0, 1]",
              train.decay_weight is not None
              and bool((train.decay_weight > 0).all())
              and bool((train.decay_weight <= 1).all()))
        check("more recent matches get higher weight than older ones",
              train.decay_weight[train.match_id == 6][0]
              > train.decay_weight[train.match_id == 1][0])
        check("n_teams reflects the full team dimension table, not just this slice",
              train.n_teams == 4)

        luton_idx_train = team_idx.id_to_idx[3]
        check("Luton has a stable index in the training set",
              luton_idx_train in train.home_idx.tolist() + train.away_idx.tolist())

        print("load_fixtures (window 2024-01-15 .. +7d):")
        fixtures = load_fixtures(
            conn, start_date="2024-01-15", horizon_days=7,
            team_index=team_idx, division_index=div_idx,
        )
        check("fixture window includes the scheduled match (id 7)",
              7 in fixtures.match_id.tolist())
        check("fixture window includes the held-out played match (id 8)",
              8 in fixtures.match_id.tolist())
        check("fixture window has no decay weights (nothing to fit)",
              fixtures.decay_weight is None)

        m7 = fixtures.to_frame().set_index("match_id").loc[7]
        m8 = fixtures.to_frame().set_index("match_id").loc[8]
        check("scheduled match (id 7) has no goals yet",
              bool(pd_isna(m7["home_goals"])) and not m7["is_played"])
        check("held-out played match (id 8) carries its real goals",
              m8["is_played"] and m8["away_goals"] == 1)

        check("Luton's index is identical in training and fixtures "
              "(same team, despite the division change) — no promoted-team reset",
              team_idx.id_to_idx[3] == luton_idx_train)

        print("check_team_season_consistency:")
        mismatches = check_team_season_consistency(conn)
        check("clean DB reports zero mismatches", mismatches == [])

        conn.close()

        # Now deliberately corrupt one row and confirm the checker catches it.
        conn2 = sqlite3.connect(db_path)
        conn2.execute("PRAGMA foreign_keys = ON")
        conn2.execute(
            "UPDATE matches SET competition_id = 2 WHERE match_id = 3"
        )  # Arsenal-Chelsea PL match mislabeled as Championship
        conn2.commit()
        conn2.close()

        conn3 = connect(db_path, read_only=True)
        mismatches2 = check_team_season_consistency(conn3)
        check("corrupted row is detected",
              any(m["match_id"] == 3 for m in mismatches2))
        conn3.close()

    return FAIL


def pd_isna(x) -> bool:
    import math
    return x is None or (isinstance(x, float) and math.isnan(x))


if __name__ == "__main__":
    failures = run()
    print(f"\n{PASS} passed, {failures} failed")
    sys.exit(1 if failures else 0)