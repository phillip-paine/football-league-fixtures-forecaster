#!/usr/bin/env python3
"""
build_model_dataset.py — CLI wrapper around features.extraction, for
manual inspection and for the weekly loop to call.

Examples:
    # Training set for a Saturday fit, 180-day half-life
    python build_model_dataset.py train --db forecast.db \\
        --as-of 2026-08-15 --half-life 180

    # Upcoming fixtures to predict for that same gameweek
    python build_model_dataset.py fixtures --db forecast.db \\
        --start 2026-08-15 --horizon 3

    # Data-integrity check
    python build_model_dataset.py validate --db forecast.db
"""

from __future__ import annotations

import argparse
import sys

from features import (
    check_team_season_consistency,
    connect,
    load_fixtures,
    load_training_matches,
)


def cmd_train(args: argparse.Namespace) -> None:
    conn = connect(args.db)
    ds = load_training_matches(
        conn,
        as_of_date=args.as_of,
        half_life_days=args.half_life,
        competitions=args.competitions,
    )
    print(f"training set: {ds.n_matches} matches, {ds.n_teams} teams, "
          f"{ds.n_divisions} divisions, as_of={args.as_of}, "
          f"half_life={args.half_life}d")
    print(ds.to_frame().tail(10).to_string(index=False))


def cmd_fixtures(args: argparse.Namespace) -> None:
    conn = connect(args.db)
    ds = load_fixtures(
        conn,
        start_date=args.start,
        horizon_days=args.horizon,
        competitions=args.competitions,
    )
    print(f"fixture window: {ds.n_matches} matches from {args.start} "
          f"over {args.horizon} day(s)")
    print(ds.to_frame().to_string(index=False))


def cmd_validate(args: argparse.Namespace) -> None:
    conn = connect(args.db)
    mismatches = check_team_season_consistency(conn)
    if not mismatches:
        print("team_season consistency: OK (0 mismatches)")
        return
    print(f"team_season consistency: {len(mismatches)} mismatch(es)")
    for m in mismatches[:20]:
        print(f"  {m}")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="path to the sqlite db")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="load a training MatchDataset")
    p_train.add_argument("--as-of", required=True, help="YYYY-MM-DD cutoff")
    p_train.add_argument("--half-life", type=float, required=True,
                         help="decay half-life in days")
    p_train.add_argument("--competitions", type=int, nargs="*", default=None)
    p_train.set_defaults(func=cmd_train)

    p_fix = sub.add_parser("fixtures", help="load a fixture/prediction window")
    p_fix.add_argument("--start", required=True, help="YYYY-MM-DD window start")
    p_fix.add_argument("--horizon", type=int, default=7,
                       help="window length in days (default 7)")
    p_fix.add_argument("--competitions", type=int, nargs="*", default=None)
    p_fix.set_defaults(func=cmd_fixtures)

    p_val = sub.add_parser("validate", help="run data-integrity checks")
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
