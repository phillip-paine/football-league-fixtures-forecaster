"""
Sanity-check a predictions CSV before it's trusted enough to publish.

Usage:
    python scripts/sanity_check_predictions.py data/predictions/2026-08-10.csv

Exits non-zero (and prints what failed) if the file looks structurally
wrong in any way. This is deliberately a cheap, fast set of checks —
"is this a valid probability distribution over a sane set of matches",
not a judgement about whether the model is any good. Intended to run as
a gate in the weekly workflow: a non-zero exit here should stop the
workflow before any commit/push step runs, so a bad run never publishes.
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "match_id",
    "match_date",
    "division",
    "home_team",
    "away_team",
    "is_played",
    "p_home_win",
    "p_draw",
    "p_away_win",
]

PROB_COLUMNS = ["p_home_win", "p_draw", "p_away_win"]

# Loose bounds — this is a tripwire for "something is very wrong"
# (e.g. an empty df, or a date-window bug producing 100x too many rows),
# not a precise expectation. Adjust once real weekly runs establish what
# a normal row count actually looks like.
MIN_ROWS = 1
MAX_ROWS = 5000

PROB_SUM_TOLERANCE = 1e-3


def fail(msg: str, failures: list[str]) -> None:
    failures.append(msg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Path to a predictions CSV")
    parser.add_argument(
        "--expect-start-date",
        default=None,
        help="YYYY-MM-DD — if given, checks all match_date values fall on/after this",
    )
    parser.add_argument(
        "--expect-end-date",
        default=None,
        help="YYYY-MM-DD — if given, checks all match_date values fall before this",
    )
    args = parser.parse_args()

    failures: list[str] = []

    # 1. File exists and isn't empty
    if not args.csv_path.exists():
        print(f"FAIL: {args.csv_path} does not exist")
        return 1

    if args.csv_path.stat().st_size == 0:
        print(f"FAIL: {args.csv_path} is empty")
        return 1

    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"FAIL: could not read {args.csv_path} as CSV: {e}")
        return 1

    # Drop the unnamed pandas-index column if present, doesn't affect checks either way
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    # 2. Expected columns present
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        fail(f"missing expected column(s): {missing}", failures)
        # Can't run the remaining checks without these columns — report now.
        print("SANITY CHECK FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    # 3. Row count roughly sane
    n_rows = len(df)
    if n_rows < MIN_ROWS:
        fail(f"only {n_rows} rows (expected at least {MIN_ROWS})", failures)
    if n_rows > MAX_ROWS:
        fail(f"{n_rows} rows exceeds sanity ceiling of {MAX_ROWS} — possible date-window bug", failures)

    # 4. No NaN/null in probability columns
    for col in PROB_COLUMNS:
        n_null = df[col].isna().sum()
        if n_null > 0:
            fail(f"{n_null} null value(s) in '{col}'", failures)

    # 5. Probabilities within [0, 1]
    for col in PROB_COLUMNS:
        out_of_range = df[(df[col] < 0) | (df[col] > 1)]
        if len(out_of_range) > 0:
            fail(f"{len(out_of_range)} row(s) have '{col}' outside [0, 1]", failures)

    # 6. Probabilities sum to ~1 per match
    prob_sums = df[PROB_COLUMNS].sum(axis=1)
    bad_sums = df[(prob_sums - 1.0).abs() > PROB_SUM_TOLERANCE]
    if len(bad_sums) > 0:
        worst = (prob_sums - 1.0).abs().max()
        fail(
            f"{len(bad_sums)} row(s) have p_home_win + p_draw + p_away_win != 1 "
            f"(worst deviation: {worst:.6f}, match_ids: {bad_sums['match_id'].head(5).tolist()})",
            failures,
        )

    # 7. Match dates within expected window, if given
    if args.expect_start_date or args.expect_end_date:
        dates = pd.to_datetime(df["match_date"])
        if args.expect_start_date:
            start = pd.Timestamp(args.expect_start_date)
            too_early = df[dates < start]
            if len(too_early) > 0:
                fail(
                    f"{len(too_early)} row(s) have match_date before expected start {args.expect_start_date}",
                    failures,
                )
        if args.expect_end_date:
            end = pd.Timestamp(args.expect_end_date)
            too_late = df[dates >= end]
            if len(too_late) > 0:
                fail(
                    f"{len(too_late)} row(s) have match_date on/after expected end {args.expect_end_date}",
                    failures,
                )

    if failures:
        print(f"SANITY CHECK FAILED ({args.csv_path}):")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"Sanity check passed: {args.csv_path} ({n_rows} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
