"""
CLI: walk-forward backtest sweep + cross-experiment reporting.

    # run a walk-forward backtest, monthly refit cadence
    python run_backtest.py run --db data/forecast.db \\
        --experiment baseline --method dixon_coles \\
        --start-date 2023-08-01 --end-date 2025-06-01 \\
        --half-life 180 --min-history-days 180 \\
        --draws 500 --tune 500 --chains 2 \\
        --out-dir data/backtest

    # aggregate + compare one or more experiments (repeat --experiment)
    python run_backtest.py report \\
        --backtest-dir data/backtest \\
        --experiment baseline --experiment with_weather \\
        --out data/backtest/comparison_report.csv

This depends on the real `features` and `model` packages and a populated
`forecast.db` — not run as part of this handoff's test suite
(tests/test_backtest.py exercises backtest/ against a synthetic in-memory
history instead, via dependency-injected loader functions).
"""

import argparse
from datetime import date

import pandas as pd

from features import connect, build_team_index, build_division_index, load_training_matches, load_fixtures
from features.promotion import build_promotion_status
from model import fit as model_fit, posterior_rates, match_outcome_probs
from backtest import BacktestConfig, run_walk_forward_backtest, load_experiment, monthly_summary, season_summary


def cmd_run(args):
    conn = connect(args.db)

    promotion_status = build_promotion_status(conn)
    loader_kwargs = {
        "promotion_status": promotion_status,
        "include_promotion_covariate": args.include_promotion_covariate,
    }

    config = BacktestConfig(
        experiment=args.experiment,
        method=args.method,
        half_life_days=args.half_life,
        min_history_days=args.min_history_days,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        random_seed=args.seed,
        loader_kwargs=loader_kwargs,
    )
    results = run_walk_forward_backtest(
        conn,
        start_date=date.fromisoformat(args.start_date),
        end_date=date.fromisoformat(args.end_date),
        config=config,
        build_team_index_fn=build_team_index,
        build_division_index_fn=build_division_index,
        load_training_matches_fn=load_training_matches,
        load_fixtures_fn=load_fixtures,
        fit_fn=model_fit,
        posterior_rates_fn=posterior_rates,
        match_outcome_probs_fn=match_outcome_probs,
        out_dir=args.out_dir,
    )

    print(f"\nScored {len(results)} matches across {results['as_of_date'].nunique()} checkpoints.")
    print(monthly_summary(results).to_string(index=False))

    overall_rps = results["rps"].mean()
    overall_naive_rps = results["naive_rps"].mean()
    print(
        f"\nOverall RPS: {overall_rps:.4f}  |  naive baseline RPS: {overall_naive_rps:.4f}  "
        f"|  skill vs naive: {1 - overall_rps / overall_naive_rps:.3f}"
    )


def cmd_report(args):
    dfs = [load_experiment(args.backtest_dir, exp) for exp in args.experiment]
    combined = pd.concat(dfs, ignore_index=True)

    monthly = monthly_summary(combined)
    season = season_summary(combined)

    print("=== Monthly summary ===")
    print(monthly.to_string(index=False))
    print("\n=== Season summary ===")
    print(season.to_string(index=False))

    if args.out:
        monthly.to_csv(args.out, index=False)
        print(f"\nSaved monthly summary to {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run a walk-forward backtest sweep (monthly refit cadence).")
    p_run.add_argument("--db", default='data/forecast.db', help="Location of db that holds match results")
    p_run.add_argument("--experiment", required=True, help="e.g. 'baseline', 'with_weather' — one directory per experiment.")
    p_run.add_argument("--method", required=True, help="Model variant label within the experiment, e.g. 'dixon_coles'.")
    p_run.add_argument("--start-date", required=True, help="YYYY-MM-DD; first monthly checkpoint on/after this date.")
    p_run.add_argument("--end-date", required=True, help="YYYY-MM-DD; last checkpoint's scoring window ends here.")
    p_run.add_argument("--half-life", type=float, required=True)
    p_run.add_argument("--min-history-days", type=int, default=180, help="Skip checkpoints with less training history than this (burn-in).")
    p_run.add_argument("--draws", type=int, default=500)
    p_run.add_argument("--tune", type=int, default=500)
    p_run.add_argument("--chains", type=int, default=2)
    p_run.add_argument("--target-accept", type=float, default=0.9)
    p_run.add_argument("--seed", type=int, default=None)
    p_run.add_argument("--out-dir", default="data/backtest")
    p_run.add_argument(
        "--include-promotion-covariate",
        action="store_true",
        help="Fit with the promoted/relegated attack-defense shift terms (default off — the ablation baseline).",
    )
    p_run.set_defaults(func=cmd_run)

    p_report = sub.add_parser("report", help="Aggregate and compare one or more experiments.")
    p_report.add_argument("--backtest-dir", default="data/backtest")
    p_report.add_argument("--experiment", action="append", required=True, help="Repeatable — one or more experiment names to compare.")
    p_report.add_argument("--out", default=None)
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
