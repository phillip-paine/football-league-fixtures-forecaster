"""
CLI for the minimal Dixon-Coles model: fit against the DB, or predict an
upcoming fixture window from a saved posterior.

    # one-time / weekly refit
    python run_dixon_coles.py fit --db data/forecast.db \\
        --as-of 2025-08-01 --half-life 180 \\
        --draws 1000 --tune 1000 --chains 4 \\
        --out data/model/dc_posterior_2025-08-01.nc

    # weekly prediction loop
    python run_dixon_coles.py predict --db data/forecast.db \\
        --idata data/model/dc_posterior_2025-08-01.nc \\
        --start-date 2025-08-16 --horizon-days 7 \\
        --out data/predictions/2025-08-16.csv

    # full-season prediction (use --end-date instead of --horizon-days)
    python run_dixon_coles.py predict --db data/forecast.db \\
        --idata data/model/dc_posterior_2024-08-01.nc \\
        --start-date 2024-08-01 --end-date 2025-06-01 \\
        --out data/predictions/2024-25_season.csv

"""

import argparse
import sys

import arviz as az
import pandas as pd

from features import (
    connect,
    build_team_index,
    build_division_index,
    load_training_matches,
    load_fixtures,
    check_team_season_consistency,
)
from model import build_model, fit as fit_model, ModelConfig, posterior_rates, match_outcome_probs


def resolve_id_to_name(conn, table: str, id_col: str, name_col: str) -> dict:
    """id -> human name lookup against a dimension table (teams,
    competitions). IndexMap only maps position<->id (confirmed against
    the real features.IndexMap: it stores id_to_idx/idx_to_id, nothing
    else) — names live here, in the actual dimension tables, one join
    away. Used purely for display/output; nothing in model/ depends on
    this, so a naming issue here can never affect fit correctness.
    """
    rows = conn.execute(f"SELECT {id_col}, {name_col} FROM {table}").fetchall()
    return dict(rows)


def cmd_fit(args):
    conn = connect(args.db)

    mismatches = check_team_season_consistency(conn)
    if mismatches:
        print(
            f"WARNING: check_team_season_consistency found {len(mismatches)} "
            "mismatch(es) between matches.competition_id and team_season. "
            "Division assignment may not be fully trustworthy yet:",
            file=sys.stderr,
        )
        for m in mismatches[:10]:
            print(f"  {m}", file=sys.stderr)
        if not args.ignore_consistency_warnings:
            print(
                "Re-run with --ignore-consistency-warnings to fit anyway.",
                file=sys.stderr,
            )
            sys.exit(1)

    team_index = build_team_index(conn)
    division_index = build_division_index(conn)

    train = load_training_matches(
        conn,
        as_of_date=args.as_of,
        half_life_days=args.half_life,
        team_index=team_index,
        division_index=division_index,
    )
    print(
        f"Loaded {train.n_matches} training matches "
        f"(n_teams={train.n_teams}, n_divisions={train.n_divisions})."
    )

    config = ModelConfig()
    _, idata = fit_model(
        train,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        random_seed=args.seed,
        config=config,
    )

    summary = az.summary(
        idata, var_names=["mu_intercept", "mu_home", "sigma_attack", "sigma_defense", "rho"]
    )
    print(summary)

    idata.to_netcdf(args.out)
    print(f"Saved posterior to {args.out}")


def cmd_predict(args):
    conn = connect(args.db)
    idata = az.from_netcdf(args.idata)

    team_index = build_team_index(conn)
    division_index = build_division_index(conn)
    n_teams_now = len(team_index.idx_to_id)
    n_divisions_now = len(division_index.idx_to_id)

    # Consistency check: compare team/division COUNTS between the fitted
    # posterior and what the DB produces today. We deliberately don't
    # compare idata's coordinate *labels* here — any posterior fit before
    # this patch has meaningless positional-string labels baked in (see
    # model_handoff.md), so label content can't be trusted as a
    # comparison basis. Counts are a weaker check (won't catch a
    # same-count reshuffle), but are always meaningful regardless of when
    # the posterior was fit.
    n_teams_fitted = idata.posterior.sizes["team"]
    n_divisions_fitted = idata.posterior.sizes["division"]
    if n_teams_fitted != n_teams_now:
        print(
            f"ERROR: fitted posterior has {n_teams_fitted} teams but the DB's "
            f"team_index currently has {n_teams_now}. Refit before predicting.",
            file=sys.stderr,
        )
        sys.exit(1)
    if n_divisions_fitted != n_divisions_now:
        print(
            f"ERROR: fitted posterior has {n_divisions_fitted} divisions but the "
            f"DB's division_index currently has {n_divisions_now}. Refit before predicting.",
            file=sys.stderr,
        )
        sys.exit(1)

    fixtures = load_fixtures(
        conn,
        start_date=args.start_date,
        horizon_days=args.horizon_days,
        end_date=args.end_date,
        team_index=team_index,
        division_index=division_index,
    )
    window_desc = f"end_date={args.end_date}" if args.end_date else f"horizon_days={args.horizon_days}"
    print(f"Loaded {fixtures.n_matches} fixtures from {args.start_date} ({window_desc}).")

    rates = posterior_rates(idata, fixtures.home_idx, fixtures.away_idx, fixtures.division_idx)
    probs = match_outcome_probs(rates, max_goals=args.max_goals)

    # Resolve real names straight from the dimension tables, via the
    # freshly-rebuilt IndexMap's idx_to_id (position -> raw id) — this is
    # correct regardless of what idata's own coordinate labels say.
    team_id_to_name = resolve_id_to_name(conn, "teams", "team_id", "canonical_name")
    division_id_to_name = resolve_id_to_name(conn, "competitions", "competition_id", "name")

    home_names = [team_id_to_name[team_index.idx_to_id[i]] for i in fixtures.home_idx]
    away_names = [team_id_to_name[team_index.idx_to_id[i]] for i in fixtures.away_idx]
    division_names_col = [division_id_to_name[division_index.idx_to_id[d]] for d in fixtures.division_idx]

    out = pd.DataFrame(
        {
            "match_id": fixtures.match_id,
            "match_date": fixtures.match_date,
            "division": division_names_col,
            "home_team": home_names,
            "away_team": away_names,
            "is_played": fixtures.is_played,
            "home_goals": fixtures.home_goals,
            "away_goals": fixtures.away_goals,
            "p_home_win": probs[:, 0],
            "p_draw": probs[:, 1],
            "p_away_win": probs[:, 2],
        }
    )
    out.to_csv(args.out, index=False)
    print(f"Saved {len(out)} predictions to {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fit = sub.add_parser("fit", help="Fit the model against training matches up to --as-of.")
    p_fit.add_argument("--db", required=True)
    p_fit.add_argument("--as-of", required=True, help="YYYY-MM-DD; train on matches strictly before this date.")
    p_fit.add_argument("--half-life", type=float, required=True, help="Decay half-life in days.")
    p_fit.add_argument("--draws", type=int, default=1000)
    p_fit.add_argument("--tune", type=int, default=1000)
    p_fit.add_argument("--chains", type=int, default=4)
    p_fit.add_argument("--target-accept", type=float, default=0.9)
    p_fit.add_argument("--seed", type=int, default=None)
    p_fit.add_argument("--out", required=True, help="Path to save the posterior (arviz netcdf, .nc).")
    p_fit.add_argument(
        "--ignore-consistency-warnings",
        action="store_true",
        help="Fit even if check_team_season_consistency() finds mismatches.",
    )
    p_fit.set_defaults(func=cmd_fit)

    p_pred = sub.add_parser("predict", help="Predict an upcoming fixture window from a saved posterior.")
    p_pred.add_argument("--db", required=True)
    p_pred.add_argument("--idata", required=True, help="Path to a posterior saved by `fit` (.nc).")
    p_pred.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    window = p_pred.add_mutually_exclusive_group(required=True)
    window.add_argument("--horizon-days", type=int, help="Predict [start-date, start-date + N days).")
    window.add_argument("--end-date", help="YYYY-MM-DD; predict [start-date, end-date) — use this for a full season rather than a weekly slate.")
    p_pred.add_argument("--max-goals", type=int, default=10, help="Scoreline grid truncation.")
    p_pred.add_argument("--out", required=True, help="Path to write predictions (.csv).")
    p_pred.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
