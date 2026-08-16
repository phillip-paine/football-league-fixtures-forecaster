"""
Convert run_model.py predict's CSV output into the JSON files index.html
fetches. Run this after every predict run.

Usage:
    python convert_predictions.py predictions.csv --out-dir data

Writes, per division found in the CSV:
    data/<division-slug>/standings.json        current table + Monte Carlo
                                                 season projection (see below)
    data/<division-slug>/season_fixtures.json   every match, played + upcoming
                                                 (feeds the Results Grid)
    data/<division-slug>/fixtures.json          the next upcoming gameweek
                                                 (feeds the weekly fixture list)

standings.json's projected_pts_* / title_prob / top5_prob / relegation_prob
are produced by simulating the season's remaining (unplayed) fixtures many
times, drawing each match's outcome from the model's own p_home_win/p_draw/
p_away_win -- not a separate model, just repeated sampling of what the CSV
already gives us. This is a simplified stand-in for the full posterior-based
Monte Carlo in forecast_engine_design_notes.md (it treats each match as an
independent categorical draw rather than sampling correlated team-strength
draws from the fitted posterior, and breaks points-ties without a
goal-difference tiebreak) -- fine for an honest "roughly how likely" read,
worth revisiting once run_model.py grows a real season-simulation mode.
If a division has no unplayed matches left (season already finished), these
fields stay null rather than simulating nothing.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Promotion/European/relegation slot counts per division, used only for the
# Monte Carlo probability columns below. Extend this as you add divisions;
# an unlisted division falls back to a generic top-6 / bottom-4 guess.
DIVISION_CONFIG = {
    "Premier League": {"title_slots": 1, "top_slots": 5, "relegation_slots": 3},
    "Championship":   {"title_slots": 2, "top_slots": 6, "relegation_slots": 4},
    "League One":     {"title_slots": 2, "top_slots": 6, "relegation_slots": 4},
    "League Two":     {"title_slots": 2, "top_slots": 6, "relegation_slots": 2},
}
DEFAULT_CONFIG = {"title_slots": 1, "top_slots": 6, "relegation_slots": 4}
N_TRIALS = 50  # just for testing - will change for full runs later
RNG_SEED = 42


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def derive_gameweek_cutoff_date(df_div: pd.DataFrame, gameweek: int) -> pd.Timestamp | None:
    """Approximates a gameweek boundary purely from match_date order, since
    the CSV has no explicit round/gameweek column. Assumes a standard
    single round-robin (n_teams // 2 matches per gameweek): all matches are
    sorted by date and chunked into blocks of that size in order; gameweek
    N (1-indexed) is the N-th block, and the cutoff is that block's last
    match_date -- "as of the end of gameweek N".

    This is an approximation, not a real fixture-list lookup: postponed or
    rearranged matches mean a date-sorted chunk of size n_teams//2 won't
    always line up with the real published gameweek numbering. If you know
    the actual calendar cutoff you want, use --as-of-date instead -- it's
    exact and (for a CSV spanning multiple divisions) applies the same real
    date everywhere, rather than each division's own gameweek clock.

    Returns None if gameweek is out of range for this division's data.
    """
    n_teams = len(pd.unique(pd.concat([df_div.home_team, df_div.away_team])))
    round_size = max(n_teams // 2, 1)
    ordered = df_div.sort_values("match_date")
    total_rounds = -(-len(ordered) // round_size)  # ceil division
    if gameweek <= 0 or gameweek > total_rounds:
        return None
    end_idx = min(gameweek * round_size, len(ordered))
    return pd.Timestamp(ordered.iloc[end_idx - 1].match_date)


def apply_as_of_cutoff(df_div: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """Returns a copy of df_div with is_played/home_goals/away_goals
    overridden to reflect "as of cutoff_date":

    - Matches on or before cutoff_date are untouched -- if the CSV already
      says they're played with a real score, that real score is exactly
      what feeds the current table and its points. This is deliberate:
      genuinely-known results should count as genuinely known.
    - Matches after cutoff_date are forced to is_played=False with goals
      blanked out, even if the CSV (e.g. a completed historical season
      used for validation) shows them as already played. This is what
      makes the gameweek right after the cutoff "predict into the
      unknown" rather than quietly leaking the real answer -- every
      downstream function (build_standings, simulate_season,
      build_next_fixtures) already branches on is_played, so overriding
      it here is the only change needed; nothing else has to know a
      cutoff was applied.
    """
    out = df_div.copy()
    match_dates = pd.to_datetime(out.match_date)
    after_cutoff = match_dates >= cutoff_date
    out.loc[after_cutoff, "is_played"] = False
    out.loc[after_cutoff, "home_goals"] = np.nan
    out.loc[after_cutoff, "away_goals"] = np.nan
    return out


def build_standings(df_div: pd.DataFrame) -> list[dict]:
    played = df_div[df_div.is_played]
    teams = pd.unique(pd.concat([df_div.home_team, df_div.away_team]))

    rows = []
    for team in teams:
        pts = gf = ga = wins = draws = losses = 0

        home = played[played.home_team == team]
        for _, m in home.iterrows():
            gf += m.home_goals
            ga += m.away_goals
            if m.home_goals > m.away_goals:
                pts += 3; wins += 1
            elif m.home_goals == m.away_goals:
                pts += 1; draws += 1
            else:
                losses += 1

        away = played[played.away_team == team]
        for _, m in away.iterrows():
            gf += m.away_goals
            ga += m.home_goals
            if m.away_goals > m.home_goals:
                pts += 3; wins += 1
            elif m.away_goals == m.home_goals:
                pts += 1; draws += 1
            else:
                losses += 1

        rows.append({
            "team": team,
            "played": wins + draws + losses,
            "points": pts,
            "goal_diff": int(gf - ga),
            "projected_pts_low": None,
            "projected_pts_mean": None,
            "projected_pts_high": None,
            "title_prob": None,
            "top5_prob": None,
            "relegation_prob": None,
        })

    rows.sort(key=lambda row: (-row["points"], -row["goal_diff"], row["team"]))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows


def simulate_season(df_div: pd.DataFrame, standings: list[dict], config: dict,
                    n_trials: int = N_TRIALS, seed: int = RNG_SEED,
                    full_season: bool = False) -> None:
    """Fills in projected_pts_* / title_prob / top5_prob / relegation_prob on
    `standings` in place, by simulating matches n_trials times.

    full_season=False (default): only simulates matches where is_played is
    False, starting from each team's real current points. No-op if there's
    nothing left to play (a completed season stays all-null, as before).

    full_season=True: ignores is_played entirely and simulates every match
    in df_div from a clean slate (0 points), using each match's own
    predicted probabilities -- including ones that already happened. This
    is a "pretend the season hasn't happened yet" validation mode: useful
    for feeding in an already-completed season and comparing the fully
    simulated table against the known real final table, without needing
    genuinely unplayed fixtures. Requires p_home_win/p_draw/p_away_win to
    be populated on played rows too (run_model.py predict already does
    this -- it computes probabilities regardless of is_played).
    """
    matches = df_div if full_season else df_div[~df_div.is_played]
    if matches.empty:
        return

    teams = [r["team"] for r in standings]
    team_pos = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    starting_points = np.zeros(n_teams) if full_season else \
        np.array([r["points"] for r in standings], dtype=float)

    home_idx = matches.home_team.map(team_pos).to_numpy()
    away_idx = matches.away_team.map(team_pos).to_numpy()
    p_home = matches.p_home_win.to_numpy()
    p_draw = matches.p_draw.to_numpy()
    # p_away implied as the remainder, so the three always sum to 1 even if
    # rounding in the CSV made them not quite add up.

    rng = np.random.default_rng(seed)
    n_matches = len(matches)
    r = rng.random((n_trials, n_matches))
    is_home_win = r < p_home[None, :]
    is_draw = (~is_home_win) & (r < (p_home + p_draw)[None, :])
    is_away_win = ~(is_home_win | is_draw)

    home_pts = np.where(is_home_win, 3, np.where(is_draw, 1, 0)).astype(float)
    away_pts = np.where(is_away_win, 3, np.where(is_draw, 1, 0)).astype(float)

    home_onehot = np.zeros((n_matches, n_teams))
    home_onehot[np.arange(n_matches), home_idx] = 1
    away_onehot = np.zeros((n_matches, n_teams))
    away_onehot[np.arange(n_matches), away_idx] = 1

    final_points = starting_points[None, :] + home_pts @ home_onehot + away_pts @ away_onehot

    # Rank per trial (0 = 1st place). Ties broken by current team order,
    # not goal difference -- a known approximation, see module docstring.
    order = np.argsort(-final_points, axis=1, kind="stable")
    ranks = np.argsort(order, axis=1)

    title_slots = config["title_slots"]
    top_slots = config["top_slots"]
    releg_slots = config["relegation_slots"]

    for i, row in enumerate(standings):
        row["projected_pts_low"] = int(np.percentile(final_points[:, i], 10))
        row["projected_pts_mean"] = int(round(final_points[:, i].mean()))
        row["projected_pts_high"] = int(np.percentile(final_points[:, i], 90))
        row["title_prob"] = float((ranks[:, i] < title_slots).mean())
        row["top5_prob"] = float((ranks[:, i] < top_slots).mean())
        row["relegation_prob"] = float((ranks[:, i] >= n_teams - releg_slots).mean())


def build_season_fixtures(df_div: pd.DataFrame) -> list[dict]:
    out = []
    for _, m in df_div.iterrows():
        item = {
            "match_id": int(m.match_id),
            "match_date": pd.Timestamp(m.match_date).strftime("%Y-%m-%d"),
            "home_team": m.home_team,
            "away_team": m.away_team,
            "is_played": bool(m.is_played),
        }
        if item["is_played"]:
            item["actual_score"] = {"home": int(m.home_goals), "away": int(m.away_goals)}
        else:
            item["home_win_prob"] = float(m.p_home_win)
            item["draw_prob"] = float(m.p_draw)
            item["away_win_prob"] = float(m.p_away_win)
        out.append(item)
    return out


def build_next_fixtures(season_fixtures: list[dict], window_days: int = 4,
                        limit: int = 20, anchor_on_last: bool = False) -> list[dict]:
    """Picks the fixture-list window shown in the "This Week's Fixtures"
    panel (and which fixture closestFixture() in index.html can pick a
    headline from -- it only ever sees whatever this function returns).

    anchor_on_last=False (default): the next unplayed gameweek, as normal.
    anchor_on_last=True: the season's LAST gameweek by date, regardless of
    is_played. Used for --full-season-sim, since a completed validation
    season has no unplayed matches at all -- without this, fixtures.json
    would be empty and the headline / fixture list would have nothing to
    show. Requires season_fixtures to carry probabilities on played rows
    too, which build_season_fixtures now does.
    """
    if anchor_on_last:
        if not season_fixtures:
            return []
        anchor = max(pd.Timestamp(f["match_date"]) for f in season_fixtures)
        window_start = anchor - pd.Timedelta(days=window_days)
        in_window = [f for f in season_fixtures
                     if window_start <= pd.Timestamp(f["match_date"]) <= anchor]
        in_window.sort(key=lambda f: f["match_date"])
        return in_window[:limit]

    upcoming = [f for f in season_fixtures if not f["is_played"]]
    upcoming.sort(key=lambda f: f["match_date"])
    if not upcoming:
        return []
    next_date = pd.Timestamp(upcoming[0]["match_date"])
    window_end = next_date + pd.Timedelta(days=window_days)
    in_window = [f for f in upcoming if next_date <= pd.Timestamp(f["match_date"]) <= window_end]
    return in_window[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument(
        "--full-season-sim", action="store_true",
        help="Ignore is_played and simulate every match from a clean slate "
             "using predicted probabilities, even ones that already "
             "happened. Equivalent to a cutoff before the season starts -- "
             "shorthand for validating a completed season's CSV against "
             "its known real final table with nothing held back as real. "
             "Mutually exclusive with --as-of-gameweek/--as-of-date, which "
             "let real results count up to a chosen point instead.")
    cutoff_group = ap.add_mutually_exclusive_group()
    cutoff_group.add_argument(
        "--as-of-gameweek", type=int, default=None,
        help="Validation mode: keep real results/points for matches through "
             "gameweek N, then treat everything after N as unplayed (even on "
             "an already-completed season), so gameweek N+1 becomes "
             "'predicting into the unknown' from a real point in the season. "
             "Gameweek boundaries are approximated per division from "
             "match_date order (see derive_gameweek_cutoff_date) since the "
             "CSV has no round-number column -- use --as-of-date instead if "
             "you know the exact calendar cutoff you want.")
    cutoff_group.add_argument(
        "--as-of-date", type=str, default=None,
        help="Same idea as --as-of-gameweek, but an exact calendar date "
             "(YYYY-MM-DD) rather than an approximated gameweek boundary. "
             "Applies the same real cutoff across every division in the "
             "CSV, which --as-of-gameweek can't (each division's gameweek "
             "clock runs independently).")
    args = ap.parse_args()
    if args.full_season_sim and (args.as_of_gameweek is not None or args.as_of_date is not None):
        ap.error("--full-season-sim can't be combined with --as-of-gameweek/--as-of-date "
                 "-- full-season-sim already treats the whole season as unplayed.")

    df = pd.read_csv(args.csv_path)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]  # drop a stray index column, if present
    out_root = Path(args.out_dir)

    for division, df_div in df.groupby("division"):
        slug = slugify(division)
        div_dir = out_root / slug
        div_dir.mkdir(parents=True, exist_ok=True)
        config = DIVISION_CONFIG.get(division, DEFAULT_CONFIG)

        cutoff_desc = "none"
        if args.as_of_date is not None:
            cutoff_date = pd.Timestamp(args.as_of_date)
            df_div = apply_as_of_cutoff(df_div, cutoff_date)
            cutoff_desc = f"as of {cutoff_date.date()}"
        elif args.as_of_gameweek is not None:
            cutoff_date = derive_gameweek_cutoff_date(df_div, args.as_of_gameweek)
            if cutoff_date is None:
                print(f"{division}: gameweek {args.as_of_gameweek} is out of range for this "
                      f"division's data -- skipping cutoff, using the CSV's real is_played as-is.",
                      file=sys.stderr)
            else:
                df_div = apply_as_of_cutoff(df_div, cutoff_date)
                cutoff_desc = f"as of gameweek {args.as_of_gameweek} (~{cutoff_date.date()})"

        # With the cutoff already applied above (or --full-season-sim below),
        # every function here just reads is_played -- nothing else needs to
        # know a cutoff exists.
        standings = build_standings(df_div)
        simulate_season(df_div, standings, config, full_season=args.full_season_sim)
        season_fixtures = build_season_fixtures(df_div)
        next_fixtures = build_next_fixtures(season_fixtures, anchor_on_last=args.full_season_sim)

        (div_dir / "standings.json").write_text(
            json.dumps({"competition": division, "teams": standings}, indent=2))
        (div_dir / "season_fixtures.json").write_text(
            json.dumps({"competition": division, "fixtures": season_fixtures}, indent=2))
        (div_dir / "fixtures.json").write_text(
            json.dumps({"competition": division, "fixtures": next_fixtures}, indent=2))

        simulated = "yes" if standings and standings[0]["projected_pts_mean"] is not None else "no unplayed matches"
        mode = "full-season (validation)" if args.full_season_sim else f"cutoff={cutoff_desc}"
        print(f"{division}: {len(standings)} teams, {len(season_fixtures)} fixtures total, "
              f"{len(next_fixtures)} in next fixture batch, simulated={simulated} "
              f"[{mode}] -> {div_dir}/")


if __name__ == "__main__":
    main()