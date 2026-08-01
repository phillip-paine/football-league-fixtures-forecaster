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

standings.json's projected_pts_* / title_prob / top4_prob / relegation_prob
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
from pathlib import Path

import numpy as np
import pandas as pd

# Promotion/European/relegation slot counts per division, used only for the
# Monte Carlo probability columns below. Extend this as you add divisions;
# an unlisted division falls back to a generic top-6 / bottom-4 guess.
DIVISION_CONFIG = {
    "Premier League": {"title_slots": 1, "top_slots": 4, "relegation_slots": 3},
    "Championship":   {"title_slots": 2, "top_slots": 6, "relegation_slots": 4},
    "League One":     {"title_slots": 2, "top_slots": 6, "relegation_slots": 4},
    "League Two":     {"title_slots": 2, "top_slots": 6, "relegation_slots": 2},
}
DEFAULT_CONFIG = {"title_slots": 1, "top_slots": 6, "relegation_slots": 4}
N_TRIALS = 50  # just for testing - will change for full runs later
RNG_SEED = 42


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


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
            "top4_prob": None,
            "relegation_prob": None,
        })

    rows.sort(key=lambda row: (-row["points"], -row["goal_diff"], row["team"]))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows


def simulate_season(df_div: pd.DataFrame, standings: list[dict], config: dict,
                    n_trials: int = N_TRIALS, seed: int = RNG_SEED,
                    full_season: bool = False) -> None:
    """Fills in projected_pts_* / title_prob / top4_prob / relegation_prob on
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
        row["top4_prob"] = float((ranks[:, i] < top_slots).mean())
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


def build_next_fixtures(season_fixtures: list[dict], window_days: int = 4, limit: int = 20) -> list[dict]:
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
             "happened. For validating a completed season's CSV (e.g. a "
             "past season used for testing) against its known real final "
             "table, without needing genuinely unplayed fixtures. Has no "
             "effect on season_fixtures.json / fixtures.json -- those "
             "still show real results and real is_played flags either way.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]  # drop a stray index column, if present
    out_root = Path(args.out_dir)

    for division, df_div in df.groupby("division"):
        slug = slugify(division)
        div_dir = out_root / slug
        div_dir.mkdir(parents=True, exist_ok=True)
        config = DIVISION_CONFIG.get(division, DEFAULT_CONFIG)

        standings = build_standings(df_div)
        simulate_season(df_div, standings, config, full_season=args.full_season_sim)
        season_fixtures = build_season_fixtures(df_div)
        next_fixtures = build_next_fixtures(season_fixtures)

        (div_dir / "standings.json").write_text(
            json.dumps({"competition": division, "teams": standings}, indent=2))
        (div_dir / "season_fixtures.json").write_text(
            json.dumps({"competition": division, "fixtures": season_fixtures}, indent=2))
        (div_dir / "fixtures.json").write_text(
            json.dumps({"competition": division, "fixtures": next_fixtures}, indent=2))

        simulated = "yes" if standings and standings[0]["projected_pts_mean"] is not None else "no unplayed matches"
        mode = "full-season (validation)" if args.full_season_sim else "remaining fixtures only"
        print(f"{division}: {len(standings)} teams, {len(season_fixtures)} fixtures total, "
              f"{len(next_fixtures)} in next fixture batch, simulated={simulated} "
              f"[{mode}] -> {div_dir}/")


if __name__ == "__main__":
    main()