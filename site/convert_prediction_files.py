"""
Convert run_model.py predict's CSV output into the JSON files index.html
fetches. Run this after every predict run.

Usage:
    python convert_predictions.py predictions.csv --out-dir data

Writes, per division found in the CSV:
    data/<division-slug>/standings.json        current table (real, computed
                                                 from is_played rows only)
    data/<division-slug>/season_fixtures.json   every match, played + upcoming
                                                 (feeds the Results Grid)
    data/<division-slug>/fixtures.json          the next unplayed gameweek
                                                 (feeds the weekly fixture list)

standings.json's projected_pts_* / title_prob / top4_prob / relegation_prob
fields are written as null. Those need the season-simulation (Monte Carlo)
step, which per build_log.md doesn't exist yet -- this script only ever
reports the real, played-so-far table. The frontend shows "-" for those
columns until real numbers exist; it should never fabricate them.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd


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

    rows.sort(key=lambda r: (-r["points"], -r["goal_diff"], r["team"]))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows


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


def build_next_fixtures(season_fixtures: list[dict], limit: int = 15) -> list[dict]:
    upcoming = [f for f in season_fixtures if not f["is_played"]]
    upcoming.sort(key=lambda f: f["match_date"])
    if not upcoming:
        return []
    next_date = upcoming[0]["match_date"]
    same_day = [f for f in upcoming if f["match_date"] == next_date]
    # If the "next gameweek" concept spans a weekend rather than one date in
    # your calendar, swap this for whatever grouping key you actually have
    # (a gameweek/round number is cleaner than a raw date, if you have one).
    return same_day[:limit] if same_day else upcoming[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    out_root = Path(args.out_dir)

    for division, df_div in df.groupby("division"):
        slug = slugify(division)
        div_dir = out_root / slug
        div_dir.mkdir(parents=True, exist_ok=True)

        standings = build_standings(df_div)
        season_fixtures = build_season_fixtures(df_div)
        next_fixtures = build_next_fixtures(season_fixtures)

        (div_dir / "standings.json").write_text(
            json.dumps({"competition": division, "teams": standings}, indent=2))
        (div_dir / "season_fixtures.json").write_text(
            json.dumps({"competition": division, "fixtures": season_fixtures}, indent=2))
        (div_dir / "fixtures.json").write_text(
            json.dumps({"competition": division, "fixtures": next_fixtures}, indent=2))

        print(f"{division}: {len(standings)} teams, {len(season_fixtures)} fixtures total, "
              f"{len(next_fixtures)} in next fixture batch -> {div_dir}/")


if __name__ == "__main__":
    main()
