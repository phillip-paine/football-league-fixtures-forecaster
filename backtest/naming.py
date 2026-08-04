"""
Filename/directory conventions for backtest run artifacts.

Layout:

    data/backtest/<experiment>/<experiment>__<method>__<mm>_<ss>__<window>.csv

  - <mm>_<ss>  = the refit ("as-of") checkpoint's month + season code,
                 e.g. "09_202425" = a September refit, 2024-25 season.
  - <window>   = the scored prediction window, in the same mm_ss shape —
                 a single month under the confirmed monthly-refit cadence
                 (in which case it's identical to <mm>_<ss>, since a
                 monthly checkpoint's training cutoff and scoring window
                 are the same calendar month), or a "start-end" range if
                 a future cadence ever refits less often than it scores.

One directory per experiment (e.g. "baseline", "with_weather") — an
ablation test compares two experiment directories' contents. `method`
identifies the specific model variant within an experiment (kept
separate from `experiment` so, e.g., a "half_life_sweep" experiment can
contain multiple methods like "hl90"/"hl180"/"hl365" side by side).

Season code: an English season runs ~Aug-May; a match's season is
labelled by its start year (Sept 2024 and March 2025 are both "202425").
Matches falling in Jun/Jul (rare — close-season, playoffs) are assigned
to the season that just finished, i.e. the season-boundary month is July
(month >= 7 -> that year starts the season).
"""

from datetime import date, timedelta


def season_code(d: date) -> str:
    start_year = d.year if d.month >= 7 else d.year - 1
    return f"{start_year}{str((start_year + 1) % 100).zfill(2)}"


def checkpoint_label(d: date) -> str:
    return f"{d.month:02d}_{season_code(d)}"


def window_label(window_start: date, window_end_exclusive: date) -> str:
    if window_end_exclusive <= window_start:
        raise ValueError("window_end_exclusive must be after window_start")
    last_included = window_end_exclusive - timedelta(days=1)
    start_lbl = checkpoint_label(window_start)
    end_lbl = checkpoint_label(last_included)
    return start_lbl if start_lbl == end_lbl else f"{start_lbl}-{end_lbl}"


def _validate_component(name: str, value: str) -> None:
    if "__" in value or "/" in value or "\\" in value:
        raise ValueError(f"{name} must not contain '__', '/', or '\\\\': {value!r}")


def backtest_filename(
        experiment: str, method: str, as_of: date, window_start: date, window_end_exclusive: date
) -> str:
    _validate_component("experiment", experiment)
    _validate_component("method", method)
    return (
        f"{experiment}__{method}__{checkpoint_label(as_of)}__"
        f"{window_label(window_start, window_end_exclusive)}.csv"
    )
