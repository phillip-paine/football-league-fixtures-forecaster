"""
Download football-data.co.uk season/league CSVs, with local caching.

This module needs real internet access to football-data.co.uk. Test it
on your own machine -- the sandbox this was developed in only allows
egress to package registries, not to data sources like this one.
"""
from __future__ import annotations

from pathlib import Path

import requests

from .config import BASE_URL

USER_AGENT = "forecast-engine-ingest/0.1 (personal research project)"


def season_code(start_year: int) -> str:
    """2025 -> '2526' (the format football-data.co.uk uses in its URLs)."""
    end_year = (start_year + 1) % 100
    return f"{start_year % 100:02d}{end_year:02d}"


def fetch_csv(start_year: int, league_code: str, cache_dir: Path, force: bool = False) -> Path:
    """
    Download one season/league CSV, or return the cached copy.

    Caching matters for two different reasons depending on which phase
    you're in: for the one-time historical build it avoids re-downloading
    ~30 seasons every time you re-run and tweak the loader; for the
    weekly loop it means only the current season's file is ever fetched
    fresh (force=True), everything older is untouched.
    """
    season_dir = Path(cache_dir) / season_code(start_year)
    season_dir.mkdir(parents=True, exist_ok=True)
    dest = season_dir / f"{league_code}.csv"

    if dest.exists() and not force:
        return dest

    url = f"{BASE_URL}/{season_code(start_year)}/{league_code}.csv"
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest