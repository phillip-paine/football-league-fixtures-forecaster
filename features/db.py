"""
Thin connection helper.

schema.sql's own header note is the single source of truth: every
connection must turn foreign keys on manually (SQLite defaults it
off). The extraction layer leans on FK integrity — team/competition
ids coming out of `matches` are trusted to exist in `teams` /
`competitions` without a defensive existence check — so this pragma
isn't optional.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path


def connect(db_path: str | Path, read_only: bool = True) -> sqlite3.Connection:
    """Open a connection configured the way the rest of this layer assumes.

    read_only=True opens SQLite's URI read-only mode (`mode=ro`). The
    extraction layer never writes, so this is the safe default — it
    fails loudly instead of silently if a bug ever tries to INSERT/UPDATE
    here instead of in the ingestion code where it belongs.
    """
    db_path = Path(db_path)
    if read_only:
        if not db_path.exists():
            raise FileNotFoundError(
                f"{db_path} does not exist (read-only connection can't create it)"
            )
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(str(db_path))

    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def connection(db_path: str | Path, read_only: bool = True):
    conn = connect(db_path, read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()