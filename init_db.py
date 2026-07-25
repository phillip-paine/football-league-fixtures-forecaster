"""
Create (or recreate) the forecast engine's SQLite database from schema.sql.

Usage:
    poetry run python init_db.py [path/to/db.sqlite3]

Refuses to run against an existing file, since this schema is designed
to never need migrations -- if you find yourself wanting to re-run this
against a populated DB, that's a signal something in the design needs
revisiting, not a routine operation.
"""
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB_PATH = "forecast.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_db(db_path: str) -> None:
    path = Path(db_path)
    if path.exists():
        raise SystemExit(
            f"{db_path} already exists. This script only creates a fresh DB; "
            "delete it manually first if you're sure you want to start over."
        )

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")  # safe for a reader + weekly writer
        conn.executescript(SCHEMA_PATH.read_text())
        conn.commit()
        print(f"Created {db_path} from {SCHEMA_PATH.name}")
    finally:
        conn.close()


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB_PATH
    init_db(target)