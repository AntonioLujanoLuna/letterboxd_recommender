"""Database maintenance operations."""

import sqlite3

from ..config import DB_PATH


def run_maintenance(vacuum: bool = True, analyze: bool = True) -> None:
    """
    Run optional VACUUM/ANALYZE after bulk loads.
    Uses a dedicated connection to avoid interfering with pooled transactions.
    """
    if not vacuum and not analyze:
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        if vacuum:
            conn.execute("VACUUM")
        if analyze:
            conn.execute("ANALYZE")
        conn.commit()
    finally:
        conn.close()
