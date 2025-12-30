"""Scraping session tracking."""

from datetime import datetime

from .connection import get_db


def create_scrape_session() -> int:
    """Create a new scraping session record and return its ID."""
    with get_db() as conn:
        cursor = conn.execute("""
            INSERT INTO scrape_sessions (started_at, status, last_activity)
            VALUES (?, 'running', ?)
        """, (datetime.now().isoformat(), datetime.now().isoformat()))
        return cursor.lastrowid


def update_session_progress(session_id: int, users_scraped: int, films_added: int) -> None:
    """Update session progress counters and heartbeat."""
    with get_db() as conn:
        conn.execute("""
            UPDATE scrape_sessions
            SET users_scraped = ?, films_added = ?, last_activity = ?
            WHERE id = ?
        """, (users_scraped, films_added, datetime.now().isoformat(), session_id))


def complete_session(session_id: int, status: str = "completed") -> None:
    """Mark a scraping session as completed/interrupted."""
    with get_db() as conn:
        conn.execute("""
            UPDATE scrape_sessions
            SET status = ?, completed_at = ?, last_activity = ?
            WHERE id = ?
        """, (status, datetime.now().isoformat(), datetime.now().isoformat(), session_id))


def get_session_history(limit: int = 10) -> list[dict]:
    """Return recent scraping sessions for visibility."""
    with get_db(read_only=True) as conn:
        rows = conn.execute("""
            SELECT id, started_at, completed_at, status, users_scraped, films_added, last_activity
            FROM scrape_sessions
            ORDER BY started_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]
