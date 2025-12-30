"""Discovery and pending queue management."""

from datetime import datetime

from .connection import get_db


def get_discovery_source(source_type: str, source_id: str) -> dict | None:
    """
    Get cached discovery source information.

    Returns dict with last_page_scraped, total_users_found, and scraped_at,
    or None if source not found.
    """
    with get_db(read_only=True) as conn:
        cursor = conn.execute("""
            SELECT last_page_scraped, total_users_found, scraped_at
            FROM discovery_sources
            WHERE source_type = ? AND source_id = ?
        """, (source_type, source_id))
        row = cursor.fetchone()
        return dict(row) if row else None


def update_discovery_source(
    source_type: str,
    source_id: str,
    last_page: int,
    total_users: int
) -> None:
    """Update discovery source cache with new page and user count."""
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO discovery_sources
            (source_type, source_id, last_page_scraped, total_users_found, scraped_at)
            VALUES (?, ?, ?, ?, ?)
        """, (source_type, source_id, last_page, total_users, datetime.now().isoformat()))


def add_pending_users(
    usernames: list[str],
    source_type: str,
    source_id: str,
    priority: int = 50
) -> int:
    """
    Add discovered usernames to pending_users queue.

    Returns count of newly added users (ignores duplicates and existing users).
    """
    with get_db() as conn:
        # Get existing users in user_films and pending_users
        existing_scraped = {
            r['username'] for r in conn.execute(
                "SELECT DISTINCT username FROM user_films"
            )
        }
        existing_pending = {
            r['username'] for r in conn.execute(
                "SELECT username FROM pending_users"
            )
        }

        # Filter to truly new users and de-duplicate this batch while preserving order
        seen = set()
        new_users = []
        for u in usernames:
            if u in seen:
                continue
            seen.add(u)
            if u in existing_scraped or u in existing_pending:
                continue
            new_users.append(u)

        if not new_users:
            return 0

        # Insert new pending users
        timestamp = datetime.now().isoformat()
        conn.executemany("""
            INSERT OR IGNORE INTO pending_users
            (username, discovered_from_type, discovered_from_id, discovered_at, priority)
            VALUES (?, ?, ?, ?, ?)
        """, [(u, source_type, source_id, timestamp, priority) for u in new_users])

        return len(new_users)


def get_pending_users(limit: int = 50) -> list[dict]:
    """
    Get pending users from queue, ordered by priority (desc) then discovery time (asc).

    Returns list of dicts with username, discovered_from_type, discovered_from_id,
    discovered_at, and priority.
    """
    with get_db(read_only=True) as conn:
        cursor = conn.execute("""
            SELECT username, discovered_from_type, discovered_from_id, discovered_at, priority
            FROM pending_users
            ORDER BY priority DESC, discovered_at ASC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]


def remove_pending_user(username: str) -> None:
    """Remove a user from pending_users queue after successful scrape."""
    with get_db() as conn:
        conn.execute("DELETE FROM pending_users WHERE username = ?", (username,))


def get_pending_queue_stats() -> dict:
    """Get statistics about the pending users queue."""
    with get_db(read_only=True) as conn:
        cursor = conn.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(DISTINCT discovered_from_type) as source_types,
                AVG(priority) as avg_priority
            FROM pending_users
        """)
        row = cursor.fetchone()

        # Get breakdown by source type
        breakdown_cursor = conn.execute("""
            SELECT discovered_from_type, COUNT(*) as count
            FROM pending_users
            GROUP BY discovered_from_type
            ORDER BY count DESC
        """)
        breakdown = {r['discovered_from_type']: r['count'] for r in breakdown_cursor.fetchall()}

        return {
            'total': row['total'],
            'source_types': row['source_types'],
            'avg_priority': row['avg_priority'],
            'breakdown': breakdown
        }
