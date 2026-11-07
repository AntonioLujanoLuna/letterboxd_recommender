"""Discovery and pending queue management."""

import time
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
            WHERE claimed_at IS NULL AND COALESCE(attempts, 0) < ?
            ORDER BY priority DESC, discovered_at ASC
            LIMIT ?
        """, (MAX_QUEUE_ATTEMPTS, limit))
        return [dict(row) for row in cursor.fetchall()]


def remove_pending_user(username: str) -> None:
    """Remove a user from pending_users queue after successful scrape."""
    with get_db() as conn:
        conn.execute("DELETE FROM pending_users WHERE username = ?", (username,))


# A queued user that fails this many times is parked (kept in the table, never claimed again)
# instead of being retried forever or silently dropped.
MAX_QUEUE_ATTEMPTS = 3


def claim_pending_user(exclude: "set[str] | frozenset[str] | None" = None) -> dict | None:
    """Atomically claim the next unclaimed user (highest priority first).

    `exclude` skips usernames (e.g. ones that already failed in this run, so a failing profile
    isn't retried back-to-back).

    The row stays in pending_users, marked with claimed_at, until the caller removes it on
    success or releases it on failure. If the process dies in between, the claim goes stale
    and release_stale_claims() puts the user back, so nobody is lost.
    """
    excluded = sorted(exclude or ())
    skip_sql = f" AND username NOT IN ({','.join('?' * len(excluded))})" if excluded else ""
    for _ in range(5):  # lost races retry; each retry picks the next candidate
        with get_db() as conn:
            row = conn.execute(f"""
                SELECT username, discovered_from_type, discovered_from_id, discovered_at, priority
                FROM pending_users
                WHERE claimed_at IS NULL AND COALESCE(attempts, 0) < ?{skip_sql}
                ORDER BY priority DESC, discovered_at ASC
                LIMIT 1
            """, (MAX_QUEUE_ATTEMPTS, *excluded)).fetchone()
            if row is None:
                return None
            claimed = conn.execute(
                "UPDATE pending_users SET claimed_at = ? WHERE username = ? AND claimed_at IS NULL",
                (datetime.now().isoformat(), row['username']),
            ).rowcount
        if claimed:
            return dict(row)
    return None


def release_pending_user(username: str, count_attempt: bool = True) -> None:
    """Give a claimed user back to the queue (e.g. after a failed scrape).

    count_attempt=False is for interrupted work (shutdown), which isn't the user's fault.
    """
    with get_db() as conn:
        conn.execute(
            "UPDATE pending_users SET claimed_at = NULL, attempts = COALESCE(attempts, 0) + ? "
            "WHERE username = ?",
            (1 if count_attempt else 0, username),
        )


def release_stale_claims(max_age_seconds: float) -> int:
    """Release claims older than max_age_seconds (0 releases all). Returns how many."""
    cutoff = datetime.fromtimestamp(time.time() - max_age_seconds).isoformat()
    with get_db() as conn:
        return conn.execute(
            "UPDATE pending_users SET claimed_at = NULL WHERE claimed_at IS NOT NULL AND claimed_at <= ?",
            (cutoff,),
        ).rowcount


def get_pending_queue_stats() -> dict:
    """Get statistics about the pending users queue."""
    with get_db(read_only=True) as conn:
        cursor = conn.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(DISTINCT discovered_from_type) as source_types,
                AVG(priority) as avg_priority
            FROM pending_users
            WHERE COALESCE(attempts, 0) < ?
        """, (MAX_QUEUE_ATTEMPTS,))
        row = cursor.fetchone()
        parked = conn.execute(
            "SELECT COUNT(*) FROM pending_users WHERE COALESCE(attempts, 0) >= ?",
            (MAX_QUEUE_ATTEMPTS,),
        ).fetchone()[0]

        # Get breakdown by source type
        breakdown_cursor = conn.execute("""
            SELECT discovered_from_type, COUNT(*) as count
            FROM pending_users
            WHERE COALESCE(attempts, 0) < ?
            GROUP BY discovered_from_type
            ORDER BY count DESC
        """, (MAX_QUEUE_ATTEMPTS,))
        breakdown = {r['discovered_from_type']: r['count'] for r in breakdown_cursor.fetchall()}

        return {
            'total': row['total'],
            'source_types': row['source_types'],
            'avg_priority': row['avg_priority'],
            'breakdown': breakdown,
            'parked': parked,
        }
