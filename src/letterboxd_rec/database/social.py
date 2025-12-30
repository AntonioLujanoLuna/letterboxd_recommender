"""Social graph (follows) management."""

from datetime import datetime

from .connection import get_db


def save_user_follows(follower: str, followees: list[str]) -> int:
    """
    Save follow relationships for a user.

    Args:
        follower: The username doing the following
        followees: List of usernames they follow

    Returns:
        Number of new relationships saved
    """
    if not followees:
        return 0

    with get_db() as conn:
        scraped_at = datetime.now().isoformat()
        conn.executemany(
            """
            INSERT OR IGNORE INTO user_follows (follower, followee, scraped_at)
            VALUES (?, ?, ?)
            """,
            [(follower, followee, scraped_at) for followee in followees]
        )
        # Return approximate count (executemany doesn't give per-row info)
        return len(followees)


def save_user_followers(followee: str, followers: list[str]) -> int:
    """
    Save follow relationships where others follow this user.

    Args:
        followee: The username being followed
        followers: List of usernames who follow them

    Returns:
        Number of new relationships saved
    """
    if not followers:
        return 0

    with get_db() as conn:
        scraped_at = datetime.now().isoformat()
        conn.executemany(
            """
            INSERT OR IGNORE INTO user_follows (follower, followee, scraped_at)
            VALUES (?, ?, ?)
            """,
            [(follower, followee, scraped_at) for follower in followers]
        )
        return len(followers)


def get_social_graph_stats() -> dict:
    """Get statistics about the social graph."""
    with get_db(read_only=True) as conn:
        total = conn.execute("SELECT COUNT(*) FROM user_follows").fetchone()[0]
        unique_followers = conn.execute(
            "SELECT COUNT(DISTINCT follower) FROM user_follows"
        ).fetchone()[0]
        unique_followees = conn.execute(
            "SELECT COUNT(DISTINCT followee) FROM user_follows"
        ).fetchone()[0]

        return {
            "total_edges": total,
            "unique_followers": unique_followers,
            "unique_followees": unique_followees,
        }
