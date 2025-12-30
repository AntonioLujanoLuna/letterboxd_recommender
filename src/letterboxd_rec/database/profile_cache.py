"""User profile caching functionality."""

import json
import logging
from datetime import datetime

from .connection import get_db, parse_timestamp_naive

logger = logging.getLogger(__name__)


def load_cached_profile(username: str, max_age_days: int = 7) -> dict | None:
    """
    Load cached user profile if it exists and is recent enough.

    Cache is invalidated if:
    - Profile schema version doesn't match current version
    - Profile is older than max_age_days
    - User's lists have been updated more recently than the profile
    - User's films have been updated more recently than the profile
    """
    from ..config import PROFILE_SCHEMA_VERSION

    with get_db(read_only=True) as conn:
        cursor = conn.execute("""
            SELECT profile_data, updated_at, schema_version
            FROM user_profiles
            WHERE username = ?
        """, (username,))
        row = cursor.fetchone()

        if not row:
            return None

        # Check schema version FIRST before any other validation
        cached_version = row['schema_version'] if 'schema_version' in row.keys() else 0
        if cached_version != PROFILE_SCHEMA_VERSION:
            logger.debug(f"Profile cache invalidated for {username} - schema version mismatch ({cached_version} != {PROFILE_SCHEMA_VERSION})")
            return None

        # Use helper to ensure naive datetime for consistent comparisons
        profile_updated_at = parse_timestamp_naive(row['updated_at'])
        now = datetime.now()

        # Check age
        if (now - profile_updated_at).days > max_age_days:
            return None

        # Check if user_lists have been updated more recently
        lists_cursor = conn.execute("""
            SELECT MAX(scraped_at) as last_list_update
            FROM user_lists
            WHERE username = ?
        """, (username,))
        lists_row = lists_cursor.fetchone()

        if lists_row and lists_row['last_list_update']:
            last_list_update = parse_timestamp_naive(lists_row['last_list_update'])
            if last_list_update > profile_updated_at:
                logger.debug(f"Profile cache invalidated for {username} - lists updated more recently")
                return None

        # Check if user_films have been updated more recently
        films_cursor = conn.execute("""
            SELECT MAX(scraped_at) as last_film_update
            FROM user_films
            WHERE username = ?
        """, (username,))
        films_row = films_cursor.fetchone()

        if films_row and films_row['last_film_update']:
            last_film_update = parse_timestamp_naive(films_row['last_film_update'])
            if last_film_update > profile_updated_at:
                logger.debug(f"Profile cache invalidated for {username} - films updated more recently")
                return None

        return json.loads(row['profile_data'])


def save_user_profile(username: str, profile_data: dict) -> None:
    """Save user profile to cache with naive datetime timestamp and schema version."""
    from ..config import PROFILE_SCHEMA_VERSION

    with get_db() as conn:
        # Always use naive datetime for consistency
        conn.execute("""
            INSERT OR REPLACE INTO user_profiles (username, profile_data, updated_at, schema_version)
            VALUES (?, ?, ?, ?)
        """, (username, json.dumps(profile_data), datetime.now().isoformat(), PROFILE_SCHEMA_VERSION))


def purge_stale_profile_caches() -> int:
    """
    Delete all cached profiles with outdated schema versions.

    Returns:
        Count of profiles deleted
    """
    from ..config import PROFILE_SCHEMA_VERSION

    with get_db() as conn:
        cursor = conn.execute("""
            DELETE FROM user_profiles
            WHERE schema_version IS NULL OR schema_version != ?
        """, (PROFILE_SCHEMA_VERSION,))
        return cursor.rowcount
