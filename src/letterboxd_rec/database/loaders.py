"""Data loading utilities for the database."""

import json
import logging
from collections import defaultdict

from .connection import get_db

logger = logging.getLogger(__name__)


def load_json(val):
    """Safely load JSON from db field."""
    if not val:
        return []
    if isinstance(val, list):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse JSON '{val[:50]}...': {e}")
        return []


def load_user_lists(username: str) -> list[dict]:
    """Load all list entries for a user from database."""
    with get_db(read_only=True) as conn:
        cursor = conn.execute("""
            SELECT username, list_slug, list_name, is_ranked, is_favorites, position, film_slug
            FROM user_lists
            WHERE username = ?
        """, (username,))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def load_user_films_batch(usernames: list[str]) -> dict[str, list[dict]]:
    """
    Load films for multiple users in a single query.

    Much more efficient than N separate queries for collaborative filtering.
    """
    if not usernames:
        return {}

    with get_db(read_only=True) as conn:
        placeholders = ','.join('?' * len(usernames))
        rows = conn.execute(f"""
            SELECT username, film_slug as slug, rating, watched, watchlisted, liked, scraped_at
            FROM user_films
            WHERE username IN ({placeholders})
        """, usernames).fetchall()

    result = defaultdict(list)
    for row in rows:
        row_dict = dict(row)
        username = row_dict.pop('username')
        result[username].append(row_dict)

    return dict(result)


def load_films_by_attribute(
    attribute_type: str,
    attribute_values: list[str],
    limit_per_value: int = 50
) -> dict[str, list[dict]]:
    """
    Efficiently load films grouped by attribute (genre, director, etc).
    Useful for "find more films by X" queries.
    """
    if not attribute_values:
        return {}

    with get_db(read_only=True) as conn:
        if attribute_type == 'director':
            table, col = 'film_directors', 'director'
        elif attribute_type == 'genre':
            table, col = 'film_genres', 'genre'
        elif attribute_type == 'actor':
            table, col = 'film_cast', 'actor'
        else:
            raise ValueError(f"Unknown attribute type: {attribute_type}")

        placeholders = ','.join('?' * len(attribute_values))

        rows = conn.execute(f"""
            WITH ranked AS (
                SELECT
                    a.{col} as attr_value,
                    f.*,
                    ROW_NUMBER() OVER (PARTITION BY a.{col} ORDER BY f.rating_count DESC) as rn
                FROM {table} a
                JOIN films f ON a.film_slug = f.slug
                WHERE a.{col} IN ({placeholders})
            )
            SELECT * FROM ranked WHERE rn <= ?
        """, [*attribute_values, limit_per_value]).fetchall()

    result = defaultdict(list)
    for row in rows:
        row_dict = dict(row)
        attr_value = row_dict.pop('attr_value')
        row_dict.pop('rn', None)
        result[attr_value].append(row_dict)

    return dict(result)
