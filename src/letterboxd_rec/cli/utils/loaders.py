"""Data loading utilities for CLI commands."""

import argparse
import logging
from collections import defaultdict

from ...database import get_db, load_user_lists

logger = logging.getLogger(__name__)


def _load_all_user_films(conn, username_filter: str | None = None) -> dict[str, list[dict]]:
    """
    Load all user films in a single query with optional username filter.

    Args:
        conn: Database connection
        username_filter: If provided, only load films for this user (useful for single-user operations)

    Returns:
        Dict mapping username -> list of film interaction dicts.
    """
    all_user_films = defaultdict(list)

    if username_filter:
        # Optimized single-user query
        rows = conn.execute("""
            SELECT username, film_slug as slug, rating, watched, watchlisted, liked
            FROM user_films
            WHERE username = ?
        """, (username_filter,)).fetchall()
    else:
        # Load all users (for collaborative filtering)
        rows = conn.execute("""
            SELECT username, film_slug as slug, rating, watched, watchlisted, liked
            FROM user_films
        """).fetchall()

    for row in rows:
        row_dict = dict(row)
        username = row_dict.pop('username')
        all_user_films[username].append(row_dict)

    return dict(all_user_films)


def _load_films_with_filters(
    conn,
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = None
) -> dict[str, dict]:
    """
    Load films from database with optional SQL-side filtering.

    This reduces memory usage and improves performance by filtering at the database level
    rather than loading all films into memory.

    Args:
        conn: Database connection
        min_year: Minimum release year filter
        max_year: Maximum release year filter
        min_rating: Minimum average rating filter

    Returns:
        Dict mapping slug -> film dict
    """
    where_clauses = []
    params = []

    if min_year is not None:
        where_clauses.append("year >= ?")
        params.append(min_year)

    if max_year is not None:
        where_clauses.append("year <= ?")
        params.append(max_year)

    if min_rating is not None:
        where_clauses.append("avg_rating >= ?")
        params.append(min_rating)

    query = "SELECT * FROM films"
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    rows = conn.execute(query, params).fetchall()
    return {r['slug']: dict(r) for r in rows}


def _load_recommendation_data(
    conn,
    username: str,
    strategy: str,
    args: argparse.Namespace,
) -> tuple[list[dict], dict[str, dict], dict[str, list[dict]] | None, list[dict] | None]:
    """
    Load all data needed for the selected recommendation strategy while
    avoiding duplicate queries.
    """
    all_user_films: dict[str, list[dict]] | None = None

    if strategy in ('hybrid', 'collaborative', 'svd'):
        all_user_films = _load_all_user_films(conn)
        user_films = all_user_films.get(username, [])
    else:
        user_films = [dict(r) for r in conn.execute("""
            SELECT film_slug as slug, rating, watched, watchlisted, liked, scraped_at
            FROM user_films WHERE username = ?
        """, (username,))]

    all_films_filtered = _load_films_with_filters(
        conn,
        min_year=args.min_year,
        max_year=args.max_year,
        min_rating=args.min_rating
    )

    if strategy in ('hybrid', 'collaborative', 'graph', 'svd'):
        all_films = {r['slug']: dict(r) for r in conn.execute("SELECT * FROM films")}
    else:
        all_films = all_films_filtered

    user_lists = load_user_lists(username) if strategy == 'metadata' else None

    return user_films, all_films, all_user_films, user_lists


def _warn_missing_metadata(user_films: list[dict], all_films: dict[str, dict], username: str) -> None:
    """Emit a warning when the user's history references films without metadata."""
    seen_slugs = {f['slug'] for f in user_films}
    missing_slugs = seen_slugs - set(all_films.keys())
    if missing_slugs:
        logger.warning(
            f"Missing metadata for {len(missing_slugs)} films in {username}'s history. "
            "Consider running 'scrape' again."
        )
