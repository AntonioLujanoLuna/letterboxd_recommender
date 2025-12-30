"""Data management CLI commands (export, import, stats, refresh)."""

import argparse
import json
import logging
from datetime import datetime

from ...database import (
    init_db,
    get_db,
    load_json,
    run_maintenance,
    compute_and_store_idf,
    populate_normalized_tables_batch,
)
from ...config import (
    EXPORT_CHUNK_SIZE,
    IMPORT_CHUNK_SIZE,
    RUN_VACUUM_ANALYZE_DEFAULT,
)
from ...scraper import LetterboxdScraper
from ..utils.scrapers import _scrape_film_metadata

logger = logging.getLogger(__name__)


def cmd_stats(args: argparse.Namespace) -> None:
    """Show database statistics."""
    with get_db() as conn:
        user_count = conn.execute("SELECT COUNT(DISTINCT username) FROM user_films").fetchone()[0]
        film_count = conn.execute("SELECT COUNT(*) FROM films").fetchone()[0]
        interaction_count = conn.execute("SELECT COUNT(*) FROM user_films").fetchone()[0]
        rated_count = conn.execute("SELECT COUNT(*) FROM user_films WHERE rating IS NOT NULL").fetchone()[0]

        logger.info(f"\nDatabase Statistics:")
        logger.info(f"  Users: {user_count}")
        logger.info(f"  Films: {film_count}")
        logger.info(f"  Total interactions: {interaction_count}")
        logger.info(f"  Rated interactions: {rated_count}")

        if user_count > 0:
            top_users = conn.execute("""
                SELECT username, COUNT(*) as film_count
                FROM user_films
                GROUP BY username
                ORDER BY film_count DESC
                LIMIT 5
            """).fetchall()

            logger.info(f"\nTop users by film count:")
            for user, count in top_users:
                logger.info(f"  {user}: {count} films")

        verbose = getattr(args, 'verbose', False)
        if verbose:
            missing_metadata = conn.execute("""
                SELECT COUNT(DISTINCT uf.film_slug)
                FROM user_films uf
                LEFT JOIN films f ON uf.film_slug = f.slug
                WHERE f.slug IS NULL
            """).fetchone()[0]

            logger.info(f"\n  Films without metadata: {missing_metadata}")

            oldest = conn.execute("""
                SELECT username, MIN(scraped_at) as oldest_scrape
                FROM user_films
                GROUP BY username
                ORDER BY oldest_scrape
                LIMIT 5
            """).fetchall()

            if oldest:
                logger.info(f"\nOldest scraped users:")
                for user, scrape_time in oldest:
                    logger.info(f"  {user}: {scrape_time}")

            film_genres = conn.execute("SELECT genres FROM films WHERE genres IS NOT NULL").fetchall()
            from collections import Counter
            genre_counts = Counter()
            for (genres_json,) in film_genres:
                genres = load_json(genres_json)
                for g in genres:
                    genre_counts[g] += 1

            if genre_counts:
                logger.info(f"\nTop genres in database:")
                for genre, count in genre_counts.most_common(10):
                    logger.info(f"  {genre}: {count} films")


def cmd_export(args: argparse.Namespace) -> None:
    """Export database to JSON file."""
    def _stream_rows(conn, query: str):
        cursor = conn.execute(query)
        while True:
            chunk = cursor.fetchmany(EXPORT_CHUNK_SIZE)
            if not chunk:
                break
            for row in chunk:
                yield dict(row)

    with get_db(read_only=True) as conn, open(args.file, 'w') as f:
        f.write('{"user_films":[')
        first = True
        uf_count = 0
        for row in _stream_rows(conn, "SELECT * FROM user_films"):
            if not first:
                f.write(',')
            json.dump(row, f)
            first = False
            uf_count += 1
        f.write('],"films":[')
        first = True
        film_count = 0
        for row in _stream_rows(conn, "SELECT * FROM films"):
            if not first:
                f.write(',')
            json.dump(row, f)
            first = False
            film_count += 1
        f.write('], "exported_at": "%s"}' % datetime.now().isoformat())

    logger.info(f"Exported {uf_count} user interactions and {film_count} films to {args.file}")


def cmd_import(args: argparse.Namespace) -> None:
    """Import database from JSON file."""
    with open(args.file, 'r') as f:
        data = json.load(f)

    init_db()

    def _batched(items, size=IMPORT_CHUNK_SIZE):
        for i in range(0, len(items), size):
            yield items[i:i+size]

    with get_db() as conn:
        if 'films' in data:
            for chunk in _batched(data['films']):
                conn.executemany("""
                    INSERT OR REPLACE INTO films
                    (slug, title, year, directors, genres, cast, themes, runtime, avg_rating, rating_count,
                     countries, languages, writers, cinematographers, composers)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [(
                    film['slug'], film.get('title'), film.get('year'),
                    film.get('directors'), film.get('genres'),
                    film.get('cast'), film.get('themes'),
                    film.get('runtime'), film.get('avg_rating'), film.get('rating_count'),
                    film.get('countries'), film.get('languages'),
                    film.get('writers'), film.get('cinematographers'), film.get('composers')
                ) for film in chunk])
                # Keep normalized tables in sync with imported films
                populate_normalized_tables_batch(conn, chunk)
            logger.info(f"Imported {len(data['films'])} films")

        if 'user_films' in data:
            for chunk in _batched(data['user_films']):
                conn.executemany("""
                    INSERT OR REPLACE INTO user_films
                    (username, film_slug, rating, watched, watchlisted, liked, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [(
                    uf['username'], uf['film_slug'], uf.get('rating'),
                    uf.get('watched'), uf.get('watchlisted'), uf.get('liked'),
                    uf.get('scraped_at')
                ) for uf in chunk])
            logger.info(f"Imported {len(data['user_films'])} user interactions")

    # Recompute IDF to clear stale attribute weights after import
    with get_db() as conn:
        conn.execute("DELETE FROM attribute_idf")
    compute_and_store_idf()

    if getattr(args, "maintenance", RUN_VACUUM_ANALYZE_DEFAULT):
        run_maintenance(vacuum=True, analyze=True)

    logger.info(f"Import completed from {args.file}")


def cmd_refresh_metadata(args: argparse.Namespace) -> None:
    """Refresh missing or low-quality film metadata."""
    init_db()
    with get_db(read_only=True) as conn:
        rows = conn.execute(
            """
            SELECT slug FROM films
            WHERE avg_rating IS NULL
               OR rating_count IS NULL
               OR rating_count < ?
               OR json_array_length(genres) IS NULL
            UNION
            SELECT uf.film_slug AS slug
            FROM user_films uf
            LEFT JOIN films f ON uf.film_slug = f.slug
            WHERE f.slug IS NULL
            LIMIT ?
            """,
            (args.min_rating_count, args.limit),
        ).fetchall()

    slugs = [r['slug'] for r in rows]
    if not slugs:
        logger.info("No films need refreshing.")
        return

    scraper = LetterboxdScraper(delay=1.0)
    try:
        _scrape_film_metadata(scraper, slugs, max_per_batch=args.batch, use_async=True)
    finally:
        scraper.close()

    if getattr(args, "maintenance", False):
        run_maintenance(vacuum=False, analyze=True)
