import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta

from .database import init_db, get_db, load_json, load_user_lists
from .scraper import LetterboxdScraper
from .recommender import MetadataRecommender, CollaborativeRecommender, Recommendation
from .profile import build_profile
from tqdm import tqdm
import asyncio

logger = logging.getLogger(__name__)


def _validate_slug(slug: str) -> str:
    """
    Sanitize a film slug to prevent injection or invalid characters.
    Returns lowercased alphanumeric + hyphens only.
    """
    import re
    # Remove invalid characters, keep only alphanumeric and hyphens
    sanitized = re.sub(r'[^a-z0-9-]', '', slug.lower())
    if sanitized != slug.lower().replace(' ', '-'):
        logger.warning(f"Slug '{slug}' sanitized to '{sanitized}'")
    return sanitized


def _validate_username(username: str) -> str:
    """
    Sanitize a Letterboxd username.
    Returns lowercased alphanumeric + underscores/hyphens only.
    """
    import re
    sanitized = re.sub(r'[^a-z0-9_-]', '', username.lower())
    if sanitized != username.lower():
        logger.warning(f"Username '{username}' sanitized to '{sanitized}'")
    return sanitized


def _load_all_user_films(conn) -> dict[str, list[dict]]:
    """
    Load all user films in a single query.
    Returns dict mapping username -> list of film interaction dicts.
    """
    all_user_films = defaultdict(list)
    rows = conn.execute("""
        SELECT username, film_slug as slug, rating, watched, watchlisted, liked
        FROM user_films
    """).fetchall()
    
    for row in rows:
        row_dict = dict(row)
        username = row_dict.pop('username')
        all_user_films[username].append(row_dict)
    
    return dict(all_user_films)


def _scrape_film_metadata(scraper, slugs, max_per_batch=100, use_async=True):
    """Helper to scrape film metadata for a list of slugs."""
    if not slugs:
        return
    
    limited_slugs = slugs[:max_per_batch]
    
    if use_async and len(limited_slugs) > 10:
        from .scraper import AsyncLetterboxdScraper
        logger.info(f"Fetching {len(limited_slugs)} films (async)...")
        async_scraper = AsyncLetterboxdScraper(delay=0.2, max_concurrent=5)
        metadata_list = asyncio.run(async_scraper.scrape_films_batch(limited_slugs))
    else:
        metadata_list = []
        for slug in tqdm(limited_slugs, desc="Metadata"):
            meta = scraper.scrape_film(slug)
            if meta:
                metadata_list.append(meta)
    
    # Batch insert all metadata
    if metadata_list:
        with get_db() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO films 
                (slug, title, year, directors, genres, cast, themes, runtime, avg_rating, rating_count,
                 countries, languages, writers, cinematographers, composers)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [(
                m.slug, m.title, m.year,
                json.dumps(m.directors), json.dumps(m.genres),
                json.dumps(m.cast), json.dumps(m.themes),
                m.runtime, m.avg_rating, m.rating_count,
                json.dumps(m.countries), json.dumps(m.languages),
                json.dumps(m.writers), json.dumps(m.cinematographers),
                json.dumps(m.composers)
            ) for m in metadata_list])
            
            # Populate normalized tables for fast queries
            from .database import populate_normalized_tables
            for m in metadata_list:
                populate_normalized_tables(conn, m)
        
        logger.info(f"Saved {len(metadata_list)} films")


def cmd_scrape(args):
    """Scrape a user's Letterboxd data."""
    init_db()
    
    # Validate and sanitize username
    username = _validate_username(args.username)
    
    scraper = LetterboxdScraper(delay=1.0)
    
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import database from JSON")
    import_parser.add_argument("file", help="Input JSON file path")
    import_parser.set_defaults(func=cmd_import)
    
    # Profile command
    profile_parser = subparsers.add_parser("profile", help="Show user's preference profile")
    profile_parser.add_argument("username", help="Letterboxd username")
    profile_parser.set_defaults(func=cmd_profile)
    
    # Similar command
    similar_parser = subparsers.add_parser("similar", help="Find films similar to a specific film")
    similar_parser.add_argument("slug", help="Film slug (e.g., 'the-matrix')")
    similar_parser.add_argument("--limit", type=int, default=10, help="Number of similar films")
    similar_parser.set_defaults(func=cmd_similar)
    
    # Triage command
    triage_parser = subparsers.add_parser("triage", help="Rank watchlist by predicted enjoyment")
    triage_parser.add_argument("username", help="Letterboxd username")
    triage_parser.add_argument("--limit", type=int, default=20, help="Number of films to show")
    triage_parser.set_defaults(func=cmd_triage)
    
    # Gaps command
    gaps_parser = subparsers.add_parser("gaps", help="Find essential missing films from favorite directors")
    gaps_parser.add_argument("username", help="Letterboxd username")
    gaps_parser.add_argument("--min-score", type=float, default=2.0, help="Minimum director affinity score")
    gaps_parser.add_argument("--limit", type=int, default=3, help="Max films per director")
    gaps_parser.set_defaults(func=cmd_gaps)
    
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    args.func(args)