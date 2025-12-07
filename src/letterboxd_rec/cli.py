import argparse
import json
import logging
import atexit
import signal
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

from .database import (
    init_db, get_db, load_json, load_user_lists, parse_timestamp_naive,
    get_discovery_source, update_discovery_source, add_pending_users,
    get_pending_users, remove_pending_user, get_pending_queue_stats,
    compute_and_store_idf, populate_normalized_tables_batch, close_pool, run_maintenance,
    create_scrape_session, update_session_progress, complete_session, get_session_history,
    load_films_by_attribute, update_idf_incremental,
)
from .config import (
    DISCOVERY_PRIORITY_MAP,
    SVD_CACHE_PATH,
    RUN_VACUUM_ANALYZE_DEFAULT,
    EXPORT_CHUNK_SIZE,
    IMPORT_CHUNK_SIZE,
    PENDING_STALE_DAYS,
    DEFAULT_MAX_PER_BATCH,
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_MAX_CONCURRENT_USERS,
    DEFAULT_ASYNC_DELAY,
    NOTIFICATION_WEBHOOK_URL,
    NOTIFICATION_INTERVAL,
)
from .scraper import LetterboxdScraper, AsyncLetterboxdScraper
from .recommender import MetadataRecommender, CollaborativeRecommender, Recommendation, _fuse_normalized
from .matrix_factorization import SVDRecommender
from .profile import build_profile, UserProfile
from .group_recommender import recommend_for_group
from tqdm import tqdm
import asyncio

logger = logging.getLogger(__name__)

# Register cleanup on exit
atexit.register(close_pool)


def send_notification(message: str) -> None:
    """Send a notification to a configured webhook (Discord/Slack-style)."""
    if not NOTIFICATION_WEBHOOK_URL:
        return

    try:
        import httpx

        httpx.post(
            NOTIFICATION_WEBHOOK_URL,
            json={"content": message},
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - best-effort notifications
        logger.warning(f"Failed to send notification: {exc}")


def _validate_slug(slug: str) -> str:
    """
    Validate a film slug to prevent injection or invalid characters.
    Raises ValueError if slug contains invalid characters.
    Returns lowercased valid slug.
    """
    import re

    cleaned = slug.strip().lower()

    prefix = ""
    core = cleaned
    if core.startswith("film:"):
        prefix = "film:"
        core = core.split(":", 1)[1]

    if not core or not re.match(r'^[a-z0-9-]+$', core):
        raise ValueError(f"Invalid slug: {slug}")

    return prefix + core


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


def _parse_weights(weights: list[str] | None) -> dict[str, float]:
    """
    Parse CLI weights arguments in the form user:weight into a dict.
    Invalid entries are ignored with a warning.
    """
    if not weights:
        return {}

    parsed: dict[str, float] = {}
    for entry in weights:
        if ":" not in entry:
            logger.warning("Ignoring weight '%s' (expected user:weight)", entry)
            continue
        user_part, weight_part = entry.split(":", 1)
        try:
            weight_value = float(weight_part)
        except ValueError:
            logger.warning("Ignoring weight '%s' (invalid number)", entry)
            continue
        parsed[_validate_username(user_part)] = weight_value

    return parsed


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


def _scrape_film_metadata(
    scraper: 'LetterboxdScraper',
    slugs: list[str],
    max_per_batch: int = 100,
    use_async: bool = True,
    async_scraper: 'AsyncLetterboxdScraper | None' = None,
) -> None:
    """Helper to scrape film metadata for a list of slugs."""
    if not slugs:
        return

    limited_slugs = slugs[:max_per_batch]

    if use_async and len(limited_slugs) > 10:
        from .scraper import AsyncLetterboxdScraper
        logger.info(f"Fetching {len(limited_slugs)} films (async)...")

        async def scrape_batch(shared: AsyncLetterboxdScraper | None):
            if shared:
                return await shared.scrape_films_batch(limited_slugs)
            async with AsyncLetterboxdScraper(delay=0.2, max_concurrent=5) as temp_async:
                return await temp_async.scrape_films_batch(limited_slugs)

        metadata_list = asyncio.run(scrape_batch(async_scraper))
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

            # Populate normalized tables for fast queries (batch operation)
            populate_normalized_tables_batch(conn, metadata_list)

        logger.info(f"Saved {len(metadata_list)} films")

        # Incrementally update IDF scores for new films
        new_slugs = [m.slug for m in metadata_list]
        update_idf_incremental(new_slugs)


async def _scrape_film_metadata_async(
    async_scraper: 'AsyncLetterboxdScraper',
    slugs: list[str],
    max_per_batch: int = 100,
) -> None:
    """
    Async helper to scrape film metadata using a shared AsyncLetterboxdScraper.
    Mirrors _scrape_film_metadata but avoids blocking the event loop.
    """
    if not slugs:
        return

    limited_slugs = slugs[:max_per_batch]
    metadata_list = await async_scraper.scrape_films_batch(limited_slugs)

    if not metadata_list:
        return

    def _persist():
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
            populate_normalized_tables_batch(conn, metadata_list)

        # Incrementally update IDF scores
        new_slugs = [m.slug for m in metadata_list]
        update_idf_incremental(new_slugs)

    await asyncio.to_thread(_persist)
    logger.info(f"Saved {len(metadata_list)} films")


async def _scrape_users_parallel(
    usernames: list[str],
    args: argparse.Namespace,
) -> None:
    """
    Parallel user scraping using the AsyncLetterboxdScraper.

    Mirrors the scrape-daemon flow but scoped to a fixed list of users discovered
    by the current command.
    """
    if not usernames:
        return

    # Guard against accidental duplicates in the pending queue
    usernames = list(dict.fromkeys(usernames))

    # Load known film slugs once (threaded to keep event loop free)
    def _load_known_slugs():
        with get_db(read_only=True) as conn:
            return {r['slug'] for r in conn.execute("SELECT slug FROM films")}

    known_film_slugs: set[str] = await asyncio.to_thread(_load_known_slugs)
    known_slugs_lock = asyncio.Lock()
    user_semaphore = asyncio.Semaphore(getattr(args, "parallel_users", 1))

    async def _persist_user_films(username: str, interactions):
        def _persist():
            with get_db() as conn:
                scraped_at = datetime.now().isoformat()
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO user_films
                    (username, film_slug, rating, watched, watchlisted, liked, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (username, i.film_slug, i.rating, i.watched, i.watchlisted, i.liked, scraped_at)
                        for i in interactions
                    ],
                )

        await asyncio.to_thread(_persist)

    async with AsyncLetterboxdScraper(
        delay=getattr(args, "async_delay", DEFAULT_ASYNC_DELAY),
        max_concurrent=getattr(args, "max_concurrent_requests", DEFAULT_MAX_CONCURRENT),
    ) as async_scraper:

        async def _process(username: str):
            async with user_semaphore:
                try:
                    interactions = await async_scraper.scrape_user(username)

                    if interactions:
                        await _persist_user_films(username, interactions)

                        async with known_slugs_lock:
                            new_slugs = [i.film_slug for i in interactions if i.film_slug not in known_film_slugs]
                            known_film_slugs.update(new_slugs)

                        if new_slugs:
                            await _scrape_film_metadata_async(
                                async_scraper,
                                new_slugs,
                                max_per_batch=getattr(args, "batch", DEFAULT_MAX_PER_BATCH),
                            )

                        logger.info(f"{username}: {len(interactions)} films (parallel)")
                    else:
                        logger.warning(f"No interactions found for {username}")

                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Error scraping {username}: {exc}")
                finally:
                    await asyncio.to_thread(remove_pending_user, username)

        await asyncio.gather(*[asyncio.create_task(_process(u)) for u in usernames])


def cmd_explore(args: argparse.Namespace) -> None:
    """Explore films by attribute (director, genre, actor)."""
    init_db()
    
    results = load_films_by_attribute(
        args.attribute_type,
        [args.value],
        limit_per_value=args.limit
    )
    
    films = results.get(args.value, [])
    if not films:
        logger.info(f"No films found for {args.attribute_type}: {args.value}")
        return
    
    logger.info(f"\nTop {len(films)} films for {args.attribute_type} '{args.value}':")
    logger.info("-" * 50)
    
    for i, film in enumerate(films, 1):
        title = film.get('title', film.get('slug', 'Unknown'))
        year = film.get('year', '?')
        rating = film.get('avg_rating') or 0
        rating_count = film.get('rating_count') or 0
        
        rating_str = f"{rating:.1f}★" if rating else "N/A"
        count_str = f"({rating_count:,} ratings)" if rating_count else ""
        
        logger.info(f"  {i:2}. {title} ({year}) - {rating_str} {count_str}")

def cmd_scrape(args: argparse.Namespace) -> None:
    """Scrape a user's Letterboxd data."""
    init_db()
    
    # Validate and sanitize username
    username = _validate_username(args.username)
    
    scraper = LetterboxdScraper(delay=1.0)
    
    try:
        # Check if refresh is needed
        refresh = getattr(args, 'refresh', None)
        if refresh:
            with get_db() as conn:
                result = conn.execute("""
                    SELECT MAX(scraped_at) as last_scrape
                    FROM user_films
                    WHERE username = ?
                """, (username,)).fetchone()
                
                if result and result['last_scrape']:
                    last_scrape = parse_timestamp_naive(result['last_scrape'])
                    age_days = (datetime.now() - last_scrape).days

                    if age_days < args.refresh:
                        logger.info(f"  Skipping {username} (last scraped {age_days} days ago, refresh threshold: {args.refresh} days)")
                        return
                    else:
                        logger.info(f"  Refreshing {username} (last scraped {age_days} days ago)")

        # Check for incremental mode
        incremental = getattr(args, 'incremental', False)
        existing_slugs = set()

        if incremental:
            with get_db() as conn:
                existing_slugs = {
                    r['film_slug'] for r in conn.execute("""
                        SELECT film_slug FROM user_films WHERE username = ?
                    """, (username,))
                }
                logger.info(f"Incremental mode: found {len(existing_slugs)} existing films for {username}")

        interactions = scraper.scrape_user(username, existing_slugs=existing_slugs, stop_on_existing=incremental)

        # Batch insert user films (get_db() context manager handles transaction)
        with get_db() as conn:
            scraped_at = datetime.now().isoformat()
            conn.executemany("""
                INSERT OR REPLACE INTO user_films
                (username, film_slug, rating, watched, watchlisted, liked, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [(username, i.film_slug, i.rating, i.watched, i.watchlisted, i.liked, scraped_at)
                  for i in interactions])

            existing = {r['slug'] for r in conn.execute("SELECT slug FROM films")}

        new_slugs = [i.film_slug for i in interactions if i.film_slug not in existing]
        
        # Scrape and batch insert film metadata
        logger.info("\nFetching film metadata...")
        _scrape_film_metadata(scraper, new_slugs)
        
        # Scrape user lists if enabled
        if args.include_lists:
            logger.info(f"\nScraping {username}'s lists...")

            # Get profile favorites (4-film showcase)
            favorites = scraper.scrape_favorites(username)
            if favorites:
                logger.info(f"  Found {len(favorites)} profile favorites")
                with get_db() as conn:
                    for slug in favorites:
                        conn.execute("""
                            INSERT OR REPLACE INTO user_lists
                            (username, list_slug, list_name, is_ranked, is_favorites, position, film_slug, scraped_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            username, "profile-favorites", "Profile Favorites",
                            0, 1, None, slug, datetime.now().isoformat()
                        ))
            
            # Get all user lists
            lists = scraper.scrape_user_lists(username, limit=args.max_lists)

            # Scrape films from each list
            for list_info in lists:
                list_slug = list_info['list_slug']
                list_name = list_info['list_name']
                is_ranked = list_info['is_ranked']

                # Detect favorites
                is_favorites = "favorite" in list_name.lower() or list_slug == "favorites"

                logger.info(f"  Scraping list: {list_name}...")
                films = scraper.scrape_list_films(username, list_slug)
                
                if not films:
                    logger.info(f"    (empty list)")
                    continue
                
                # Save to database
                with get_db() as conn:
                    for film in films:
                        conn.execute("""
                            INSERT OR REPLACE INTO user_lists
                            (username, list_slug, list_name, is_ranked, is_favorites, position, film_slug, scraped_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            username, list_slug, list_name,
                            is_ranked, is_favorites, film.get('position'),
                            film['film_slug'], datetime.now().isoformat()
                        ))
                
                logger.info(f"    {len(films)} films")

        logger.info(f"\nDone! {len(interactions)} films for {username}")
        
    finally:
        scraper.close()


def cmd_discover(args: argparse.Namespace) -> None:
    """Discover and scrape other users with caching and pending queue support."""
    init_db()
    scraper = LetterboxdScraper(delay=1.0)

    try:
        # Check if we're in continue mode (drain pending queue only)
        continue_mode = getattr(args, 'continue_mode', False)
        source_refresh_days = getattr(args, 'source_refresh_days', 7)

        if continue_mode:
            # Drain pending queue only
            queue_stats = get_pending_queue_stats()
            logger.info(f"\nPending queue stats:")
            logger.info(f"  Total pending users: {queue_stats['total']}")
            if queue_stats['breakdown']:
                logger.info(f"  By source type:")
                for source_type, count in queue_stats['breakdown'].items():
                    logger.info(f"    {source_type}: {count}")

            if queue_stats['total'] == 0:
                logger.info("\nNo pending users to scrape!")
                return

            pending = get_pending_users(limit=args.limit)
            usernames_to_scrape = [p['username'] for p in pending]
            logger.info(f"\nProcessing {len(usernames_to_scrape)} users from pending queue...")

        else:
            # Normal discovery mode with caching
            source = args.source

            if not source:
                logger.error("Error: source is required unless using --continue")
                return

            # Determine source_id based on source type
            if source in ('following', 'followers'):
                if not args.username:
                    logger.error(f"--username is required for '{source}' source")
                    return
                source_id = _validate_username(args.username)
            elif source in ('film', 'film_reviews'):
                if not args.film_slug:
                    logger.error(f"--film-slug is required for '{source}' source")
                    return
                source_id = _validate_slug(args.film_slug)
            elif source == 'popular':
                source_id = 'members'  # Popular members endpoint
            else:
                logger.error(f"Unknown source: {source}")
                return

            # Check for cached discovery source
            cached_source = get_discovery_source(source, source_id)
            start_page = 1

            if cached_source:
                scraped_at = parse_timestamp_naive(cached_source['scraped_at'])
                age_days = (datetime.now() - scraped_at).days

                if age_days < source_refresh_days:
                    # Resume from last page
                    start_page = cached_source['last_page_scraped'] + 1
                    logger.info(f"Resuming {source}:{source_id} from page {start_page} (last scraped {age_days} days ago)")
                else:
                    logger.info(f"Re-crawling {source}:{source_id} from page 1 (stale: {age_days} days > {source_refresh_days} days)")

            # Discover users
            logger.info(f"Discovering users from {source}:{source_id}...")
            all_discovered = []
            page = start_page
            priority = DISCOVERY_PRIORITY_MAP.get(source, 50)
            min_films = getattr(args, 'min_films', 50)

            # Calculate how many to discover (more than limit to account for duplicates)
            discover_limit = args.limit * 3

            # Apply activity pre-filtering
            filtered_count = 0
            activity_checked = 0

            while len(all_discovered) < discover_limit:
                # Get page of users based on source
                if source == 'following':
                    usernames = scraper.scrape_following(source_id, limit=page * 50)
                    usernames = usernames[(page-1)*50:page*50] if len(usernames) > (page-1)*50 else []
                elif source == 'followers':
                    usernames = scraper.scrape_followers(source_id, limit=page * 50)
                    usernames = usernames[(page-1)*50:page*50] if len(usernames) > (page-1)*50 else []
                elif source == 'popular':
                    usernames = scraper.scrape_popular_members(limit=page * 50)
                    usernames = usernames[(page-1)*50:page*50] if len(usernames) > (page-1)*50 else []
                elif source == 'film':
                    usernames = scraper.scrape_film_fans(source_id, limit=page * 50)
                    usernames = usernames[(page-1)*50:page*50] if len(usernames) > (page-1)*50 else []
                elif source == 'film_reviews':
                    # Get reviewers (returns list of dicts)
                    reviewers = scraper.scrape_film_reviewers(source_id, limit=page * 50)
                    reviewers = reviewers[(page-1)*50:page*50] if len(reviewers) > (page-1)*50 else []
                    usernames = [r['username'] for r in reviewers]
                else:
                    break

                if not usernames:
                    break

                # Apply activity pre-filtering for all sources
                filtered_usernames = []
                for username in usernames:
                    activity = scraper.check_user_activity(username)
                    activity_checked += 1

                    if not activity:
                        logger.debug(f"Skipping {username} (profile not accessible)")
                        filtered_count += 1
                        continue

                    # Filter by minimum film count
                    if activity['film_count'] < min_films:
                        logger.debug(f"Skipping {username} (only {activity['film_count']} films < {min_films})")
                        filtered_count += 1
                        continue

                    # Require ratings (they actually rate, not just log)
                    if not activity['has_ratings']:
                        logger.debug(f"Skipping {username} (no ratings)")
                        filtered_count += 1
                        continue

                    filtered_usernames.append(username)

                all_discovered.extend(filtered_usernames)
                page += 1

                logger.info(f"  Page {page-1}: found {len(usernames)} users, {len(filtered_usernames)} passed filters (total: {len(all_discovered)})")

            logger.info(f"\nActivity filtering: {activity_checked} checked, {filtered_count} filtered out, {len(all_discovered)} passed")

            # Add discovered users to pending queue
            new_pending = add_pending_users(all_discovered, source, source_id, priority)
            logger.info(f"Added {new_pending} new users to pending queue")

            # Update discovery source cache
            update_discovery_source(source, source_id, page - 1, len(all_discovered))

            # Queue-only mode: stop after enqueuing
            if getattr(args, 'queue_only', False):
                queue_stats = get_pending_queue_stats()
                logger.info(f"\nQueue now has {queue_stats['total']} pending users")
                logger.info("Run 'scrape-daemon' to process the queue")
                return

            # Now get users to scrape from pending queue
            pending = get_pending_users(limit=args.limit)
            usernames_to_scrape = [p['username'] for p in pending]

            if not usernames_to_scrape:
                logger.info("\nNo new users to scrape from pending queue!")
                return

            logger.info(f"\nScraping {len(usernames_to_scrape)} users from pending queue...")

        # Check for dry-run mode
        dry_run = getattr(args, 'dry_run', False)

        if dry_run:
            logger.info(f"\n[DRY RUN] Would scrape {len(usernames_to_scrape)} users:")
            for i, username in enumerate(usernames_to_scrape, 1):
                logger.info(f"  {i}. {username}")
            logger.info(f"\nTo actually scrape these users, run without --dry-run flag")
            return

        # Scrape each user
        if getattr(args, "parallel_users", 1) > 1:
            asyncio.run(_scrape_users_parallel(usernames_to_scrape, args))
        else:
            # Defensive dedupe in serial path too
            usernames_to_scrape = list(dict.fromkeys(usernames_to_scrape))
            with get_db() as conn:
                existing_film_slugs = {r['slug'] for r in conn.execute("SELECT slug FROM films")}

            for username in tqdm(usernames_to_scrape, desc="Users"):
                try:
                    # Use smart scraping when we have activity info
                    activity = scraper.check_user_activity(username)
                    if activity and activity.get('film_count'):
                        interactions = scraper.scrape_user_smart(
                            username, 
                            known_film_count=activity['film_count']
                        )
                    else:
                        interactions = scraper.scrape_user(username)

                    if not interactions:
                        logger.warning(f"No interactions found for {username}")
                        remove_pending_user(username)
                        continue

                    with get_db() as conn:
                        scraped_at = datetime.now().isoformat()
                        conn.executemany("""
                            INSERT OR REPLACE INTO user_films
                            (username, film_slug, rating, watched, watchlisted, liked, scraped_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, [(username, i.film_slug, i.rating, i.watched, i.watchlisted, i.liked, scraped_at)
                              for i in interactions])

                        existing_film_slugs.update(
                            r['slug'] for r in conn.execute("SELECT slug FROM films")
                        )

                    # Remove from pending queue after successful scrape
                    remove_pending_user(username)

                    # Scrape missing film metadata
                    new_slugs = [i.film_slug for i in interactions if i.film_slug not in existing_film_slugs]
                    _scrape_film_metadata(scraper, new_slugs, max_per_batch=args.batch)
                    existing_film_slugs.update(new_slugs)

                except Exception as e:
                    logger.error(f"Error scraping {username}: {e}")

        logger.info(f"\nDone! Scraped {len(usernames_to_scrape)} users.")

        # Show remaining pending queue stats
        queue_stats = get_pending_queue_stats()
        if queue_stats['total'] > 0:
            logger.info(f"\nRemaining in pending queue: {queue_stats['total']} users")
            logger.info("Run with --continue to scrape more from the queue")

        if getattr(args, "maintenance", False):
            run_maintenance(vacuum=True, analyze=True)

    finally:
        scraper.close()


async def _cmd_scrape_daemon_async(args: argparse.Namespace) -> None:
    """Async daemon: drain pending queue with shared client + coordinated rate limiting."""
    init_db()

    # Track session progress and allow resuming visibility
    session_id = create_scrape_session()
    scraped_count = 0
    films_added = 0
    session_start = datetime.now()

    async def _load_known_slugs() -> set[str]:
        def _load():
            with get_db(read_only=True) as conn:
                return {r['slug'] for r in conn.execute("SELECT slug FROM films")}
        return await asyncio.to_thread(_load)

    known_film_slugs = await _load_known_slugs()
    user_semaphore = asyncio.Semaphore(getattr(args, "max_concurrent_users", 1))
    shutdown_requested = False

    async def _persist_user_films(username: str, interactions):
        def _persist():
            with get_db() as conn:
                scraped_at = datetime.now().isoformat()
                conn.executemany("""
                    INSERT OR REPLACE INTO user_films
                    (username, film_slug, rating, watched, watchlisted, liked, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [
                    (username, i.film_slug, i.rating, i.watched, i.watchlisted, i.liked, scraped_at)
                    for i in interactions
                ])
        await asyncio.to_thread(_persist)

    async def _remove_pending(username: str):
        await asyncio.to_thread(remove_pending_user, username)

    async def _update_session():
        if session_id:
            await asyncio.to_thread(update_session_progress, session_id, scraped_count, films_added)

    async def _notify(scraped_count_local: int):
        if scraped_count_local and NOTIFICATION_INTERVAL and scraped_count_local % NOTIFICATION_INTERVAL == 0:
            queue_remaining = await asyncio.to_thread(lambda: get_pending_queue_stats()['total'])
            send_notification(
                f"Scrape progress: {scraped_count_local} users done, {queue_remaining} remaining"
            )

    async with AsyncLetterboxdScraper(
        delay=getattr(args, "async_delay", getattr(args, "delay", DEFAULT_ASYNC_DELAY)),
        max_concurrent=getattr(args, "max_concurrent_requests", DEFAULT_MAX_CONCURRENT),
    ) as async_scraper:

        async def _process_user(username: str):
            nonlocal films_added, scraped_count, known_film_slugs
            async with user_semaphore:
                try:
                    interactions = await async_scraper.scrape_user(username)

                    if interactions:
                        await _persist_user_films(username, interactions)
                        new_slugs = [i.film_slug for i in interactions if i.film_slug not in known_film_slugs]
                        await _scrape_film_metadata_async(async_scraper, new_slugs, max_per_batch=args.batch)
                        known_film_slugs.update(new_slugs)

                        scraped_count += 1
                        films_added += len(interactions)

                        elapsed_hours = (datetime.now() - session_start).total_seconds() / 3600
                        rate = scraped_count / elapsed_hours if elapsed_hours > 0 else 0
                        logger.info(f"[{scraped_count}] {username}: {len(interactions)} films ({rate:.1f}/hr)")
                    else:
                        logger.warning(f"No interactions found for {username}")

                    await _remove_pending(username)

                    if args.user_delay:
                        await asyncio.sleep(args.user_delay)

                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Error scraping {username}: {exc}")
                    if args.remove_on_error:
                        await _remove_pending(username)

        tasks: set[asyncio.Task] = set()

        try:
            while not shutdown_requested:
                # Refill task set up to max_concurrent_users
                while len(tasks) < getattr(args, "max_concurrent_users", 1):
                    next_batch = await asyncio.to_thread(get_pending_users, 1)
                    if not next_batch:
                        break
                    uname = next_batch[0]['username']
                    # Remove immediately to avoid multiple concurrent claims of the same user
                    await asyncio.to_thread(remove_pending_user, uname)
                    tasks.add(asyncio.create_task(_process_user(uname)))

                if not tasks:
                    if args.wait_for_queue:
                        logger.info(f"Queue empty, waiting {args.wait_seconds}s for new entries...")
                        await asyncio.sleep(args.wait_seconds)
                        continue
                    logger.info("Pending queue empty, stopping.")
                    break

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED, timeout=1)
                tasks = pending

                await _update_session()
                await _notify(scraped_count)

                if args.target and scraped_count >= args.target:
                    logger.info(f"Reached target of {args.target} users.")
                    break

        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = (datetime.now() - session_start).total_seconds()
    status = "interrupted" if shutdown_requested else "completed"
    if session_id:
        complete_session(session_id, status=status)
    logger.info(f"\nSession complete: {scraped_count} users in {elapsed/3600:.1f} hours")


def cmd_scrape_daemon(args: argparse.Namespace) -> None:
    """Entry point wrapper to run the async daemon with asyncio."""
    try:
        asyncio.run(_cmd_scrape_daemon_async(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")


def cmd_queue_status(args: argparse.Namespace) -> None:
    """Show pending queue statistics."""
    init_db()
    stats = get_pending_queue_stats()

    logger.info("\nPending Queue Status")
    logger.info("-" * 30)
    logger.info(f"Total pending users: {stats['total']}")
    if stats['avg_priority'] is not None:
        logger.info(f"Average priority: {stats['avg_priority']:.1f}")

    if stats['breakdown']:
        logger.info("\nBy source:")
        for source_type, count in sorted(stats['breakdown'].items(), key=lambda x: -x[1]):
            logger.info(f"  {source_type}: {count}")

    if stats['total'] > 0:
        est_hours = (stats['total'] * 30) / 3600  # rough estimate: ~30s/user
        logger.info(f"\nEstimated time to drain: {est_hours:.1f} hours")

    if args.verbose:
        pending = get_pending_users(limit=args.limit)
        if pending:
            logger.info("\nNext in queue:")
            for p in pending:
                logger.info(f"  [{p['priority']}] {p['username']} (from {p['discovered_from_type']})")


def cmd_queue_add(args: argparse.Namespace) -> None:
    """Manually add usernames to the pending queue."""
    init_db()

    usernames: list[str] = []
    if args.file:
        path = Path(args.file)
        if not path.exists():
            logger.error(f"File not found: {args.file}")
            return
        usernames = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    else:
        usernames = args.usernames or []

    sanitized = [_validate_username(u) for u in usernames if u]
    if not sanitized:
        logger.error("No usernames provided.")
        return

    added = add_pending_users(sanitized, "manual", "cli", priority=args.priority)
    skipped = len(sanitized) - added
    logger.info(f"Added {added} users to queue ({skipped} already existed)")


def cmd_queue_clear(args: argparse.Namespace) -> None:
    """Clear pending queue (optionally by source type)."""
    init_db()
    with get_db() as conn:
        if args.source:
            count = conn.execute(
                "DELETE FROM pending_users WHERE discovered_from_type = ?",
                (args.source,),
            ).rowcount
        else:
            count = conn.execute("DELETE FROM pending_users").rowcount
    logger.info(f"Removed {count} users from queue")


def cmd_discover_refill(args: argparse.Namespace) -> None:
    """Auto-refill queue from multiple sources when it runs low."""
    init_db()
    stats = get_pending_queue_stats()

    if stats['total'] >= args.min_queue:
        logger.info(f"Queue has {stats['total']} users (>= {args.min_queue}), skipping refill")
        return

    target_add = max(args.target - stats['total'], 0)

    # Default weighted sources
    sources = [
        ("film_reviews", "parasite", 100, 2),
        ("film_reviews", "perfect-blue", 100, 2),
        ("film_reviews", "in-the-mood-for-love", 100, 1),
        ("film_reviews", "mulholland-drive", 100, 1),
        ("popular", "members", 70, 1),
    ]

    if args.sources_file:
        import json as _json
        with open(args.sources_file) as f:
            loaded = _json.load(f)
            loaded_sources = loaded.get("sources", loaded) if isinstance(loaded, dict) else loaded

        normalized_sources = []
        for item in loaded_sources:
            if isinstance(item, dict):
                source_type = item.get("type") or item.get("source_type")
                source_id = item.get("id") or item.get("source_id") or item.get("film_slug") or item.get("film")
                priority = item.get("priority", 50)
                weight = item.get("weight", 1)
                normalized_sources.append((source_type, source_id, priority, weight))
            else:
                normalized_sources.append(tuple(item))
        sources = normalized_sources

    scraper = LetterboxdScraper(delay=1.0)
    total_added = 0

    try:
        for entry in sources:
            try:
                source_type, source_id, priority, weight = entry
            except ValueError:
                logger.warning(f"Invalid source entry {entry}, expected 4 values")
                continue

            if not source_type or not source_id:
                logger.warning(f"Skipping source with missing type/id: {entry}")
                continue

            if total_added >= target_add:
                break

            limit = min(int(50 * weight), target_add - total_added)

            cached = get_discovery_source(source_type, source_id)
            if cached and cached.get('scraped_at'):
                age_days = (datetime.now() - parse_timestamp_naive(cached['scraped_at'])).days
                if age_days < args.source_refresh_days:
                    continue

            logger.info(f"Discovering from {source_type}:{source_id} (limit {limit})...")

            if source_type == "film_reviews":
                users = [r['username'] for r in scraper.scrape_film_reviewers(source_id, limit=limit)]
            elif source_type == "popular":
                users = scraper.scrape_popular_members(limit=limit)
            elif source_type == "followers":
                users = scraper.scrape_followers(source_id, limit=limit)
            else:
                logger.warning(f"Unknown source {source_type}, skipping")
                continue

            filtered = []
            for username in users:
                activity = scraper.check_user_activity(username)
                if activity and activity['film_count'] >= args.min_films and activity['has_ratings']:
                    filtered.append(username)

            added = add_pending_users(filtered, source_type, source_id, priority)
            total_added += added
            logger.info(f"  Added {added} users")

            update_discovery_source(source_type, source_id, 1, len(filtered))

    finally:
        scraper.close()

    logger.info(f"\nRefill complete: added {total_added} users, queue now at {stats['total'] + total_added}")


def cmd_session_history(args: argparse.Namespace) -> None:
    """Show scraping session history."""
    init_db()
    sessions = get_session_history(limit=args.limit)
    if not sessions:
        logger.info("No scraping sessions recorded yet.")
        return

    logger.info("\nRecent Scraping Sessions")
    logger.info("-" * 44)
    for s in sessions:
        started = s['started_at'][:16].replace('T', ' ')
        status = s['status']
        users = s.get('users_scraped') or 0
        films = s.get('films_added') or 0

        if s.get('completed_at'):
            start_dt = parse_timestamp_naive(s['started_at'])
            end_dt = parse_timestamp_naive(s['completed_at'])
            duration = (end_dt - start_dt).total_seconds() / 3600
            duration_str = f"{duration:.1f}h"
        else:
            duration_str = "ongoing"

        logger.info(f"  [{s['id']}] {started} | {status:10} | {users:4} users | {films:5} films | {duration_str}")


def cmd_discover_from_taste(args: argparse.Namespace) -> None:
    """Discover users who reviewed films similar to your taste."""
    init_db()
    username = _validate_username(args.username)

    with get_db(read_only=True) as conn:
        rows = conn.execute("""
            SELECT film_slug FROM user_films
            WHERE username = ? AND rating >= ?
            ORDER BY rating DESC
            LIMIT ?
        """, (username, args.min_rating, args.film_limit)).fetchall()

    if not rows:
        logger.error(f"No films found for {username} with rating >= {args.min_rating}")
        return

    film_slugs = [r['film_slug'] for r in rows]
    logger.info(f"Discovering reviewers from {len(film_slugs)} of {username}'s top films...")

    scraper = LetterboxdScraper(delay=1.0)
    total_added = 0

    try:
        for slug in film_slugs:
            reviewers = scraper.scrape_film_reviewers(slug, limit=args.per_film)
            usernames = [r['username'] for r in reviewers if r.get('has_rating', True)]

            filtered = []
            for user in usernames[:args.per_film]:
                activity = scraper.check_user_activity(user)
                if activity and activity['film_count'] >= args.min_films and activity['has_ratings']:
                    filtered.append(user)

            added = add_pending_users(filtered, "taste_match", slug, priority=90)
            total_added += added
            logger.info(f"  {slug}: +{added} users")

    finally:
        scraper.close()

    logger.info(f"\nAdded {total_added} taste-matched users to queue")


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

    if strategy in ('hybrid', 'collaborative'):
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


def _run_metadata_strategy(
    user_films: list[dict],
    all_films: dict[str, dict],
    args: argparse.Namespace,
    username: str,
    user_lists: list[dict] | None,
    all_user_films: dict[str, list[dict]] | None = None,
) -> list[Recommendation]:
    """Metadata-based recommendation strategy."""
    diversity = getattr(args, 'diversity', False)
    max_per_director = getattr(args, 'max_per_director', 2)
    use_temporal_decay = not getattr(args, 'no_temporal_decay', False)
    lists = user_lists if user_lists is not None else load_user_lists(username)

    profile = build_profile(
        user_films,
        all_films,
        user_lists=lists,
        username=username,
        use_temporal_decay=use_temporal_decay,
        weighting_mode=args.weighting_mode,
    )

    recommender = MetadataRecommender(list(all_films.values()))
    return recommender.recommend(
        user_films,
        n=args.limit,
        min_year=args.min_year,
        max_year=args.max_year,
        genres=args.genres,
        exclude_genres=args.exclude_genres,
        min_rating=args.min_rating,
        diversity=diversity,
        max_per_director=max_per_director,
        username=username,
        user_lists=lists,
        profile=profile,
        weighting_mode=args.weighting_mode,
    )


def _run_collaborative_strategy(
    user_films: list[dict],
    all_films: dict[str, dict],
    args: argparse.Namespace,
    username: str,
    user_lists: list[dict] | None,
    all_user_films: dict[str, list[dict]] | None,
) -> list[Recommendation]:
    """Collaborative filtering strategy."""
    if not all_user_films:
        logger.error("Collaborative strategy requires all user data.")
        return []

    recommender = CollaborativeRecommender(all_user_films, all_films)
    return recommender.recommend(
        username,
        n=args.limit,
        min_year=args.min_year,
        max_year=args.max_year,
        genres=args.genres,
        exclude_genres=args.exclude_genres
    )


def _run_hybrid_strategy(
    user_films: list[dict],
    all_films: dict[str, dict],
    args: argparse.Namespace,
    username: str,
    user_lists: list[dict] | None,
    all_user_films: dict[str, list[dict]] | None,
) -> list[Recommendation]:
    """Hybrid metadata + collaborative strategy."""
    if not all_user_films:
        logger.error("Hybrid strategy requires all user data.")
        return []

    meta_rec = MetadataRecommender(list(all_films.values()))
    collab_rec = CollaborativeRecommender(all_user_films, all_films)

    rated_count = sum(1 for f in user_films if f.get('rating'))
    meta_weight = args.hybrid_meta_weight
    collab_weight = args.hybrid_collab_weight

    if meta_weight is None and collab_weight is None:
        # Heuristic: lean on metadata when user data or neighbors are sparse
        if len(all_user_films) < 10 or rated_count < 20:
            meta_weight = 0.7
            collab_weight = 0.3
        else:
            meta_weight = 0.6
            collab_weight = 0.4
    else:
        # If only one is provided, derive the other to preserve user intent
        if meta_weight is None:
            meta_weight = max(0.0, 1.0 - collab_weight) if collab_weight is not None else 0.6
        if collab_weight is None:
            collab_weight = max(0.0, 1.0 - meta_weight) if meta_weight is not None else 0.4

    meta_recs = meta_rec.recommend(
        user_films,
        n=args.limit * 2,
        min_year=args.min_year,
        max_year=args.max_year,
        genres=args.genres,
        exclude_genres=args.exclude_genres,
        min_rating=args.min_rating,
        username=username,
        weighting_mode=args.weighting_mode,
    )
    collab_recs = collab_rec.recommend(
        username,
        n=args.limit * 2,
        min_year=args.min_year,
        max_year=args.max_year,
        genres=args.genres,
        exclude_genres=args.exclude_genres
    )

    ranked = _fuse_normalized(meta_recs, collab_recs, weight_meta=meta_weight, weight_collab=collab_weight)

    max_per_director = getattr(args, 'max_per_director', 2)
    apply_diversity = bool(getattr(args, 'hybrid_diversity', False))
    director_counts = defaultdict(int)

    recs: list[Recommendation] = []
    for slug, score, reasons in ranked:
        film = all_films.get(slug)
        if not film:
            continue

        directors = load_json(film.get('directors'))
        if apply_diversity and max_per_director and directors:
            if any(director_counts[d] >= max_per_director for d in directors):
                continue

        recs.append(Recommendation(
            slug=slug,
            title=film.get('title', slug),
            year=film.get('year'),
            score=score,
            reasons=list(dict.fromkeys(reasons))[:3]
        ))

        for d in directors:
            director_counts[d] += 1

        if len(recs) >= args.limit:
            break

    return recs


def _run_svd_strategy(
    user_films: list[dict],
    all_films: dict[str, dict],
    args: argparse.Namespace,
    username: str,
    user_lists: list[dict] | None,
    all_user_films: dict[str, list[dict]] | None,
) -> list[Recommendation]:
    """Matrix factorization (SVD) strategy."""
    if not all_user_films:
        logger.error("SVD strategy requires all user data.")
        return []

    if username not in all_user_films:
        logger.error(f"No data for '{username}'.")
        return []

    n_users = len(all_user_films)
    n_ratings = sum(
        1 for films in all_user_films.values()
        for f in films if f.get('rating')
    )

    if n_users < 10 or n_ratings < 100:
        logger.warning(
            f"SVD works best with more data (have {n_users} users, {n_ratings} ratings). "
            "Consider using 'metadata' strategy or running 'discover' to add more users."
        )

    cache_path = SVD_CACHE_PATH
    n_factors = min(50, n_users - 1)
    use_implicit = True
    implicit_weight = 0.3
    fingerprint = SVDRecommender.compute_fingerprint(
        all_user_films,
        hyperparams={
            "n_factors": n_factors,
            "use_implicit": use_implicit,
            "implicit_weight": implicit_weight,
        },
    )
    svd = SVDRecommender.load(cache_path, expected_fingerprint=fingerprint)

    if svd:
        logger.info(f"Using cached SVD model ({n_users} users, {n_ratings} ratings).")
    else:
        logger.info(f"Fitting SVD model on {n_users} users...")
        svd = SVDRecommender(n_factors=n_factors, use_implicit=use_implicit, implicit_weight=implicit_weight)
        try:
            svd.fit(all_user_films)
        except ValueError as e:
            logger.error(f"Unable to fit SVD model: {e}")
            logger.error("Try the 'metadata' strategy or add more rated films.")
            return []

        cache_metadata = {
            "fingerprint": fingerprint,
            "n_users": n_users,
            "n_items": len(svd.item_index) if svd.item_index else 0,
            "n_ratings": n_ratings,
            "hyperparams": {
                "n_factors": n_factors,
                "use_implicit": use_implicit,
                "implicit_weight": implicit_weight,
            },
        }
        svd.save(cache_path, metadata=cache_metadata)

    seen_slugs = {f['slug'] for f in user_films}
    svd_recs = svd.recommend(username, seen_slugs, n=args.limit * 3)

    recs: list[Recommendation] = []
    for slug, predicted_rating in svd_recs:
        if slug not in all_films:
            continue

        film = all_films[slug]

        year = film.get('year')
        if args.min_year and year and year < args.min_year:
            continue
        if args.max_year and year and year > args.max_year:
            continue

        film_genres = load_json(film.get('genres', []))
        if args.genres:
            genres_lower = [g.lower() for g in args.genres]
            if not any(g in film_genres for g in genres_lower):
                continue
        if args.exclude_genres:
            exclude_lower = [g.lower() for g in args.exclude_genres]
            if any(g in film_genres for g in exclude_lower):
                continue

        if args.min_rating and film.get('avg_rating') and film['avg_rating'] < args.min_rating:
            continue

        recs.append(Recommendation(
            slug=slug,
            title=film.get('title', slug),
            year=year,
            score=predicted_rating,
            reasons=[f"Predicted rating: {predicted_rating:.1f}★"]
        ))

        if len(recs) >= args.limit:
            break

    return recs


def _output_recommendations(
    recs: list[Recommendation],
    all_films: dict[str, dict],
    args: argparse.Namespace,
    username: str,
    strategy: str,
    recommender: 'MetadataRecommender | None' = None,
    profile: 'UserProfile | None' = None,
    user_films: list[dict] | None = None,
) -> None:
    """Format and log recommendations in the requested format."""
    recs = recs or []
    output_format = getattr(args, 'format', 'text')
    explain_mode = getattr(args, 'explain', False)
    diversity_report = getattr(args, 'diversity_report', False)
    
    if output_format == 'json':
        output = []
        for r in recs:
            film = all_films.get(r.slug, {})
            rec_data = {
                "title": r.title,
                "year": r.year,
                "slug": r.slug,
                "score": round(r.score, 2),
                "reasons": r.reasons,
                "url": f"https://letterboxd.com/film/{r.slug}/",
                "directors": load_json(film.get('directors', [])),
                "genres": load_json(film.get('genres', [])),
                "cast": load_json(film.get('cast', []))[:5],
                "themes": load_json(film.get('themes', [])),
                "countries": load_json(film.get('countries', [])),
                "avg_rating": film.get('avg_rating'),
                "rating_count": film.get('rating_count')
            }
            
            # Add explanation if requested and available
            if explain_mode and recommender and profile and user_films:
                explanation = recommender.explain_recommendation_detailed(film, profile, user_films)
                rec_data["explanation"] = {
                    "summary": explanation["summary"],
                    "confidence": explanation["confidence"],
                    "discovery_potential": explanation["discovery_potential"],
                    "similar_films_you_liked": explanation["similar_films_you_liked"],
                    "contribution_breakdown": explanation.get("contribution_breakdown", {}),
                }
            
            output.append(rec_data)
        
        # Add diversity metrics if requested
        if diversity_report and recommender:
            metrics = recommender.compute_recommendation_diversity(recs)
            logger.info(json.dumps({"recommendations": output, "diversity": metrics}, indent=2))
        else:
            logger.info(json.dumps(output, indent=2))
            
    elif output_format == 'csv':
        logger.info("Title,Year,URL,Score,Reasons")
        for r in recs:
            reasons = "; ".join(r.reasons).replace('"', '""')
            logger.info(f'"{r.title}",{r.year},https://letterboxd.com/film/{r.slug}/,{r.score:.2f},"{reasons}"')
            
    elif output_format == 'markdown':
        logger.info(f"\n# Top {len(recs)} recommendations for {username} ({strategy})\n")
        for i, r in enumerate(recs, 1):
            logger.info(f"## {i}. [{r.title} ({r.year})](https://letterboxd.com/film/{r.slug}/)")
            logger.info(f"**Score**: {r.score:.1f}  ")
            logger.info(f"**Why**: {', '.join(r.reasons)}\n")
            
    else:  # text format
        logger.info(f"\nTop {len(recs)} recommendations for {username} ({strategy}):")
        
        for i, r in enumerate(recs, 1):
            logger.info(f"{i}. {r.title} ({r.year}) - Score: {r.score:.1f}")
            logger.info(f"   Why: {', '.join(r.reasons)}")
            
            # Detailed explanation if requested
            if explain_mode and recommender and profile and user_films:
                film = all_films.get(r.slug, {})
                explanation = recommender.explain_recommendation_detailed(film, profile, user_films)
                
                logger.info(f"   Confidence: {explanation['confidence']:.0%}")
                logger.info(f"   Discovery: {explanation['discovery_potential']}")
                
                if explanation['similar_films_you_liked']:
                    similar = explanation['similar_films_you_liked'][0]
                    rating_str = f" (you rated {similar['your_rating']}★)" if similar.get('your_rating') else ""
                    logger.info(f"   Because you liked: {similar['title']}{rating_str} - {similar['connection']}")
                
                if explanation.get('counterfactuals'):
                    logger.info(f"   Note: {explanation['counterfactuals'][0]}")
                
                logger.info("")  # Blank line between detailed entries
    
    # Diversity report at the end
    if diversity_report and recommender and output_format != 'json':
        metrics = recommender.compute_recommendation_diversity(recs)
        logger.info(f"\n{'=' * 50}")
        logger.info("Diversity Report:")
        logger.info(f"  Overall diversity: {metrics['diversity_score']:.0%}")
        logger.info(f"  Genres: {metrics['unique_genres']} unique ({metrics['genre_diversity']:.0%} entropy)")
        logger.info(f"  Directors: {metrics['unique_directors']} unique ({metrics['director_diversity']:.0%} entropy)")
        logger.info(f"  Countries: {metrics['unique_countries']} unique ({metrics['country_diversity']:.0%} entropy)")
        if metrics['decade_range']:
            logger.info(f"  Decades: {metrics['decade_range'][0]}s to {metrics['decade_range'][1]}s ({metrics['decade_diversity']:.0%} entropy)")


def cmd_similar_users(args: argparse.Namespace) -> None:
    """Find users with similar taste (influencers and followers)."""
    init_db()
    username = _validate_username(args.username)
    
    with get_db(read_only=True) as conn:
        all_user_films = _load_all_user_films(conn)
        all_films = {r['slug']: dict(r) for r in conn.execute("SELECT * FROM films")}
    
    if username not in all_user_films:
        logger.error(f"No data for '{username}'. Run: python main.py scrape {username}")
        return
    
    recommender = CollaborativeRecommender(all_user_films, all_films)
    influencers, followers = recommender._find_neighbors_asymmetric(username, k=args.limit)
    
    if influencers:
        logger.info(f"\nInfluencers (users whose taste predicts yours):")
        logger.info("-" * 50)
        for user, score in influencers:
            film_count = len(all_user_films.get(user, []))
            logger.info(f"  {user}: {score:.2f} similarity ({film_count} films)")
    else:
        logger.info("\nNo influencers found with sufficient overlap.")
    
    if followers:
        logger.info(f"\nFollowers (users who share your taste):")
        logger.info("-" * 50)
        for user, score in followers:
            film_count = len(all_user_films.get(user, []))
            logger.info(f"  {user}: {score:.2f} similarity ({film_count} films)")
    else:
        logger.info("\nNo followers found with sufficient overlap.")
    
    # Optionally show taste compatibility summary
    if args.verbose and (influencers or followers):
        logger.info(f"\nTaste Network Summary for {username}:")
        logger.info(f"  Total users in database: {len(all_user_films)}")
        logger.info(f"  Influencers found: {len(influencers)}")
        logger.info(f"  Followers found: {len(followers)}")
        
        if influencers:
            top_influencer = influencers[0][0]
            logger.info(f"  Closest influencer: {top_influencer}")
            logger.info(f"  Tip: Check what {top_influencer} has rated highly that you haven't seen!")


def cmd_jam(args: argparse.Namespace) -> None:
    """Generate recommendations for a group watch session."""
    usernames = [_validate_username(u) for u in args.usernames]

    if len(usernames) < 2:
        logger.error("Need at least 2 usernames for group recommendations")
        return

    weights = _parse_weights(getattr(args, "weights", None))

    logger.info(f"\n🎬 Film Jam: Finding movies for {', '.join(usernames)}\n")

    try:
        recs, group_info = recommend_for_group(
            usernames,
            n=args.limit,
            strategy=args.strategy,
            min_year=args.min_year,
            max_year=args.max_year,
            genres=args.genres,
            exclude_divisive=not args.include_divisive,
            weights=weights,
        )
    except ValueError as exc:  # e.g., not enough valid users
        logger.error(str(exc))
        return

    if args.format == "json":
        import json

        output = [
            {
                "title": rec.title,
                "year": rec.year,
                "slug": rec.slug,
                "group_score": round(rec.group_score, 2),
                "user_scores": {k: round(v, 2) for k, v in rec.user_scores.items()},
                "agreement": round(rec.agreement_score, 2),
                "unanimous": rec.is_unanimous,
                "reasons": rec.consensus_reasons,
                "url": f"https://letterboxd.com/film/{rec.slug}/",
            }
            for rec in recs
        ]
        print(json.dumps({"group_info": group_info, "recommendations": output}, indent=2))
        return

    logger.info("Group Dynamics:")
    logger.info(
        f"  Compatibility: {group_info['overall_compatibility']} - "
        f"{group_info['compatibility_label']}"
    )

    if group_info.get("consensus_genres"):
        logger.info(f"  You all like: {', '.join(group_info['consensus_genres'])}")

    if group_info.get("divisive_genres"):
        logger.info(f"  Mixed feelings on: {', '.join(group_info['divisive_genres'])}")

    if group_info.get("best_pair"):
        logger.info(f"  Best match: {group_info['best_pair']}")

    if group_info.get("challenging_pair"):
        logger.info(f"  Most different: {group_info['challenging_pair']}")

    logger.info(f"  💡 {group_info['recommendation']}")

    if group_info.get("shared_watchlist_count", 0) > 0:
        logger.info(
            f"  📋 {group_info['shared_watchlist_count']} films on multiple watchlists"
        )

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Top {len(recs)} Films for Your Group:")
    logger.info(f"{'=' * 60}\n")

    for i, rec in enumerate(recs, 1):
        unanimous = "✓" if rec.is_unanimous else ""
        logger.info(f"{i}. {rec.title} ({rec.year}) {unanimous}")
        logger.info(
            f"   Group Score: {rec.group_score:.1f} | Agreement: {rec.agreement_score:.0%}"
        )

        score_parts = [f"{user}: {score:.1f}" for user, score in rec.user_scores.items()]
        logger.info(f"   Individual: {' | '.join(score_parts)}")

        if rec.consensus_reasons:
            logger.info(f"   Why: {', '.join(rec.consensus_reasons)}")

        for warning in rec.warnings:
            logger.info(f"   {warning}")

        logger.info("")


def cmd_recommend(args: argparse.Namespace) -> None:
    """Generate recommendations."""
    username = _validate_username(args.username)
    strategy = getattr(args, 'strategy', 'metadata')

    with get_db() as conn:
        user_films, all_films, all_user_films, user_lists = _load_recommendation_data(
            conn, username, strategy, args
        )

    if not user_films:
        logger.error(f"No data for '{username}'. Run: python main.py scrape {username}")
        return

    _warn_missing_metadata(user_films, all_films, username)

    strategy_handlers = {
        'metadata': _run_metadata_strategy,
        'collaborative': _run_collaborative_strategy,
        'hybrid': _run_hybrid_strategy,
        'svd': _run_svd_strategy,
    }

    handler = strategy_handlers[strategy]
    recs = handler(user_films, all_films, args, username, user_lists, all_user_films)

    # Build recommender and profile for explanations if needed
    recommender = None
    profile = None
    if getattr(args, 'explain', False) or getattr(args, 'diversity_report', False):
        recommender = MetadataRecommender(list(all_films.values()))
        profile = build_profile(
            user_films,
            all_films,
            user_lists=user_lists,
            username=username,
            weighting_mode=args.weighting_mode,
        )

    _output_recommendations(
        recs, all_films, args, username, strategy,
        recommender=recommender,
        profile=profile,
        user_films=user_films,
    )


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


def cmd_profile(args: argparse.Namespace) -> None:
    """Show user's preference profile."""
    # Validate username
    username = _validate_username(args.username)

    with get_db() as conn:
        user_films = [dict(r) for r in conn.execute("""
            SELECT film_slug as slug, rating, watched, watchlisted, liked
            FROM user_films WHERE username = ?
        """, (username,))]
        all_films = {r['slug']: dict(r) for r in conn.execute("SELECT * FROM films")}
    
    if not user_films:
        logger.error(f"No data for '{username}'. Run: python main.py scrape {username}")
        return

    profile = build_profile(user_films, all_films, username=username)

    logger.info(f"\nProfile for {username}")
    logger.info(f"  Films: {profile.n_films} ({profile.n_rated} rated, {profile.n_liked} liked)")
    if profile.avg_liked_rating:
        logger.info(f"  Average rating: {profile.avg_liked_rating:.2f}★")
    
    if profile.genres:
        logger.info("\nTop genres:")
        for g, score in sorted(profile.genres.items(), key=lambda x: -x[1])[:10]:
            logger.info(f"  {g}: {score:+.2f}")
    
    if profile.directors:
        logger.info("\nTop directors:")
        for d, score in sorted(profile.directors.items(), key=lambda x: -x[1])[:10]:
            logger.info(f"  {d}: {score:+.2f}")
    
    if profile.actors:
        logger.info("\nTop actors:")
        for a, score in sorted(profile.actors.items(), key=lambda x: -x[1])[:10]:
            logger.info(f"  {a}: {score:+.2f}")
    
    if profile.decades:
        logger.info("\nDecade preferences:")
        for dec in sorted(profile.decades.keys()):
            score = profile.decades[dec]
            bar_length = int(max(0, score * 2))
            bar = "█" * bar_length
            logger.info(f"  {dec}s: {bar} ({score:+.1f})")


def cmd_similar(args: argparse.Namespace) -> None:
    """Find films similar to a specific film."""
    # Validate slug
    slug = _validate_slug(args.slug)
    
    with get_db() as conn:
        all_films = [dict(r) for r in conn.execute("SELECT * FROM films")]
    
    if not all_films:
        logger.error("No films in database. Run scrape first.")
        return
    
    recommender = MetadataRecommender(all_films)
    recs = recommender.similar_to(slug, n=args.limit)
    
    if not recs:
        logger.error(f"No film found with slug '{args.slug}'")
        return
    
    logger.info(f"\nFilms similar to {args.slug}:")
    for i, r in enumerate(recs, 1):
        logger.info(f"{i}. {r.title} ({r.year}) - Score: {r.score:.1f}")
        logger.info(f"   Why: {', '.join(r.reasons)}")


def cmd_triage(args: argparse.Namespace) -> None:
    """Rank user's watchlist by predicted enjoyment."""
    # Validate username
    username = _validate_username(args.username)

    with get_db() as conn:
        user_films = [dict(r) for r in conn.execute("""
            SELECT film_slug as slug, rating, watched, watchlisted, liked
            FROM user_films WHERE username = ?
        """, (username,))]

        watchlist = [r['film_slug'] for r in conn.execute("""
            SELECT film_slug FROM user_films
            WHERE username = ? AND watchlisted = 1
        """, (username,))]
        
        all_films = {r['slug']: dict(r) for r in conn.execute("SELECT * FROM films")}
    
    if not user_films:
        logger.error(f"No data for '{username}'. Run: python main.py scrape {username}")
        return

    if not watchlist:
        logger.error(f"No watchlist data for '{username}'.")
        return

    recommender = MetadataRecommender(list(all_films.values()))
    recs = recommender.recommend_from_candidates(user_films, watchlist, n=args.limit)

    logger.info(f"\nWatchlist Triage for {username} (Top {len(recs)}):")
    for i, r in enumerate(recs, 1):
        logger.info(f"{i}. {r.title} ({r.year}) - Score: {r.score:.1f}")
        logger.info(f"   Why: {', '.join(r.reasons)}")


def cmd_svd_info(args: argparse.Namespace) -> None:
    """Show SVD model diagnostics."""
    from .matrix_factorization import SVDRecommender
    
    with get_db() as conn:
        all_user_films = _load_all_user_films(conn)
    
    if len(all_user_films) < 5:
        logger.error("Need at least 5 users for SVD. Run 'discover' first.")
        return
    
    svd = SVDRecommender(n_factors=min(50, len(all_user_films) - 1))
    try:
        svd.fit(all_user_films)
    except ValueError as e:
        logger.error(f"Unable to fit SVD model: {e}")
        return
    
    logger.info(f"\nSVD Model Info:")
    logger.info(f"  Users: {len(svd.user_index)}")
    logger.info(f"  Films: {len(svd.item_index)}")
    logger.info(f"  Latent factors: {svd.n_factors}")
    logger.info(f"  Global mean rating: {svd.global_mean:.2f}★")
    
    # Show user with highest/lowest bias
    if svd.user_biases is not None:
        usernames = list(svd.user_index.keys())
        harshest_idx = svd.user_biases.argmin()
        generous_idx = svd.user_biases.argmax()
        logger.info(f"\n  Most generous rater: {usernames[generous_idx]} (+{svd.user_biases[generous_idx]:.2f})")
        logger.info(f"  Harshest rater: {usernames[harshest_idx]} ({svd.user_biases[harshest_idx]:.2f})")


def cmd_gaps(args: argparse.Namespace) -> None:
    """Find gaps in filmography of favorite directors."""
    # Validate username
    username = _validate_username(args.username)

    with get_db() as conn:
        user_films = [dict(r) for r in conn.execute("""
            SELECT film_slug as slug, rating, watched, watchlisted, liked
            FROM user_films WHERE username = ?
        """, (username,))]
        
        all_films = {r['slug']: dict(r) for r in conn.execute("SELECT * FROM films")}
    
    if not user_films:
        logger.error(f"No data for '{username}'. Run: python main.py scrape {username}")
        return

    recommender = MetadataRecommender(list(all_films.values()))
    gaps = recommender.find_gaps(
        user_films,
        min_director_score=args.min_score,
        limit_per_director=args.limit,
        min_year=args.min_year,
        max_year=args.max_year
    )

    if not gaps:
        logger.error(f"No gaps found for {username}. Try lowering --min-score.")
        return

    logger.info(f"\nFilmography Gaps for {username}:")
    for director, recs in sorted(gaps.items(), key=lambda x: -len(x[1])):
        logger.info(f"\n{director}:")
        for r in recs:
            logger.info(f"  - {r.title} ({r.year}) [{r.score:.1f}]")


def cmd_rebuild_idf(args: argparse.Namespace) -> None:
    """Rebuild IDF (Inverse Document Frequency) scores for all attributes."""
    init_db()

    logger.info("Computing IDF scores for all attributes...")
    logger.info("This may take a while depending on database size...")

    results = compute_and_store_idf()

    if not results:
        logger.error("\nNo films found in database. IDF table is empty.")
        logger.error("Run 'python main.py scrape <username>' to add films first.")
        return

    logger.info("\nIDF computation complete!")
    logger.info("\nAttribute counts:")
    for attr_type, count in sorted(results.items()):
        logger.info(f"  {attr_type}: {count} unique values")

    total = sum(results.values())
    logger.info(f"\nTotal: {total} attribute values indexed")
    logger.info("\nIDF scores will now be used to prioritize rare/distinctive preferences in recommendations.")


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


def cmd_prune_pending(args: argparse.Namespace) -> None:
    """Prune stale or low-priority pending users."""
    init_db()
    cutoff = f"-{args.older_than} days"
    removed = 0
    with get_db() as conn:
        removed += conn.execute(
            "DELETE FROM pending_users WHERE discovered_at < datetime('now', ?)",
            (cutoff,),
        ).rowcount
        if args.max_priority is not None:
            removed += conn.execute(
                "DELETE FROM pending_users WHERE priority <= ?",
                (args.max_priority,),
            ).rowcount
    logger.info(f"Pruned {removed} pending users")


def setup_jam_parser(subparsers):
    jam_parser = subparsers.add_parser(
        "jam",
        help="Generate recommendations for a group watch session",
        description="Find films that work for everyone in your watch party",
    )
    jam_parser.add_argument(
        "usernames",
        nargs="+",
        help="Letterboxd usernames (at least 2)",
    )
    jam_parser.add_argument(
        "--strategy",
        choices=[
            "least_misery",
            "most_pleasure",
            "average",
            "fairness",
            "approval",
            "multiplicative",
        ],
        default="fairness",
        help="How to combine preferences (default: fairness)",
    )
    jam_parser.add_argument("--limit", type=int, default=15, help="Number of recommendations")
    jam_parser.add_argument("--min-year", type=int, help="Minimum release year")
    jam_parser.add_argument("--max-year", type=int, help="Maximum release year")
    jam_parser.add_argument("--genres", nargs="+", help="Required genres")
    jam_parser.add_argument(
        "--include-divisive",
        action="store_true",
        help="Include films with divisive genres/directors",
    )
    jam_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    jam_parser.add_argument(
        "--weights",
        nargs="+",
        help="User weights as user:weight pairs (e.g., alex:2.0 for birthday person)",
    )
    jam_parser.set_defaults(func=cmd_jam)


def main():
    parser = argparse.ArgumentParser(description="Letterboxd Recommender")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Scrape user data")
    scrape_parser.add_argument("username", help="Letterboxd username")
    scrape_parser.add_argument("--refresh", type=int, metavar="DAYS",
                                help="Only scrape if last scraped more than N days ago")
    scrape_parser.add_argument("--include-lists", action="store_true", default=True,
                                help="Include user lists (favorites, ranked lists)")
    scrape_parser.add_argument("--no-include-lists", dest="include_lists", action="store_false",
                                help="Skip scraping user lists")
    scrape_parser.add_argument("--max-lists", type=int, default=50,
                                help="Maximum number of lists to scrape per user (default: 50)")
    scrape_parser.add_argument("--incremental", action="store_true",
                                help="Stop pagination when hitting already-scraped films (faster for updates)")
    scrape_parser.set_defaults(func=cmd_scrape)
    
    # Discover command
    discover_parser = subparsers.add_parser("discover", help="Discover and scrape other users")
    discover_parser.add_argument("source",
                                  nargs='?',
                                  choices=['following', 'followers', 'popular', 'film', 'film_reviews'],
                                  help="Source for user discovery (optional with --continue)")
    discover_parser.add_argument("--username", help="Username (for following/followers)")
    discover_parser.add_argument("--film-slug", help="Film slug (for film/film_reviews source, e.g., 'perfect-blue')")
    discover_parser.add_argument("--limit", type=int, default=50, help="Number of users to scrape")
    discover_parser.add_argument("--dry-run", action="store_true",
                                  help="Show which users would be scraped without actually scraping")
    discover_parser.add_argument("--continue", dest="continue_mode", action="store_true",
                                  help="Continue scraping from pending user queue without discovering new users")
    discover_parser.add_argument("--source-refresh-days", type=int, default=7,
                                  help="Days before re-crawling a source from page 1 (default: 7)")
    discover_parser.add_argument("--min-films", type=int, default=50,
                                  help="Minimum film count for activity pre-filtering (default: 50)")
    discover_parser.add_argument("--maintenance", action="store_true", default=False,
                                  help="Run VACUUM/ANALYZE after scraping batch")
    discover_parser.add_argument("--parallel-users", type=int, default=1,
                                  help="Scrape discovered users in parallel (async) with this many users at once")
    discover_parser.add_argument("--max-concurrent-requests", type=int, default=DEFAULT_MAX_CONCURRENT,
                                  help="Max concurrent HTTP requests when using --parallel-users")
    discover_parser.add_argument("--async-delay", type=float, default=DEFAULT_ASYNC_DELAY,
                                  help="Base async delay between requests when using --parallel-users")
    discover_parser.add_argument("--batch", type=int, default=DEFAULT_MAX_PER_BATCH,
                                  help="Batch size for film metadata fetches (parallel or serial)")
    discover_parser.add_argument("--queue-only", action="store_true",
                                 help="Only add discovered users to queue without scraping")
    discover_parser.set_defaults(func=cmd_discover)

    # Scrape daemon command
    daemon_parser = subparsers.add_parser("scrape-daemon", help="Continuously scrape from pending queue")
    daemon_parser.add_argument("--target", type=int, help="Stop after scraping N users")
    daemon_parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    daemon_parser.add_argument("--user-delay", type=float, default=0, help="Extra delay between users (seconds)")
    daemon_parser.add_argument("--wait-for-queue", action="store_true", help="Wait for new queue entries instead of exiting when empty")
    daemon_parser.add_argument("--wait-seconds", type=int, default=60, help="Wait duration when queue is empty and --wait-for-queue is set")
    daemon_parser.add_argument("--batch", type=int, default=DEFAULT_MAX_PER_BATCH, help="Batch size for metadata fetches")
    daemon_parser.add_argument("--remove-on-error", action="store_true", help="Remove user from queue on scrape error")
    daemon_parser.add_argument("--max-concurrent-users", type=int, default=DEFAULT_MAX_CONCURRENT_USERS,
                               help="Max users to scrape in parallel (async daemon)")
    daemon_parser.add_argument("--max-concurrent-requests", type=int, default=DEFAULT_MAX_CONCURRENT,
                               help="Max concurrent HTTP requests across tasks (async)")
    daemon_parser.add_argument("--async-delay", type=float, default=DEFAULT_ASYNC_DELAY,
                               help="Base async delay between requests (seconds)")
    daemon_parser.set_defaults(func=cmd_scrape_daemon)
    
    # Recommend command
    rec_parser = subparsers.add_parser("recommend", help="Generate recommendations")
    rec_parser.add_argument("username", help="Letterboxd username")
    rec_parser.add_argument("--strategy", choices=['metadata', 'collaborative', 'hybrid', 'svd'], 
                            default='metadata', help="Recommendation strategy")
    rec_parser.add_argument("--limit", type=int, default=20, help="Number of recommendations")
    rec_parser.add_argument("--min-year", type=int, help="Minimum release year")
    rec_parser.add_argument("--max-year", type=int, help="Maximum release year")
    rec_parser.add_argument("--genres", nargs="+", help="Filter by genres")
    rec_parser.add_argument("--exclude-genres", nargs="+", help="Exclude genres")
    rec_parser.add_argument("--min-rating", type=float, help="Minimum community rating (metadata only)")
    rec_parser.add_argument("--diversity", action="store_true", help="Enable diversity mode (metadata only)")
    rec_parser.add_argument("--max-per-director", type=int, default=2, help="Max films per director (diversity mode)")
    rec_parser.add_argument("--no-temporal-decay", action="store_true",
                            help="Disable temporal decay (treat old and new ratings equally)")
    rec_parser.add_argument(
        "--weighting-mode",
        choices=["absolute", "normalized", "blended"],
        default="absolute",
        help="Scoring weights: absolute (default), normalized (per-user z-score), or blended",
    )
    rec_parser.add_argument("--hybrid-meta-weight", type=float, help="Weight for metadata component in hybrid fusion")
    rec_parser.add_argument("--hybrid-collab-weight", type=float, help="Weight for collaborative component in hybrid fusion")
    rec_parser.add_argument("--hybrid-diversity", action="store_true", help="Apply diversity constraint to hybrid output")
    rec_parser.add_argument("--format", choices=['text', 'json', 'markdown', 'csv'], default='text',
                            help="Output format")
    rec_parser.add_argument("--explain", action="store_true",
                            help="Show detailed explanation for each recommendation")
    rec_parser.add_argument("--diversity-report", action="store_true",
                            help="Show diversity metrics for the recommendation set")
    rec_parser.set_defaults(func=cmd_recommend)
    
    # Explore command
    explore_parser = subparsers.add_parser("explore", help="Explore films by attribute")
    explore_parser.add_argument("attribute_type", choices=["director", "genre", "actor"],
                                 help="Type of attribute to explore")
    explore_parser.add_argument("value", help="Attribute value (e.g., 'Bong Joon-ho' for director)")
    explore_parser.add_argument("--limit", type=int, default=20,
                                 help="Number of films to show (default: 20)")
    explore_parser.set_defaults(func=cmd_explore)

    # Similar users command
    similar_users_parser = subparsers.add_parser("similar-users",
                                                   help="Find users with similar taste")
    similar_users_parser.add_argument("username", help="Your Letterboxd username")
    similar_users_parser.add_argument("--limit", type=int, default=10,
                                       help="Number of similar users to find")
    similar_users_parser.add_argument("--verbose", "-v", action="store_true",
                                       help="Show additional statistics")
    similar_users_parser.set_defaults(func=cmd_similar_users)
    
    # Group recommendation command
    setup_jam_parser(subparsers)
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.set_defaults(func=cmd_stats)

    # SVD command
    svd_parser = subparsers.add_parser("svd-info", help="Show SVD model diagnostics")
    svd_parser.set_defaults(func=cmd_svd_info)
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export database to JSON")
    export_parser.add_argument("file", help="Output JSON file path")
    export_parser.set_defaults(func=cmd_export)
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import database from JSON")
    import_parser.add_argument("file", help="Input JSON file path")
    import_parser.add_argument("--maintenance", action="store_true", default=RUN_VACUUM_ANALYZE_DEFAULT,
                               help="Run VACUUM/ANALYZE after import")
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
    gaps_parser.add_argument("--min-year", type=int, help="Minimum release year")
    gaps_parser.add_argument("--max-year", type=int, help="Maximum release year")
    gaps_parser.set_defaults(func=cmd_gaps)

    # Rebuild-IDF command
    rebuild_idf_parser = subparsers.add_parser("rebuild-idf", help="Rebuild IDF scores for attribute rarity weighting")
    rebuild_idf_parser.set_defaults(func=cmd_rebuild_idf)

    # Refresh metadata command
    refresh_meta_parser = subparsers.add_parser("refresh-metadata", help="Refresh missing/low-quality film metadata")
    refresh_meta_parser.add_argument("--limit", type=int, default=200, help="Max films to refresh")
    refresh_meta_parser.add_argument("--min-rating-count", type=int, default=50, help="Minimum rating count threshold to consider metadata low-quality")
    refresh_meta_parser.add_argument("--batch", type=int, default=DEFAULT_MAX_PER_BATCH, help="Batch size for metadata scraping")
    refresh_meta_parser.add_argument("--maintenance", action="store_true", default=False, help="Run ANALYZE after refresh")
    refresh_meta_parser.set_defaults(func=cmd_refresh_metadata)

    # Prune pending command
    prune_pending_parser = subparsers.add_parser("prune-pending", help="Prune stale pending users")
    prune_pending_parser.add_argument("--older-than", type=int, default=PENDING_STALE_DAYS, help="Age in days to prune pending entries")
    prune_pending_parser.add_argument("--max-priority", type=int, help="Remove pending entries at or below this priority")
    prune_pending_parser.set_defaults(func=cmd_prune_pending)

    # Queue management commands
    queue_status_parser = subparsers.add_parser("queue-status", help="Show pending queue status")
    queue_status_parser.add_argument("--verbose", action="store_true", help="Show next pending users")
    queue_status_parser.add_argument("--limit", type=int, default=10, help="Number of pending users to show with --verbose")
    queue_status_parser.set_defaults(func=cmd_queue_status)

    queue_add_parser = subparsers.add_parser("queue-add", help="Add usernames to queue")
    queue_add_parser.add_argument("usernames", nargs="*", help="Usernames to add")
    queue_add_parser.add_argument("--file", "-f", help="File with usernames (one per line)")
    queue_add_parser.add_argument("--priority", type=int, default=50, help="Priority for new queue entries")
    queue_add_parser.set_defaults(func=cmd_queue_add)

    queue_clear_parser = subparsers.add_parser("queue-clear", help="Clear pending queue")
    queue_clear_parser.add_argument("--source", help="Only clear users discovered from this source type")
    queue_clear_parser.set_defaults(func=cmd_queue_clear)

    # Discovery helpers
    refill_parser = subparsers.add_parser("discover-refill", help="Auto-refill queue from configured sources")
    refill_parser.add_argument("--min-queue", type=int, default=50, help="Only refill if queue size is below this")
    refill_parser.add_argument("--target", type=int, default=200, help="Target queue size after refill")
    refill_parser.add_argument("--source-refresh-days", type=int, default=7, help="Minimum age before reusing a discovery source")
    refill_parser.add_argument("--sources-file", help="JSON file with source definitions")
    refill_parser.add_argument("--min-films", type=int, default=50, help="Minimum films for activity filter")
    refill_parser.set_defaults(func=cmd_discover_refill)

    taste_parser = subparsers.add_parser("discover-taste", help="Discover users from your top films")
    taste_parser.add_argument("username", help="Your Letterboxd username")
    taste_parser.add_argument("--min-rating", type=float, default=4.0, help="Minimum rating to consider a film a favorite")
    taste_parser.add_argument("--film-limit", type=int, default=20, help="Number of top films to use")
    taste_parser.add_argument("--per-film", type=int, default=25, help="Reviewers per film to consider")
    taste_parser.add_argument("--min-films", type=int, default=50, help="Minimum film count for candidate users")
    taste_parser.set_defaults(func=cmd_discover_from_taste)

    # Session history
    session_parser = subparsers.add_parser("session-history", help="Show scraping session history")
    session_parser.add_argument("--limit", type=int, default=10, help="Number of sessions to display")
    session_parser.set_defaults(func=cmd_session_history)

    args = parser.parse_args()
    
    # Configure logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    args.func(args)