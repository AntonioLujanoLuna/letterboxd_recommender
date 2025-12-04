import argparse
import json
import logging
import atexit
from collections import defaultdict
from datetime import datetime, timedelta

from .database import (
    init_db, get_db, load_json, load_user_lists, parse_timestamp_naive,
    get_discovery_source, update_discovery_source, add_pending_users,
    get_pending_users, remove_pending_user, get_pending_queue_stats,
    compute_and_store_idf, populate_normalized_tables_batch, close_pool
)
from .config import DISCOVERY_PRIORITY_MAP
from .scraper import LetterboxdScraper
from .recommender import MetadataRecommender, CollaborativeRecommender, Recommendation
from .profile import build_profile
from tqdm import tqdm
import asyncio

logger = logging.getLogger(__name__)

# Register cleanup on exit
atexit.register(close_pool)


def _validate_slug(slug: str) -> str:
    """
    Validate a film slug to prevent injection or invalid characters.
    Raises ValueError if slug contains invalid characters.
    Returns lowercased valid slug.
    """
    import re
    if not re.match(r'^[a-z0-9-]+$', slug.lower()):
        raise ValueError(f"Invalid slug: {slug}")
    return slug.lower()


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


def _scrape_film_metadata(scraper: 'LetterboxdScraper', slugs: list[str], max_per_batch: int = 100, use_async: bool = True) -> None:
    """Helper to scrape film metadata for a list of slugs."""
    if not slugs:
        return

    limited_slugs = slugs[:max_per_batch]

    if use_async and len(limited_slugs) > 10:
        from .scraper import AsyncLetterboxdScraper
        logger.info(f"Fetching {len(limited_slugs)} films (async)...")

        async def scrape_batch():
            async with AsyncLetterboxdScraper(delay=0.2, max_concurrent=5) as async_scraper:
                return await async_scraper.scrape_films_batch(limited_slugs)

        metadata_list = asyncio.run(scrape_batch())
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
        with get_db() as conn:
            existing_film_slugs = {r['slug'] for r in conn.execute("SELECT slug FROM films")}

        for username in tqdm(usernames_to_scrape, desc="Users"):
            try:
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
                _scrape_film_metadata(scraper, new_slugs, max_per_batch=100)
                existing_film_slugs.update(new_slugs)

            except Exception as e:
                logger.error(f"Error scraping {username}: {e}")

        logger.info(f"\nDone! Scraped {len(usernames_to_scrape)} users.")

        # Show remaining pending queue stats
        queue_stats = get_pending_queue_stats()
        if queue_stats['total'] > 0:
            logger.info(f"\nRemaining in pending queue: {queue_stats['total']} users")
            logger.info("Run with --continue to scrape more from the queue")

    finally:
        scraper.close()


def cmd_recommend(args: argparse.Namespace) -> None:
    """Generate recommendations."""
    # Validate username
    username = _validate_username(args.username)

    strategy = getattr(args, 'strategy', 'metadata')

    with get_db() as conn:
        # Load user films based on strategy to avoid duplicate queries
        if strategy in ('hybrid', 'collaborative'):
            # Load all users' films once (includes target user)
            all_user_films = _load_all_user_films(conn)
            user_films = all_user_films.get(username, [])
        else:
            # Metadata-only: just load target user's films (include scraped_at for temporal decay)
            user_films = [dict(r) for r in conn.execute("""
                SELECT film_slug as slug, rating, watched, watchlisted, liked, scraped_at
                FROM user_films WHERE username = ?
            """, (username,))]

        if not user_films:
            logger.error(f"No data for '{username}'. Run: python main.py scrape {username}")
            return

        # Load film metadata with SQL-side filtering for better performance
        # This avoids loading all films into memory when filters are specified
        all_films = _load_films_with_filters(
            conn,
            min_year=args.min_year,
            max_year=args.max_year,
            min_rating=args.min_rating
        )

        # Check for missing metadata
        seen_slugs = {f['slug'] for f in user_films}
        missing_slugs = seen_slugs - set(all_films.keys())
        if missing_slugs:
            logger.warning(f"Missing metadata for {len(missing_slugs)} films in {username}'s history. Consider running 'scrape' again.")

        if strategy == 'hybrid':
            # Hybrid: combine metadata and collaborative (all_user_films already loaded)
            
            # Get recommendations from both strategies
            meta_rec = MetadataRecommender(list(all_films.values()))
            collab_rec = CollaborativeRecommender(all_user_films, all_films)
            
            meta_recs = meta_rec.recommend(
                user_films,
                n=args.limit * 2,
                min_year=args.min_year,
                max_year=args.max_year,
                genres=args.genres,
                exclude_genres=args.exclude_genres,
                min_rating=args.min_rating,
                username=username
            )
            collab_recs = collab_rec.recommend(
                username,
                n=args.limit * 2,
                min_year=args.min_year,
                max_year=args.max_year,
                genres=args.genres,
                exclude_genres=args.exclude_genres
            )
            
            # Merge using normalized scores to preserve score magnitude
            # Normalize scores from each strategy to [0, 1] range for fair combination
            scores = {}
            reasons_map = {}

            # Normalize metadata scores
            meta_scores = {r.slug: r.score for r in meta_recs}
            if meta_scores:
                max_meta_score = max(meta_scores.values())
                min_meta_score = min(meta_scores.values())
                score_range = max_meta_score - min_meta_score

                for r in meta_recs:
                    # Normalize to [0, 1] and weight by 0.5
                    if score_range > 0:
                        normalized = (r.score - min_meta_score) / score_range
                    else:
                        normalized = 1.0
                    scores[r.slug] = normalized * 0.5
                    if r.slug not in reasons_map:
                        reasons_map[r.slug] = []
                    reasons_map[r.slug].extend(r.reasons)

            # Normalize collaborative scores
            collab_scores = {r.slug: r.score for r in collab_recs}
            if collab_scores:
                max_collab_score = max(collab_scores.values())
                min_collab_score = min(collab_scores.values())
                score_range = max_collab_score - min_collab_score

                for r in collab_recs:
                    # Normalize to [0, 1] and weight by 0.5
                    if score_range > 0:
                        normalized = (r.score - min_collab_score) / score_range
                    else:
                        normalized = 1.0
                    scores[r.slug] = scores.get(r.slug, 0) + normalized * 0.5
                    if r.slug not in reasons_map:
                        reasons_map[r.slug] = []
                    reasons_map[r.slug].extend(r.reasons)
            
            # Build final recommendations
            merged = sorted(scores.items(), key=lambda x: -x[1])[:args.limit]
            recs = []
            for slug, score in merged:
                if slug in all_films:
                    film = all_films[slug]
                    # Deduplicate reasons while preserving order
                    unique_reasons = list(dict.fromkeys(reasons_map.get(slug, [])))
                    recs.append(Recommendation(
                        slug=slug,
                        title=film.get('title', slug),
                        year=film.get('year'),
                        score=score,
                        reasons=unique_reasons[:3] # Top 3 combined reasons
                    ))
                    
        elif strategy == 'collaborative':
            # Collaborative: all_user_films already loaded above
            recommender = CollaborativeRecommender(all_user_films, all_films)
            recs = recommender.recommend(
                username,
                n=args.limit,
                min_year=args.min_year,
                max_year=args.max_year,
                genres=args.genres,
                exclude_genres=args.exclude_genres
            )
        else:
            # Metadata-based (default)
            diversity = getattr(args, 'diversity', False)
            max_per_director = getattr(args, 'max_per_director', 2)
            use_temporal_decay = not getattr(args, 'no_temporal_decay', False)
            # Load user lists for profile building
            user_lists = load_user_lists(username)

            # Build profile with temporal decay support
            from .profile import build_profile
            profile = build_profile(
                user_films,
                all_films,
                user_lists=user_lists,
                username=username,
                use_temporal_decay=use_temporal_decay
            )

            recommender = MetadataRecommender(list(all_films.values()))
            recs = recommender.recommend(
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
                user_lists=user_lists
            )
    
    # Format results
    output_format = getattr(args, 'format', 'text')
    if output_format == 'json':
        output = []
        for r in recs:
            film = all_films.get(r.slug, {})
            output.append({
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
            })
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
    else:
        logger.info(f"\nTop {len(recs)} recommendations for {username} ({strategy}):")
        for i, r in enumerate(recs, 1):
            logger.info(f"{i}. {r.title} ({r.year}) - Score: {r.score:.1f}")
            logger.info(f"   Why: {', '.join(r.reasons)}")


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
    with get_db() as conn:
        user_films = [dict(r) for r in conn.execute("SELECT * FROM user_films")]
        films = [dict(r) for r in conn.execute("SELECT * FROM films")]
    
    data = {
        "user_films": user_films,
        "films": films,
        "exported_at": datetime.now().isoformat()
    }
    
    with open(args.file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Exported {len(user_films)} user interactions and {len(films)} films to {args.file}")


def cmd_import(args: argparse.Namespace) -> None:
    """Import database from JSON file."""
    with open(args.file, 'r') as f:
        data = json.load(f)
    
    init_db()
    
    with get_db() as conn:
        if 'films' in data:
            for film in data['films']:
                conn.execute("""
                    INSERT OR REPLACE INTO films
                    (slug, title, year, directors, genres, cast, themes, runtime, avg_rating, rating_count,
                     countries, languages, writers, cinematographers, composers)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    film['slug'], film.get('title'), film.get('year'),
                    film.get('directors'), film.get('genres'),
                    film.get('cast'), film.get('themes'),
                    film.get('runtime'), film.get('avg_rating'), film.get('rating_count'),
                    film.get('countries'), film.get('languages'),
                    film.get('writers'), film.get('cinematographers'), film.get('composers')
                ))
            logger.info(f"Imported {len(data['films'])} films")
        
        if 'user_films' in data:
            for uf in data['user_films']:
                conn.execute("""
                    INSERT OR REPLACE INTO user_films 
                    (username, film_slug, rating, watched, watchlisted, liked, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    uf['username'], uf['film_slug'], uf.get('rating'),
                    uf.get('watched'), uf.get('watchlisted'), uf.get('liked'),
                    uf.get('scraped_at')
                ))
            logger.info(f"Imported {len(data['user_films'])} user interactions")
    
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
    discover_parser.set_defaults(func=cmd_discover)
    
    # Recommend command
    rec_parser = subparsers.add_parser("recommend", help="Generate recommendations")
    rec_parser.add_argument("username", help="Letterboxd username")
    rec_parser.add_argument("--strategy", choices=['metadata', 'collaborative', 'hybrid'], 
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
    rec_parser.add_argument("--format", choices=['text', 'json', 'markdown', 'csv'], default='text',
                            help="Output format")
    rec_parser.set_defaults(func=cmd_recommend)
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.set_defaults(func=cmd_stats)
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export database to JSON")
    export_parser.add_argument("file", help="Output JSON file path")
    export_parser.set_defaults(func=cmd_export)
    
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
    gaps_parser.add_argument("--min-year", type=int, help="Minimum release year")
    gaps_parser.add_argument("--max-year", type=int, help="Maximum release year")
    gaps_parser.set_defaults(func=cmd_gaps)

    # Rebuild-IDF command
    rebuild_idf_parser = subparsers.add_parser("rebuild-idf", help="Rebuild IDF scores for attribute rarity weighting")
    rebuild_idf_parser.set_defaults(func=cmd_rebuild_idf)

    args = parser.parse_args()
    
    # Configure logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    args.func(args)