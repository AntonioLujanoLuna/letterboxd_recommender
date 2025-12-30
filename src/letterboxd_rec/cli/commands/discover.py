"""Discovery-related CLI commands."""

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from ...database import (
    parse_timestamp_naive,
    remove_pending_user,
    run_maintenance,
    save_user_follows,
    save_user_followers,
)
from ...config import (
    DEFAULT_MAX_PER_BATCH,
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_ASYNC_DELAY,
    DEFAULT_SCRAPER_DELAY,
)
from ..utils.helpers import _validate_username, _require_slug
from ..utils.scrapers import _scrape_film_metadata, _scrape_users_parallel

logger = logging.getLogger(__name__)


def _get_cli():
    """Late import cli module to support monkeypatching in tests."""
    from letterboxd_rec import cli
    return cli


def cmd_discover(args: argparse.Namespace) -> None:
    """Discover and scrape other users with caching and pending queue support."""
    cli = _get_cli()
    cli.init_db()
    scraper = cli.LetterboxdScraper(delay=getattr(args, "discover_delay", DEFAULT_SCRAPER_DELAY))

    try:
        # Check if we're in continue mode (drain pending queue only)
        continue_mode = getattr(args, 'continue_mode', False)
        source_refresh_days = getattr(args, 'source_refresh_days', 7)

        if continue_mode:
            # Drain pending queue only
            queue_stats = cli.get_pending_queue_stats()
            logger.info(f"\nPending queue stats:")
            logger.info(f"  Total pending users: {queue_stats['total']}")
            if queue_stats['breakdown']:
                logger.info(f"  By source type:")
                for source_type, count in queue_stats['breakdown'].items():
                    logger.info(f"    {source_type}: {count}")

            if queue_stats['total'] == 0:
                logger.info("\nNo pending users to scrape!")
                return

            pending = cli.get_pending_users(limit=args.limit)
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
                source_id = _require_slug(args.film_slug)
            elif source == 'popular':
                source_id = 'members'  # Popular members endpoint
            else:
                logger.error(f"Unknown source: {source}")
                return

            # Check for cached discovery source
            cached_source = cli.get_discovery_source(source, source_id)
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
            total_added = 0  # track users actually enqueued as we stream inserts
            page = start_page
            priority = cli.DISCOVERY_PRIORITY_MAP.get(source, 50)
            min_films = getattr(args, 'min_films', 50)

            # Calculate how many to discover (more than limit to account for duplicates)
            discover_limit = args.limit * getattr(args, "discover_overfetch", 3)

            # Apply activity pre-filtering
            filtered_count = 0
            activity_checked = 0

            while len(all_discovered) < discover_limit:
                # Get page of users based on source
                if source == 'following':
                    usernames = scraper.scrape_following(source_id, limit=page * 50)
                    usernames = usernames[(page-1)*50:page*50] if len(usernames) > (page-1)*50 else []
                    # Store social graph: source_id follows these usernames
                    if usernames:
                        save_user_follows(source_id, usernames)
                elif source == 'followers':
                    usernames = scraper.scrape_followers(source_id, limit=page * 50)
                    usernames = usernames[(page-1)*50:page*50] if len(usernames) > (page-1)*50 else []
                    # Store social graph: these usernames follow source_id
                    if usernames:
                        save_user_followers(source_id, usernames)
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

                # Apply activity pre-filtering for all sources (can be skipped for speed)
                if getattr(args, "skip_activity_checks", False):
                    filtered_usernames = usernames
                else:
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

                if filtered_usernames:
                    added = cli.add_pending_users(filtered_usernames, source, source_id, priority)
                    total_added += added
                    logger.info(f"  Enqueued {added} users this page (total enqueued: {total_added})")

                page += 1

                logger.info(f"  Page {page-1}: found {len(usernames)} users, {len(filtered_usernames)} passed filters (total: {len(all_discovered)})")

            logger.info(f"\nActivity filtering: {activity_checked} checked, {filtered_count} filtered out, {len(all_discovered)} passed")

            logger.info(f"Added {total_added} new users to pending queue (discovered {len(all_discovered)})")

            # Update discovery source cache
            cli.update_discovery_source(source, source_id, page - 1, len(all_discovered))

            # Queue-only mode: stop after enqueuing
            if getattr(args, 'queue_only', False):
                queue_stats = cli.get_pending_queue_stats()
                logger.info(f"\nQueue now has {queue_stats['total']} pending users")
                logger.info("Run 'scrape-daemon' to process the queue")
                return

            # Now get users to scrape from pending queue
            pending = cli.get_pending_users(limit=args.limit)
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
            with cli.get_db() as conn:
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

                    with cli.get_db() as conn:
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
        queue_stats = cli.get_pending_queue_stats()
        if queue_stats['total'] > 0:
            logger.info(f"\nRemaining in pending queue: {queue_stats['total']} users")
            logger.info("Run with --continue to scrape more from the queue")

        if getattr(args, "maintenance", False):
            run_maintenance(vacuum=True, analyze=True)

    finally:
        scraper.close()


def cmd_discover_refill(args: argparse.Namespace) -> None:
    """Auto-refill queue from multiple sources when it runs low."""
    cli = _get_cli()
    cli.init_db()
    stats = cli.get_pending_queue_stats()

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

    scraper = cli.LetterboxdScraper(delay=1.0)
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

            cached = cli.get_discovery_source(source_type, source_id)
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

            added = cli.add_pending_users(filtered, source_type, source_id, priority)
            total_added += added
            logger.info(f"  Added {added} users")

            cli.update_discovery_source(source_type, source_id, 1, len(filtered))

    finally:
        scraper.close()

    logger.info(f"\nRefill complete: added {total_added} users, queue now at {stats['total'] + total_added}")


def cmd_discover_from_taste(args: argparse.Namespace) -> None:
    """Discover users who reviewed films similar to your taste."""
    cli = _get_cli()
    cli.init_db()
    username = _validate_username(args.username)

    with cli.get_db(read_only=True) as conn:
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

    scraper = cli.LetterboxdScraper(delay=1.0)
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

            added = cli.add_pending_users(filtered, "taste_match", slug, priority=90)
            total_added += added
            logger.info(f"  {slug}: +{added} users")

    finally:
        scraper.close()

    logger.info(f"\nAdded {total_added} taste-matched users to queue")
