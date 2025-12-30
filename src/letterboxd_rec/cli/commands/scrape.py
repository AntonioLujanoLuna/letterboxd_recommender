"""Scraping-related CLI commands."""

import argparse
import asyncio
import logging
from datetime import datetime

from tqdm import tqdm

from ...database import (
    init_db,
    get_db,
    parse_timestamp_naive,
    remove_pending_user,
    create_scrape_session,
    update_session_progress,
    complete_session,
    get_pending_users,
    save_user_follows,
    save_user_followers,
)
from ...scraper import LetterboxdScraper, AsyncLetterboxdScraper
from ...config import (
    DEFAULT_MAX_PER_BATCH,
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_MAX_CONCURRENT_USERS,
    DEFAULT_ASYNC_DELAY,
    NOTIFICATION_INTERVAL,
)
from ..utils.helpers import _validate_username, send_notification
from ..utils.scrapers import _scrape_film_metadata, _scrape_film_metadata_async

logger = logging.getLogger(__name__)


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

        # Scrape social graph if enabled
        if getattr(args, "include_social", False):
            logger.info(f"\nScraping {username}'s social graph...")

            following = scraper.scrape_following(username, limit=args.social_limit)
            if following:
                save_user_follows(username, following)
                logger.info(f"  Saved {len(following)} following")

            followers = scraper.scrape_followers(username, limit=args.social_limit)
            if followers:
                save_user_followers(username, followers)
                logger.info(f"  Saved {len(followers)} followers")

        logger.info(f"\nDone! {len(interactions)} films for {username}")

    finally:
        scraper.close()


def cmd_scrape_social(args: argparse.Namespace) -> None:
    """Scrape and store social graph (following/followers) for a user."""
    init_db()
    username = _validate_username(args.username)
    scraper = LetterboxdScraper(delay=1.0)

    try:
        total_saved = 0

        if args.following or args.both:
            logger.info(f"Scraping who {username} follows...")
            following = scraper.scrape_following(username, limit=args.limit)
            saved = save_user_follows(username, following)
            logger.info(f"  Saved {saved} following relationships")
            total_saved += saved

        if args.followers or args.both:
            logger.info(f"Scraping who follows {username}...")
            followers = scraper.scrape_followers(username, limit=args.limit)
            saved = save_user_followers(username, followers)
            logger.info(f"  Saved {saved} follower relationships")
            total_saved += saved

        logger.info(f"\nTotal: {total_saved} social edges saved")

        # Show current graph stats
        from ...database import get_social_graph_stats
        stats = get_social_graph_stats()
        logger.info(f"\nSocial graph now has:")
        logger.info(f"  {stats['total_edges']} total edges")
        logger.info(f"  {stats['unique_followers']} unique users following others")
        logger.info(f"  {stats['unique_followees']} unique users being followed")

    finally:
        scraper.close()


def cmd_backfill_social(args: argparse.Namespace) -> None:
    """Backfill social graph for users already in the database."""
    init_db()

    with get_db(read_only=True) as conn:
        users_with_social: set[str] = set()
        for row in conn.execute("SELECT DISTINCT follower FROM user_follows"):
            users_with_social.add(row["follower"])
        for row in conn.execute("SELECT DISTINCT followee FROM user_follows"):
            users_with_social.add(row["followee"])

        all_users = {row["username"] for row in conn.execute("SELECT DISTINCT username FROM user_films")}

    users_needing_social = all_users - users_with_social

    if not users_needing_social:
        logger.info("All users already have social data")
        return

    logger.info(f"Found {len(users_needing_social)} users without social data")

    if args.dry_run:
        sample = list(users_needing_social)[:20]
        for u in sample:
            logger.info(f"  Would scrape: {u}")
        if len(users_needing_social) > len(sample):
            logger.info(f"  ... and {len(users_needing_social) - len(sample)} more")
        return

    scraper = LetterboxdScraper(delay=1.0)
    try:
        for username in tqdm(list(users_needing_social)[: args.limit], desc="Social"):
            following = scraper.scrape_following(username, limit=args.social_limit)
            if following:
                save_user_follows(username, following)

            followers = scraper.scrape_followers(username, limit=args.social_limit)
            if followers:
                save_user_followers(username, followers)
    finally:
        scraper.close()

    from ...database import get_social_graph_stats
    stats = get_social_graph_stats()
    logger.info(f"\nSocial graph now has {stats['total_edges']} edges")
    logger.info(f"Unique followers: {stats['unique_followers']}, unique followees: {stats['unique_followees']}")


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
            from ...database import get_pending_queue_stats
            queue_remaining = await asyncio.to_thread(lambda: get_pending_queue_stats()['total'])
            send_notification(
                f"Scrape progress: {scraped_count_local} users done, {queue_remaining} remaining"
            )

    async with AsyncLetterboxdScraper(
        delay=getattr(args, "async_delay", getattr(args, "delay", DEFAULT_ASYNC_DELAY)),
        max_concurrent=getattr(args, "max_concurrent_requests", DEFAULT_MAX_CONCURRENT),
    ) as async_scraper:

        async def _scrape_social_async(username: str):
            """Async helper to scrape and persist social graph edges."""
            following = await async_scraper.scrape_following_async(username, limit=getattr(args, "social_limit", 200))
            if following:
                await asyncio.to_thread(save_user_follows, username, following)
                logger.info(f"  Saved {len(following)} following edges for {username}")

            followers = await async_scraper.scrape_followers_async(username, limit=getattr(args, "social_limit", 200))
            if followers:
                await asyncio.to_thread(save_user_followers, username, followers)
                logger.info(f"  Saved {len(followers)} follower edges for {username}")

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

                        if getattr(args, "include_social", False):
                            await _scrape_social_async(username)

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
