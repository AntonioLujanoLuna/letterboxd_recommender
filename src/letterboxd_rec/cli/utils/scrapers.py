"""Scraping utilities for CLI commands."""

import argparse
import asyncio
import json
import logging
from datetime import datetime

from tqdm import tqdm

from ...database import (
    get_db,
    populate_normalized_tables_batch,
    update_idf_incremental,
    remove_pending_user,
)
from ...scraper import LetterboxdScraper, AsyncLetterboxdScraper
from ...config import DEFAULT_MAX_PER_BATCH, DEFAULT_ASYNC_DELAY, DEFAULT_MAX_CONCURRENT

logger = logging.getLogger(__name__)


def _get_cli():
    """Late import cli module to support monkeypatching in tests."""
    from letterboxd_rec import cli
    return cli


def _scrape_film_metadata(
    scraper: 'LetterboxdScraper',
    slugs: list[str],
    max_per_batch: int = 100,
    use_async: bool = True,
    async_scraper: 'AsyncLetterboxdScraper | None' = None,
) -> None:
    """Scrape and save metadata for *all* given slugs, in chunks of `max_per_batch`.

    The chunk size only bounds how many films are in flight/saved at once. It is deliberately
    not a cap: callers treat every slug as handled afterwards, so dropping the overflow would
    leave those films without metadata permanently.
    """
    if not slugs:
        return

    chunk_size = max(1, max_per_batch)
    for start in range(0, len(slugs), chunk_size):
        chunk = slugs[start:start + chunk_size]

        if use_async and len(chunk) > 10:
            logger.info(f"Fetching {len(chunk)} films (async)... [{start + len(chunk)}/{len(slugs)}]")

            async def scrape_batch(shared: 'AsyncLetterboxdScraper | None', batch=chunk):
                if shared:
                    return await shared.scrape_films_batch(batch)
                async with _get_cli().make_async_scraper(delay=0.2, max_concurrent=5) as temp_async:
                    return await temp_async.scrape_films_batch(batch)

            metadata_list = asyncio.run(scrape_batch(async_scraper))
        else:
            metadata_list = []
            for slug in tqdm(chunk, desc="Metadata"):
                meta = scraper.scrape_film(slug)
                if meta:
                    metadata_list.append(meta)

        _persist_metadata_batch(metadata_list)


def _persist_metadata_batch(metadata_list: list) -> None:
    """Persist scraped film metadata and keep normalized tables/idf in sync."""
    if not metadata_list:
        return

    with get_db() as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO films
            (slug, title, year, directors, genres, cast, themes, runtime, avg_rating, rating_count,
             fan_count, is_short, is_animation,
             countries, languages, writers, cinematographers, composers, tmdb_id, imdb_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [(
            m.slug, m.title, m.year,
            json.dumps(m.directors), json.dumps(m.genres),
            json.dumps(m.cast), json.dumps(m.themes),
            m.runtime, m.avg_rating, m.rating_count,
            m.fan_count, int(m.is_short), int(m.is_animation),
            json.dumps(m.countries), json.dumps(m.languages),
            json.dumps(m.writers), json.dumps(m.cinematographers),
            json.dumps(m.composers), getattr(m, "tmdb_id", None), getattr(m, "imdb_id", None),
        ) for m in metadata_list])

        # Populate normalized tables for fast queries (batch operation)
        populate_normalized_tables_batch(conn, metadata_list)

    # Incrementally update IDF scores for new films
    new_slugs = [m.slug for m in metadata_list]
    update_idf_incremental(new_slugs)


async def _scrape_film_metadata_async(
    async_scraper: 'AsyncLetterboxdScraper',
    slugs: list[str],
    max_per_batch: int = 100,
) -> set[str]:
    """
    Async helper to scrape film metadata using a shared AsyncLetterboxdScraper.
    Fetches *all* given slugs in chunks of `max_per_batch` and avoids blocking the event loop.
    """
    saved: set[str] = set()
    if not slugs:
        return saved

    # max_per_batch is the chunk size, not a cap: callers treat every slug as handled afterwards,
    # so anything not fetched here would never get metadata. The returned set lets them see which
    # slugs failed so those can be retried later.
    chunk = max(1, max_per_batch)
    for start in range(0, len(slugs), chunk):
        metadata_list = await async_scraper.scrape_films_batch(slugs[start:start + chunk])
        if not metadata_list:
            continue
        await asyncio.to_thread(_persist_metadata_batch, metadata_list)
        saved.update(m.slug for m in metadata_list)
        logger.info(f"Saved {len(metadata_list)} films")
    return saved


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

    async with _get_cli().make_async_scraper(
        delay=getattr(args, "async_delay", DEFAULT_ASYNC_DELAY),
        max_concurrent=getattr(args, "max_concurrent_requests", DEFAULT_MAX_CONCURRENT),
        max_rps=getattr(args, "max_rps", None),
        rps_ceiling=getattr(args, "auto_rps", None),
    ) as async_scraper:

        async def _process(username: str):
            async with user_semaphore:
                try:
                    interactions = await async_scraper.scrape_user_smart(username)

                    if interactions:
                        await _persist_user_films(username, interactions)

                        async with known_slugs_lock:
                            new_slugs = [i.film_slug for i in interactions if i.film_slug not in known_film_slugs]
                            known_film_slugs.update(new_slugs)

                        if new_slugs:
                            saved = await _scrape_film_metadata_async(
                                async_scraper,
                                new_slugs,
                                max_per_batch=getattr(args, "batch", DEFAULT_MAX_PER_BATCH),
                            )
                            async with known_slugs_lock:
                                known_film_slugs.difference_update(set(new_slugs) - saved)

                        logger.info(f"{username}: {len(interactions)} films (parallel)")
                    else:
                        logger.warning(f"No interactions found for {username}")

                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Error scraping {username}: {exc}")
                finally:
                    await asyncio.to_thread(remove_pending_user, username)

        await asyncio.gather(*[asyncio.create_task(_process(u)) for u in usernames])
