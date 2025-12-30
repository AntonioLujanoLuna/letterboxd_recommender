"""Asynchronous Letterboxd scraper implementation."""

import asyncio
import logging

import httpx
from selectolax.parser import HTMLParser

from ..config import (
    HTTP_TIMEOUT,
    MAX_CONSECUTIVE_EXISTING,
    MAX_HTTP_RETRIES,
    DEFAULT_RETRY_AFTER,
    SCRAPER_HTTP2,
    SCRAPER_ADAPTIVE_DELAY_MAX,
    SCRAPER_429_BACKOFF,
    SCRAPER_PAGE_SIZE,
    LETTERBOXD_COOKIE,
)
from .types import FilmInteraction, FilmMetadata
from .utils import (
    parse_cookie_header,
    validate_slug,
    parse_rating_span,
    parse_fan_count_from_html,
    parse_film_page,
)

logger = logging.getLogger(__name__)


class AsyncLetterboxdScraper:
    """Async scraper for parallel metadata fetching with coordinated rate limiting."""

    BASE = "https://letterboxd.com"

    def __init__(self, delay: float = 0.2, max_concurrent: int = 5):
        self.delay = delay
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.client = None
        self.cookies = parse_cookie_header(LETTERBOXD_COOKIE)
        # Coordinated rate limiting: when one task hits 429, all tasks pause
        self._rate_limit_event = asyncio.Event()
        self._rate_limit_event.set()  # Start in "not rate limited" state

    @staticmethod
    def _parse_rating(span) -> float | None:
        """Reuse the shared rating parser."""
        return parse_rating_span(span)

    @staticmethod
    def _detect_soft_block(tree: HTMLParser | None) -> bool:
        """Detect CAPTCHA/please-wait soft blocks to avoid burning retries."""
        if not tree:
            return False

        if tree.css_first("form[action*='captcha'], .captcha-container"):
            return True

        body_el = tree.css_first("body")
        body_text = body_el.text() if body_el else ""
        soft_block_phrases = [
            "please wait",
            "too many requests",
            "try again later",
            "access denied",
        ]
        return any(phrase in body_text.lower() for phrase in soft_block_phrases)

    async def __aenter__(self):
        """Async context manager entry."""
        self.client = httpx.AsyncClient(
            headers={"User-Agent": "Mozilla/5.0 (compatible; letterboxd-rec/1.0)"},
            follow_redirects=True,
            timeout=HTTP_TIMEOUT,
            http2=SCRAPER_HTTP2,
            cookies=self.cookies or None,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()
        return False

    async def _get(self, url: str) -> HTMLParser | None:
        """
        Internal async GET with coordinated rate limiting and retry logic.
        Requires the context manager to have initialized self.client.
        """
        if not self.client:
            raise RuntimeError("AsyncLetterboxdScraper must be used as an async context manager for user scraping")

        async with self.semaphore:
            await asyncio.sleep(self.delay)

            for attempt in range(MAX_HTTP_RETRIES):
                await self._rate_limit_event.wait()

                try:
                    resp = await self.client.get(url)

                    if resp.status_code == 404:
                        return None

                    if resp.status_code == 429:
                        retry_after = int(resp.headers.get("Retry-After", DEFAULT_RETRY_AFTER))
                        logger.warning(
                            f"Rate limited on {url}, pausing ALL tasks for {retry_after}s "
                            f"(attempt {attempt + 1}/{MAX_HTTP_RETRIES})"
                        )
                        self._rate_limit_event.clear()
                        await asyncio.sleep(retry_after)
                        self._rate_limit_event.set()

                        import random
                        jitter = random.uniform(0, self.delay * 2)
                        await asyncio.sleep(jitter)
                        self.delay = min(SCRAPER_ADAPTIVE_DELAY_MAX, self.delay * SCRAPER_429_BACKOFF)
                        continue

                    resp.raise_for_status()
                    return HTMLParser(resp.text)

                except httpx.TimeoutException:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Timeout on {url}, retrying in {wait_time}s (attempt {attempt + 1}/{MAX_HTTP_RETRIES})"
                    )
                    await asyncio.sleep(wait_time)

                except httpx.HTTPStatusError as exc:
                    logger.error(f"HTTP {exc.response.status_code} on {url}: {exc}")
                    return None

                except httpx.HTTPError as exc:
                    logger.error(f"Request error on {url}: {type(exc).__name__}: {exc}")
                    return None

            logger.error(f"Max retries exceeded for {url}")
            return None

    async def _get_with_soft_block_recovery(self, url: str) -> HTMLParser | None:
        """Enhanced _get with soft block detection and a short backoff sequence."""
        tree = await self._get(url)
        if tree and self._detect_soft_block(tree):
            logger.warning(f"Soft block detected on {url}, backing off...")
            for wait_time in [1, 2, 4]:
                await asyncio.sleep(wait_time)
                tree = await self._get(url)
                if tree and not self._detect_soft_block(tree):
                    return tree
            logger.error(f"Persistent soft block on {url}")
            return None
        return tree

    async def scrape_user(self, username: str, existing_slugs: set[str] | None = None, stop_on_existing: bool = False) -> list[FilmInteraction]:
        """
        Async variant of user scraping. Uses the shared client/semaphore so requests are globally rate limited.
        """
        films: dict[str, FilmInteraction] = {}

        logger.info(f"Scraping {username}'s films (async)...")
        page = 1
        consecutive_existing = 0

        while True:
            tree = await self._get_with_soft_block_recovery(f"{self.BASE}/{username}/films/page/{page}/")
            if not tree:
                break

            items = tree.css("li.griditem")
            if not items:
                break

            page_had_new_films = False
            stop_after_page = len(items) < SCRAPER_PAGE_SIZE

            for item in items:
                react_comp = item.css_first("div.react-component")
                if not react_comp:
                    continue

                slug = validate_slug(react_comp.attributes.get("data-item-slug"))
                if not slug:
                    continue

                if stop_on_existing and existing_slugs and slug in existing_slugs:
                    consecutive_existing += 1
                    if consecutive_existing >= MAX_CONSECUTIVE_EXISTING:
                        logger.info(
                            f"  Found {consecutive_existing} consecutive existing films, stopping early (incremental mode)"
                        )
                        return list(films.values())
                    continue
                else:
                    consecutive_existing = 0
                    page_had_new_films = True

                rating = None
                liked = False
                viewing_data = item.css_first("p.poster-viewingdata")
                if viewing_data:
                    rating_span = viewing_data.css_first("span.rating")
                    if rating_span:
                        rating = self._parse_rating(rating_span)
                    liked = viewing_data.css_first("span.like") is not None

                films[slug] = FilmInteraction(slug, rating, True, False, liked)

            page += 1
            logger.debug(f"  Watched page {page-1}: {len(items)} films")

            if stop_on_existing and not page_had_new_films and existing_slugs:
                logger.info(f"  No new films found on page {page-1}, stopping early (incremental mode)")
                break
            if stop_after_page:
                break

        logger.info(f"Scraping {username}'s watchlist (async)...")
        page = 1
        while True:
            tree = await self._get_with_soft_block_recovery(f"{self.BASE}/{username}/watchlist/page/{page}/")
            if not tree:
                break

            items = tree.css("li.griditem")
            if not items:
                break

            stop_after_page = len(items) < SCRAPER_PAGE_SIZE
            for item in items:
                react_comp = item.css_first("div.react-component")
                if not react_comp:
                    continue

                slug = validate_slug(react_comp.attributes.get("data-item-slug"))
                if not slug:
                    continue

                if slug in films:
                    films[slug].watchlisted = True
                else:
                    films[slug] = FilmInteraction(slug, None, False, True, False)

            page += 1
            if stop_after_page:
                break

        n_rated = sum(1 for f in films.values() if f.rating)
        n_liked = sum(1 for f in films.values() if f.liked)
        logger.info(f"Total: {len(films)} films ({n_rated} rated, {n_liked} liked)")
        return list(films.values())

    async def scrape_user_smart(
        self,
        username: str,
        known_film_count: int | None = None,
    ) -> list[FilmInteraction]:
        """
        Async smart scraping variant with adaptive paging/early stop.
        Mirrors the synchronous scrape_user_smart but keeps everything async-friendly.
        """
        films: dict[str, FilmInteraction] = {}

        estimated_pages = None
        if known_film_count:
            estimated_pages = (known_film_count // SCRAPER_PAGE_SIZE) + 1
            if estimated_pages > 10:
                logger.info(f"Large profile detected ({known_film_count} films, ~{estimated_pages} pages)")

        page = 1
        empty_pages = 0

        while True:
            tree = await self._get_with_soft_block_recovery(f"{self.BASE}/{username}/films/page/{page}/")
            if not tree:
                break

            items = tree.css("li.griditem")
            if not items:
                empty_pages += 1
                if empty_pages >= 2:
                    break
                page += 1
                continue

            empty_pages = 0
            stop_after_page = len(items) < SCRAPER_PAGE_SIZE * 0.8

            for item in items:
                react_comp = item.css_first("div.react-component")
                if not react_comp:
                    continue

                slug = validate_slug(react_comp.attributes.get("data-item-slug"))
                if not slug:
                    continue

                rating = None
                liked = False

                viewing_data = item.css_first("p.poster-viewingdata")
                if viewing_data:
                    rating_span = viewing_data.css_first("span.rating")
                    if rating_span:
                        rating = parse_rating_span(rating_span)

                    liked = viewing_data.css_first("span.like") is not None

                films[slug] = FilmInteraction(slug, rating, True, False, liked)

            page += 1
            if stop_after_page:
                break
            if estimated_pages and page > estimated_pages + 2:
                break

        # Watchlist sweep (reuse standard logic with smart requests)
        page = 1
        while True:
            tree = await self._get_with_soft_block_recovery(f"{self.BASE}/{username}/watchlist/page/{page}/")
            if not tree:
                break

            items = tree.css("li.griditem")
            if not items:
                break

            stop_after_page = len(items) < SCRAPER_PAGE_SIZE
            for item in items:
                react_comp = item.css_first("div.react-component")
                if not react_comp:
                    continue

                slug = validate_slug(react_comp.attributes.get("data-item-slug"))
                if not slug:
                    continue

                if slug in films:
                    films[slug].watchlisted = True
                else:
                    films[slug] = FilmInteraction(slug, None, False, True, False)

            page += 1
            if stop_after_page:
                break

        n_rated = sum(1 for f in films.values() if f.rating)
        n_liked = sum(1 for f in films.values() if f.liked)
        logger.info(f"Smart scrape total (async): {len(films)} films ({n_rated} rated, {n_liked} liked)")
        return list(films.values())

    async def scrape_following_async(self, username: str, limit: int = 100) -> list[str]:
        """Scrape usernames that the target user follows (async)."""
        usernames: list[str] = []
        page = 1

        logger.info(f"Scraping {username}'s following (async)...")
        while len(usernames) < limit:
            tree = await self._get_with_soft_block_recovery(f"{self.BASE}/{username}/following/page/{page}/")
            if not tree:
                break

            links = tree.css("a.name")
            if not links:
                break

            for link in links:
                href = link.attributes.get("href", "")
                if href.startswith("/") and href.endswith("/"):
                    user = href.strip("/")
                    if user and user not in usernames:
                        usernames.append(user)
                        if len(usernames) >= limit:
                            break

            page += 1

        logger.info(f"  Found {len(usernames)} following")
        return usernames

    async def scrape_followers_async(self, username: str, limit: int = 100) -> list[str]:
        """Scrape usernames that follow the target user (async)."""
        usernames: list[str] = []
        page = 1

        logger.info(f"Scraping {username}'s followers (async)...")
        while len(usernames) < limit:
            tree = await self._get_with_soft_block_recovery(f"{self.BASE}/{username}/followers/page/{page}/")
            if not tree:
                break

            links = tree.css("a.name")
            if not links:
                break

            for link in links:
                href = link.attributes.get("href", "")
                if href.startswith("/") and href.endswith("/"):
                    user = href.strip("/")
                    if user and user not in usernames:
                        usernames.append(user)
                        if len(usernames) >= limit:
                            break

            page += 1

        logger.info(f"  Found {len(usernames)} followers")
        return usernames

    async def scrape_films_batch(self, slugs: list[str]) -> list[FilmMetadata]:
        """
        Scrape multiple films concurrently.

        This method can be called either:
        1. With the async context manager: async with AsyncLetterboxdScraper() as scraper
        2. Standalone (creates and cleans up a temporary client automatically)
        """
        if self.client:
            # Use existing client from context manager
            return await self._scrape_batch_with_client(self.client, slugs)

        # Create temporary client - async with ensures cleanup even on exceptions
        async with httpx.AsyncClient(
            headers={"User-Agent": "Mozilla/5.0 (compatible; letterboxd-rec/1.0)"},
            follow_redirects=True,
            timeout=HTTP_TIMEOUT,
            http2=SCRAPER_HTTP2,
            cookies=self.cookies or None,
        ) as temp_client:
            return await self._scrape_batch_with_client(temp_client, slugs)

    async def _scrape_batch_with_client(self, client: httpx.AsyncClient, slugs: list[str]) -> list[FilmMetadata]:
        """
        Internal method to scrape batch with provided client.

        Returns list of successfully scraped FilmMetadata objects.
        Failures are logged with comprehensive error information.
        """
        tasks = [self._scrape_film_async(client, slug) for slug in slugs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Track successes and failures
        successful = []
        failed = []
        error_summary = {}

        for slug, result in zip(slugs, results):
            if isinstance(result, Exception):
                error_type = type(result).__name__
                error_msg = str(result)
                logger.error(f"Failed to scrape {slug}: {error_type}: {error_msg}")
                failed.append(slug)

                # Aggregate error types for summary
                if error_type not in error_summary:
                    error_summary[error_type] = []
                error_summary[error_type].append(slug)
            elif result is not None:
                successful.append(result)
            else:
                # None result (e.g., 404 or max retries exceeded)
                logger.debug(f"No result for {slug} (likely 404 or max retries)")
                failed.append(slug)

        # Log summary
        if failed:
            logger.warning(
                f"Batch complete: {len(successful)}/{len(slugs)} successful, {len(failed)} failed"
            )
            if error_summary:
                logger.info(f"Error breakdown: {dict((k, len(v)) for k, v in error_summary.items())}")
        else:
            logger.info(f"Batch complete: {len(successful)}/{len(slugs)} successful")

        return successful

    async def _fetch_fan_count_async(self, client: httpx.AsyncClient, slug: str) -> int | None:
        """Async helper to fetch fan count from ratings-summary CSI endpoint."""
        try:
            resp = await client.get(f"{self.BASE}/csi/film/{slug}/ratings-summary/")
            if resp.status_code != 200:
                return None
            return parse_fan_count_from_html(resp.text)
        except httpx.HTTPError as exc:
            logger.debug(f"Fan count fetch failed for {slug}: {exc}")
            return None

    async def _scrape_film_async(self, client: httpx.AsyncClient, slug: str) -> FilmMetadata | None:
        """Scrape a single film asynchronously with proper retry logic and coordinated rate limiting."""
        async with self.semaphore:
            await asyncio.sleep(self.delay)

            for attempt in range(MAX_HTTP_RETRIES):
                # Wait if globally rate limited by another task
                await self._rate_limit_event.wait()

                try:
                    resp = await client.get(f"{self.BASE}/film/{slug}/")

                    if resp.status_code == 404:
                        logger.debug(f"Film not found: {slug}")
                        return None

                    if resp.status_code == 429:
                        retry_after = int(resp.headers.get("Retry-After", DEFAULT_RETRY_AFTER))
                        logger.warning(
                            f"Rate limited on {slug}, pausing ALL tasks for {retry_after}s "
                            f"(attempt {attempt + 1}/{MAX_HTTP_RETRIES})"
                        )

                        # Pause all concurrent tasks
                        self._rate_limit_event.clear()
                        await asyncio.sleep(retry_after)
                        # Resume all tasks
                        self._rate_limit_event.set()
                        # Add jitter to prevent thundering herd
                        import random

                        jitter = random.uniform(0, self.delay * 2)
                        await asyncio.sleep(jitter)
                        # Adjust delay upwards for future calls
                        self.delay = min(SCRAPER_ADAPTIVE_DELAY_MAX, self.delay * SCRAPER_429_BACKOFF)
                        continue

                    resp.raise_for_status()
                    tree = HTMLParser(resp.text)
                    meta = parse_film_page(tree, slug)
                    meta.fan_count = await self._fetch_fan_count_async(client, slug)
                    return meta

                except httpx.TimeoutException:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Timeout on {slug}, retrying in {wait_time}s (attempt {attempt + 1}/{MAX_HTTP_RETRIES})"
                    )
                    await asyncio.sleep(wait_time)

                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP {e.response.status_code} on {slug}: {e}")
                    return None

                except httpx.HTTPError as e:
                    logger.error(f"Request error on {slug}: {type(e).__name__}: {e}")
                    return None

            logger.error(f"Max retries exceeded for {slug}")
            return None
