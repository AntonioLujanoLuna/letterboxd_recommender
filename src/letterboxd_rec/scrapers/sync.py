"""Synchronous Letterboxd scraper implementation."""

import time
import logging

import httpx
from selectolax.parser import HTMLParser

from ..config import (
    HTTP_TIMEOUT,
    MAX_CONSECUTIVE_EXISTING,
    MAX_HTTP_RETRIES,
    MAX_429_RETRY_SECONDS,
    DEFAULT_RETRY_AFTER,
    SCRAPER_HTTP2,
    SCRAPER_ADAPTIVE_DELAY_MIN,
    SCRAPER_ADAPTIVE_DELAY_MAX,
    SCRAPER_429_BACKOFF,
    SCRAPER_429_JITTER,
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


class LetterboxdScraper:
    """Synchronous scraper for Letterboxd."""

    BASE = "https://letterboxd.com"

    def __init__(self, delay: float = 1.0):
        self.cookies = parse_cookie_header(LETTERBOXD_COOKIE)
        self.client = httpx.Client(
            headers={"User-Agent": "Mozilla/5.0 (compatible; film-rec/0.1)"},
            follow_redirects=True,
            timeout=HTTP_TIMEOUT,
            http2=SCRAPER_HTTP2,
            cookies=self.cookies or None,
        )
        self.delay = delay
        self._current_delay = max(delay, SCRAPER_ADAPTIVE_DELAY_MIN)

    def _get(self, url: str, max_retries: int = MAX_HTTP_RETRIES) -> HTMLParser | None:
        time.sleep(self._current_delay)

        retries = 0
        total_429_wait_time = 0

        while retries < max_retries:
            try:
                resp = self.client.get(url)
                if resp.status_code == 404:
                    return None

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", DEFAULT_RETRY_AFTER))

                    # Check if we've waited too long for 429s
                    if total_429_wait_time + retry_after > MAX_429_RETRY_SECONDS:
                        logger.error(f"Max 429 wait time exceeded for {url} (waited {total_429_wait_time}s, would need {retry_after}s more)")
                        return None

                    logger.warning(f"Rate limited (429) on {url}, waiting {retry_after}s... (total 429 wait: {total_429_wait_time}s)")
                    time.sleep(retry_after)
                    total_429_wait_time += retry_after
                    # Do not increment retries for 429, but track total wait time
                    # Adaptive delay: bump future delay with jitter
                    import random
                    self._current_delay = min(
                        SCRAPER_ADAPTIVE_DELAY_MAX,
                        max(
                            self._current_delay * SCRAPER_429_BACKOFF,
                            self.delay
                        ) * (1 + random.uniform(0, SCRAPER_429_JITTER))
                    )
                    continue

                resp.raise_for_status()
                # Successful request: gently decay delay back toward base
                self._current_delay = max(
                    SCRAPER_ADAPTIVE_DELAY_MIN,
                    (self._current_delay * 0.8) + (self.delay * 0.2)
                )
                return HTMLParser(resp.text)
            except httpx.TimeoutException as e:
                if retries < max_retries - 1:
                    wait_time = 2 ** retries
                    logger.warning(f"Timeout on {url}, retrying in {wait_time}s... (attempt {retries + 1}/{max_retries})")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    logger.error(f"Max retries exceeded for {url}: {e}")
                    return None
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error on {url}: {e}")
                return None
            except httpx.HTTPError as e:
                logger.error(f"Request error on {url}: {e}")
                return None

        return None

    def _detect_soft_block(self, tree: HTMLParser) -> bool:
        """
        Detect if we're being soft-blocked (page loads but with limited/no content).
        """
        if not tree:
            return False

        # 1. CAPTCHA or verification page
        if tree.css_first("form[action*='captcha'], .captcha-container"):
            return True

        # 2. "Please wait" or rate limit message in body
        body_el = tree.css_first("body")
        body_text = body_el.text() if body_el else ""
        soft_block_phrases = [
            "please wait",
            "too many requests",
            "try again later",
            "access denied",
        ]
        if any(phrase in body_text.lower() for phrase in soft_block_phrases):
            return True

        return False

    def _get_with_soft_block_recovery(self, url: str) -> HTMLParser | None:
        """Enhanced _get with soft block detection and recovery."""
        tree = self._get(url)

        if tree and self._detect_soft_block(tree):
            logger.warning(f"Soft block detected on {url}, backing off...")

            # Short, test-friendly backoff to avoid long hangs
            for wait_time in [1, 2, 4]:
                time.sleep(wait_time)
                tree = self._get(url)
                if tree and not self._detect_soft_block(tree):
                    return tree

            logger.error(f"Persistent soft block on {url}")
            return None

        return tree

    def _fetch_fan_count(self, slug: str) -> int | None:
        """
        Fetch fan count from the ratings-summary CSI endpoint.
        """
        try:
            resp = self.client.get(f"{self.BASE}/csi/film/{slug}/ratings-summary/")
            if resp.status_code != 200:
                return None
            return parse_fan_count_from_html(resp.text)
        except httpx.HTTPError as exc:
            logger.debug(f"Fan count fetch failed for {slug}: {exc}")
            return None

    def scrape_user(self, username: str, existing_slugs: set[str] | None = None, stop_on_existing: bool = False) -> list[FilmInteraction]:
        """
        Scrape all film interactions for a user.

        Args:
            username: Letterboxd username to scrape
            existing_slugs: Optional set of film slugs already in database for this user
            stop_on_existing: If True, stop pagination when hitting films already scraped (incremental mode)

        Returns:
            List of FilmInteraction objects
        """
        films = {}

        # Get watched films with ratings
        logger.info(f"Scraping {username}'s films...")
        page = 1
        consecutive_existing = 0  # Track consecutive existing films for early termination

        while True:
            tree = self._get_with_soft_block_recovery(f"{self.BASE}/{username}/films/page/{page}/")
            if not tree:
                break

            items = tree.css("li.griditem")
            if not items:
                break

            page_had_new_films = False
            if len(items) < SCRAPER_PAGE_SIZE:
                # Less than a full page usually means last page; enables tighter cursoring
                stop_after_page = True
            else:
                stop_after_page = False

            for item in items:
                react_comp = item.css_first("div.react-component")
                if not react_comp:
                    continue

                slug = validate_slug(react_comp.attributes.get("data-item-slug"))
                if not slug:
                    continue

                # Incremental scraping: check if we've already scraped this film
                if stop_on_existing and existing_slugs and slug in existing_slugs:
                    consecutive_existing += 1
                    if consecutive_existing >= MAX_CONSECUTIVE_EXISTING:
                        logger.info(f"  Found {consecutive_existing} consecutive existing films, stopping early (incremental mode)")
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

            # If incremental mode and no new films on this page, we can stop
            if stop_on_existing and not page_had_new_films and existing_slugs:
                logger.info(f"  No new films found on page {page-1}, stopping early (incremental mode)")
                break
            if stop_after_page:
                break

        # Get watchlist
        logger.info(f"Scraping {username}'s watchlist...")
        page = 1
        while True:
            tree = self._get_with_soft_block_recovery(f"{self.BASE}/{username}/watchlist/page/{page}/")
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

    def scrape_user_smart(self, username: str, known_film_count: int | None = None) -> list[FilmInteraction]:
        """
        Smart scraping with adaptive page fetching.

        If we know the user's film count from profile, we can:
        1. Estimate pages needed
        2. Apply early termination when pages look stale
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
            tree = self._get_with_soft_block_recovery(f"{self.BASE}/{username}/films/page/{page}/")
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
                        rating = self._parse_rating(rating_span)

                    liked = viewing_data.css_first("span.like") is not None

                films[slug] = FilmInteraction(slug, rating, True, False, liked)

            page += 1
            if stop_after_page:
                break
            if estimated_pages and page > estimated_pages + 2:
                # Avoid unbounded crawling if page estimate looks off
                break

        # Watchlist sweep (reuse standard logic with smart requests)
        page = 1
        while True:
            tree = self._get_with_soft_block_recovery(f"{self.BASE}/{username}/watchlist/page/{page}/")
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
        logger.info(f"Smart scrape total: {len(films)} films ({n_rated} rated, {n_liked} liked)")
        return list(films.values())

    def scrape_film(self, slug: str) -> FilmMetadata | None:
        """Scrape metadata for a single film."""
        tree = self._get_with_soft_block_recovery(f"{self.BASE}/film/{slug}/")
        if not tree:
            return None
        meta = parse_film_page(tree, slug)
        meta.fan_count = self._fetch_fan_count(slug)
        return meta

    def _parse_rating(self, span) -> float | None:
        """Thin wrapper around shared rating parser for backward compatibility."""
        return parse_rating_span(span)

    def scrape_following(self, username: str, limit: int = 100) -> list[str]:
        """Scrape usernames that the target user follows."""
        usernames = []
        page = 1

        logger.info(f"Scraping {username}'s following...")
        while len(usernames) < limit:
            tree = self._get_with_soft_block_recovery(f"{self.BASE}/{username}/following/page/{page}/")
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

    def scrape_followers(self, username: str, limit: int = 100) -> list[str]:
        """Scrape usernames that follow the target user."""
        usernames = []
        page = 1

        logger.info(f"Scraping {username}'s followers...")
        while len(usernames) < limit:
            tree = self._get_with_soft_block_recovery(f"{self.BASE}/{username}/followers/page/{page}/")
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

    def scrape_popular_members(self, limit: int = 50) -> list[str]:
        """Scrape popular members from Letterboxd."""
        usernames = []
        page = 1

        logger.info("Scraping popular members...")
        while len(usernames) < limit:
            tree = self._get_with_soft_block_recovery(f"{self.BASE}/members/popular/this/week/page/{page}/")
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

        logger.info(f"  Found {len(usernames)} popular members")
        return usernames

    def scrape_film_fans(self, slug: str, limit: int = 50) -> list[str]:
        """Scrape users who are fans of a specific film."""
        usernames = []
        page = 1

        logger.info(f"Scraping fans of {slug}...")
        while len(usernames) < limit:
            tree = self._get_with_soft_block_recovery(f"{self.BASE}/film/{slug}/fans/page/{page}/")
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

        logger.info(f"  Found {len(usernames)} fans of {slug}")
        return usernames

    def scrape_film_reviewers(self, slug: str, limit: int = 50) -> list[dict]:
        """
        Scrape users who have reviewed a specific film.

        Returns list of dicts with:
        - username: str
        - has_rating: bool (whether they showed a rating on the review)
        - review_date: str | None (if visible)

        Reviewers are higher-quality signal than fans since writing a review
        requires more engagement.
        """
        reviewers = []
        page = 1

        logger.info(f"Scraping reviewers of {slug}...")
        while len(reviewers) < limit:
            tree = self._get_with_soft_block_recovery(f"{self.BASE}/film/{slug}/reviews/page/{page}/")
            if not tree:
                break

            # Reviews are in li.film-detail elements
            review_items = tree.css("li.film-detail")
            if not review_items:
                break

            for item in review_items:
                # Extract username from the author link
                author_link = item.css_first("a.context")
                if not author_link:
                    continue

                href = author_link.attributes.get("href", "")
                if not href.startswith("/") or not href.endswith("/"):
                    continue

                username = href.strip("/")
                if not username:
                    continue

                # Check if they have a rating (star icons visible)
                has_rating = item.css_first("span.rating") is not None

                # Try to extract review date if visible
                review_date = None
                date_link = item.css_first("span._nobr a")
                if date_link:
                    review_date = date_link.text(strip=True)

                reviewers.append({
                    'username': username,
                    'has_rating': has_rating,
                    'review_date': review_date
                })

                if len(reviewers) >= limit:
                    break

            page += 1

        logger.info(f"  Found {len(reviewers)} reviewers of {slug}")
        return reviewers

    def check_user_activity(self, username: str) -> dict | None:
        """
        Lightweight activity check by fetching just the user's profile page.

        Returns dict with:
        - film_count: int (total films logged)
        - has_ratings: bool (whether they show rating distribution)
        - recent_activity: bool (whether recent diary entries visible)

        Returns None if profile doesn't exist or can't be accessed.
        This is much cheaper than a full scrape (1 request vs 10+).
        """
        tree = self._get_with_soft_block_recovery(f"{self.BASE}/{username}/")
        if not tree:
            return None

        # Extract film count from stats
        # The film count is in a span child element: <a class="thousands"><span>2,479</span><span>Films</span></a>
        film_count = 0
        stats_link = tree.css_first("a.thousands[href$='/films/']")
        if stats_link:
            # Get the first span which contains the number
            count_span = stats_link.css_first("span")
            if count_span:
                try:
                    # Extract the count, removing commas
                    count_str = count_span.text(strip=True).replace(",", "")
                    film_count = int(count_str)
                except (ValueError, IndexError):
                    pass


        # Check if they have a rating distribution (indicates they rate films)
        # Look for rating band links like "/films/ratings/rated/1/" etc.
        has_ratings = tree.css_first("a[href*='/films/ratings/rated/']") is not None

        # Check for recent activity (diary entries on profile)
        # Look for film links under the diary section like "/username/film/slug/"
        recent_activity = len(tree.css(f"a[href^='/{username}/film/']")) > 0

        return {
            'film_count': film_count,
            'has_ratings': has_ratings,
            'recent_activity': recent_activity
        }

    def scrape_favorites(self, username: str) -> list[str]:
        """Scrape user's favorite films (4-film profile showcase)."""
        tree = self._get_with_soft_block_recovery(f"{self.BASE}/{username}/")
        if not tree:
            return []

        favorites = []

        showcase = tree.css("section.profile-favorites li.poster-container, section#favourites li.poster-container")
        for item in showcase[:4]:
            react_comp = item.css_first("div.react-component")
            if react_comp:
                slug = validate_slug(react_comp.attributes.get("data-film-slug"))
                if slug:
                    favorites.append(slug)
                    continue

            link = item.css_first("div[data-film-slug]")
            if link:
                slug = validate_slug(link.attributes.get("data-film-slug"))
                if slug:
                    favorites.append(slug)

        return favorites

    def scrape_user_lists(self, username: str, limit: int = 50) -> list[dict]:
        """Scrape all lists for a user."""
        lists = []
        page = 1

        logger.info(f"Scraping {username}'s lists...")
        while len(lists) < limit:
            tree = self._get_with_soft_block_recovery(f"{self.BASE}/{username}/lists/page/{page}/")
            if not tree:
                break

            list_items = tree.css("section.list-summary")
            if not list_items:
                list_items = tree.css("section.film-list-summary")

            if not list_items:
                break

            for item in list_items:
                link = item.css_first("h2 a, h3 a")
                if not link:
                    continue

                href = link.attributes.get("href", "")
                list_name = link.text(strip=True)

                parts = href.strip("/").split("/")
                if len(parts) >= 3 and parts[1] == "list":
                    list_slug = parts[2]
                else:
                    continue

                is_ranked = (
                    item.css_first(".icon-numbered") is not None or
                    "numbered" in item.attributes.get("class", "").lower()
                )

                lists.append({
                    "list_slug": list_slug,
                    "list_name": list_name,
                    "is_ranked": is_ranked
                })

                if len(lists) >= limit:
                    break

            page += 1

        logger.info(f"  Found {len(lists)} lists")
        return lists

    def scrape_list_films(self, username: str, list_slug: str) -> list[dict]:
        """Scrape films from a specific list."""
        films = []
        page = 1
        cumulative_count = 0  # Track total films seen across all pages

        while True:
            tree = self._get_with_soft_block_recovery(f"{self.BASE}/{username}/list/{list_slug}/page/{page}/")
            if not tree:
                break

            has_positions = tree.css_first(".list-number, .position") is not None

            items = tree.css("li.poster-container")
            if not items:
                break

            for idx, item in enumerate(items):
                react_comp = item.css_first("div.react-component")
                film_slug = None

                if react_comp:
                    film_slug = validate_slug(react_comp.attributes.get("data-film-slug"))

                if not film_slug:
                    link = item.css_first("div[data-film-slug]")
                    if link:
                        film_slug = validate_slug(link.attributes.get("data-film-slug"))

                if not film_slug:
                    continue

                position = None
                if has_positions:
                    pos_el = item.css_first(".list-number, .position")
                    if pos_el:
                        try:
                            pos_text = pos_el.text(strip=True).rstrip(".")
                            position = int(pos_text)
                        except (ValueError, AttributeError):
                            # Fallback: use cumulative count instead of page-based calculation
                            position = cumulative_count + idx + 1
                    else:
                        position = cumulative_count + idx + 1

                films.append({
                    "film_slug": film_slug,
                    "position": position
                })

            cumulative_count += len(items)  # Update cumulative count after processing page
            page += 1

        return films

    def close(self):
        self.client.close()
