import httpx
import json
import time
import logging
import re
from selectolax.parser import HTMLParser
from dataclasses import dataclass
from tqdm import tqdm
import asyncio
from .config import (
    SCRAPER_MAX_CAST,
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
)

logger = logging.getLogger(__name__)


def validate_slug(slug: str | None) -> str | None:
    """
    Validate film slug format to prevent injection or malformed data.

    Returns cleaned slug or None if invalid.
    Letterboxd slugs are typically lowercase alphanumeric with hyphens, but
    some endpoints now emit a namespaced format like 'film:482919'. We allow
    that prefix while still validating the core slug characters.
    """
    if not slug:
        return None

    cleaned = slug.strip().lower()

    prefix = ""
    core = cleaned
    if core.startswith("film:"):
        prefix = "film:"
        core = core.split(":", 1)[1]

    # Require alphanumeric/hyphen core even when prefixed with "film:"
    if not core or not re.match(r'^[a-z0-9-]+$', core):
        logger.warning(f"Invalid slug format (contains disallowed characters): '{slug}'")
        return None

    full_slug = prefix + core

    # Additional safety: reject excessively long slugs (Letterboxd slugs are typically < 100 chars)
    if len(full_slug) > 200:
        logger.warning(f"Slug exceeds maximum length: '{slug[:50]}...'")
        return None

    return full_slug


def _parse_rating_count(text: str) -> int | None:
    """
    Parse rating count from text like '1.5M', '500K', '1.2B', or '12,345'.
    Returns integer count or None if parsing fails.
    """
    text = text.strip().replace(",", "")
    multipliers = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000}

    for suffix, mult in multipliers.items():
        if text.endswith(suffix):
            try:
                return int(float(text[:-1]) * mult)
            except ValueError:
                return None

    try:
        return int(text) if text.isdigit() else None
    except ValueError:
        return None


@dataclass
class FilmInteraction:
    film_slug: str
    rating: float | None
    watched: bool
    watchlisted: bool
    liked: bool

@dataclass
class FilmMetadata:
    slug: str
    title: str
    year: int | None
    directors: list[str]
    genres: list[str]
    cast: list[str]
    themes: list[str]
    runtime: int | None
    avg_rating: float | None
    rating_count: int | None
    countries: list[str]
    languages: list[str]
    writers: list[str]
    cinematographers: list[str]
    composers: list[str]


def parse_film_page(tree: HTMLParser, slug: str) -> FilmMetadata:
    """
    Shared parsing logic for film pages.
    Used by both sync and async scrapers to avoid code duplication.
    """
    # Title
    title_el = tree.css_first("h1.headline-1")
    title = title_el.text(strip=True) if title_el else slug

    # Year (Letterboxd markup drifts; try several fallbacks)
    year = None
    year_el = tree.css_first("small.number a, div.releaseyear a")
    if year_el:
        try:
            year = int(year_el.text(strip=True))
        except ValueError:
            year = None
    if year is None:
        alt_year = tree.css_first("a[href*='/films/year/']")
        if alt_year:
            text = alt_year.text(strip=True)
            m = re.search(r"(19|20|21)\d{2}", text)
            if m:
                year = int(m.group(0))
    if year is None:
        # As a last resort, try to parse from og:title like "Barbie (2023)"
        og_title = tree.css_first("meta[property='og:title']")
        if og_title:
            content = og_title.attributes.get("content", "")
            m = re.search(r"(19|20|21)\d{2}", content)
            if m:
                year = int(m.group(0))

    # Directors
    directors = list(dict.fromkeys([a.text(strip=True) for a in tree.css("a[href*='/director/']") if a.text(strip=True)]))

    # Genres (normalize to lowercase for consistent matching)
    genres = list(dict.fromkeys([a.text(strip=True).lower() for a in tree.css("a[href*='/films/genre/']") if a.text(strip=True)]))

    # Cast (top billed)
    cast = list(dict.fromkeys([a.text(strip=True) for a in tree.css("a[href*='/actor/']")[:SCRAPER_MAX_CAST] if a.text(strip=True)]))

    # Themes/tags
    themes = list(dict.fromkeys([a.text(strip=True) for a in tree.css("a[href*='/films/theme/'], a[href*='/films/mini-theme/']") if a.text(strip=True)]))

    # Runtime
    runtime = None
    runtime_el = tree.css_first("p.text-link.text-footer")
    if runtime_el and "mins" in runtime_el.text():
        try:
            runtime = int(runtime_el.text().split()[0])
        except (ValueError, IndexError):
            pass

    # Average rating
    avg_rating = None
    meta = tree.css_first("meta[name='twitter:data2']")
    if meta:
        try:
            avg_rating = float(meta.attributes.get("content", "").split()[0])
        except (ValueError, IndexError):
            pass

    # Rating count
    rating_count = None
    ratings_el = tree.css_first("a[href*='/ratings/']")
    if ratings_el:
        rating_count = _parse_rating_count(ratings_el.text(strip=True))

    # Countries
    countries = list(dict.fromkeys([a.text(strip=True) for a in tree.css("a[href*='/films/country/']") if a.text(strip=True)]))

    # Languages
    languages = list(dict.fromkeys([a.text(strip=True) for a in tree.css("a[href*='/films/language/']") if a.text(strip=True)]))

    # Writers
    writers = list(dict.fromkeys([a.text(strip=True) for a in tree.css("a[href*='/writer/']") if a.text(strip=True)]))

    # Cinematographers
    cinematographers = list(dict.fromkeys([a.text(strip=True) for a in tree.css("a[href*='/cinematography/']") if a.text(strip=True)]))

    # Composers
    composers = list(dict.fromkeys([a.text(strip=True) for a in tree.css("a[href*='/composer/']") if a.text(strip=True)]))

    return FilmMetadata(
        slug=slug, title=title, year=year, directors=directors,
        genres=genres, cast=cast, themes=themes,
        runtime=runtime, avg_rating=avg_rating, rating_count=rating_count,
        countries=countries, languages=languages, writers=writers,
        cinematographers=cinematographers, composers=composers
    )


class LetterboxdScraper:
    BASE = "https://letterboxd.com"
    
    def __init__(self, delay: float = 1.0):
        self.client = httpx.Client(
            headers={"User-Agent": "Mozilla/5.0 (compatible; film-rec/0.1)"},
            follow_redirects=True,
            timeout=HTTP_TIMEOUT,
            http2=SCRAPER_HTTP2,
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
        return parse_film_page(tree, slug)
    
    def _parse_rating(self, span) -> float | None:
        """
        Parse rating from span element with class like 'rated-8' (representing 4.0 stars).

        Returns rating float or None if parsing fails.
        Logs warning if unexpected format detected to catch Letterboxd HTML changes.
        """
        classes = span.attributes.get("class", "")
        for cls in classes.split():
            if cls.startswith("rated-"):
                try:
                    val = int(cls.replace("rated-", "")) / 2
                    if 0.5 <= val <= 5.0:
                        return val
                    else:
                        # Rating outside expected range - log for investigation
                        logger.warning(f"Rating value outside range [0.5-5.0]: {val} from class '{cls}'")
                        return None
                except ValueError as e:
                    # Non-integer after "rated-" prefix - unexpected format
                    logger.warning(f"Unexpected rating format in class '{cls}': {e}")
                    return None

        # No 'rated-*' class found - might indicate HTML structure change
        if classes:
            logger.debug(f"No 'rated-*' class found in span classes: {classes}")
        return None
    
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


class AsyncLetterboxdScraper:
    """Async scraper for parallel metadata fetching with coordinated rate limiting."""

    BASE = "https://letterboxd.com"

    def __init__(self, delay: float = 0.2, max_concurrent: int = 5):
        self.delay = delay
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.client = None
        # Coordinated rate limiting: when one task hits 429, all tasks pause
        self._rate_limit_event = asyncio.Event()
        self._rate_limit_event.set()  # Start in "not rate limited" state

    @staticmethod
    def _parse_rating(span) -> float | None:
        """
        Parse rating from span element with class like 'rated-8' (representing 4.0 stars).

        Mirrors the synchronous scraper's logic so async user scraping can share the same format handling.
        """
        classes = span.attributes.get("class", "")
        for cls in classes.split():
            if cls.startswith("rated-"):
                try:
                    val = int(cls.replace("rated-", "")) / 2
                    if 0.5 <= val <= 5.0:
                        return val
                    logger.warning(f"Rating value outside range [0.5-5.0]: {val} from class '{cls}'")
                    return None
                except ValueError as exc:
                    logger.warning(f"Unexpected rating format in class '{cls}': {exc}")
                    return None

        if classes:
            logger.debug(f"No 'rated-*' class found in span classes: {classes}")
        return None

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
                    return parse_film_page(tree, slug)

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
