import httpx
import json
import time
import logging
import re
from selectolax.parser import HTMLParser
from dataclasses import dataclass
from tqdm import tqdm
import asyncio
from contextlib import aclosing
from .scrape_stats import TimedAsyncClient, TimedClient
from .rate_control import AimdRate
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
    LETTERBOXD_COOKIE,
)

logger = logging.getLogger(__name__)


class ScrapeIncompleteError(RuntimeError):
    """A user's watched-films listing could not be fetched (blocked, 403, network error).

    Raised instead of returning a partial result: saving only a watchlist would look like a
    user who has watched nothing, which silently corrupts recommendations and evaluation.
    """

    def __init__(self, username: str, url: str):
        self.username = username
        self.url = url
        super().__init__(
            f"Could not fetch {username}'s watched films ({url}). Letterboxd may be blocking "
            "automated requests; nothing was saved for this user."
        )


def _looks_private(text: str | None) -> bool:
    """Letterboxd's 403 for private/restricted pages ("classified"), as opposed to a block."""
    t = (text or "").lower()
    return "classified" in t or "falken" in t


def _parse_cookie_header(cookie_header: str | None) -> dict[str, str]:
    """
    Convert a raw Cookie header string into a dict for httpx.

    Accepts the full "key=value; key2=value2" header; ignores malformed pairs.
    """
    if not cookie_header:
        return {}

    cookie_jar: dict[str, str] = {}
    for part in cookie_header.split(";"):
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        name = name.strip()
        value = value.strip()
        if name:
            cookie_jar[name] = value
    return cookie_jar


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


def _parse_rating_span(span) -> float | None:
    """
    Parse a rating from a span element with a class like 'rated-8' (4.0 stars).

    Shared by sync and async scrapers to avoid duplicated logic.
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


def _parse_fan_count_from_html(html: str) -> int | None:
    """
    Extract fan count from the ratings-summary CSI snippet.
    Looks for patterns like '26K fans' or '4,876,723 fans'.
    """
    match = re.search(r"([0-9][0-9,.\u202fKMB]+)\s+fans", html, re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1)
    cleaned = raw.replace("\u202f", "").replace(" ", "")
    return _parse_rating_count(cleaned)


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
    fan_count: int | None
    countries: list[str]
    languages: list[str]
    writers: list[str]
    cinematographers: list[str]
    composers: list[str]
    is_short: bool
    is_animation: bool
    tmdb_id: int | None = None   # joins to MovieLens links.csv / TMDB
    imdb_id: str | None = None   # e.g. "tt6751668"


def parse_film_page(tree: HTMLParser, slug: str) -> FilmMetadata:
    """
    Shared parsing logic for film pages.
    Used by both sync and async scrapers to avoid code duplication.
    """
    # Structured data (used for stable rating_count / avg_rating parsing)
    aggregate_rating: dict = {}
    ldjson = tree.css_first("script[type='application/ld+json']")
    if ldjson:
        raw = ldjson.text()
        # Remove comment wrappers defensively (Letterboxd wraps JSON in /* <![CDATA[ */ ... /* ]]> */)
        cleaned = re.sub(r"/\*.*?\*/", "", raw, flags=re.S).strip()

        parsed_obj = None
        for candidate in (raw.strip(), cleaned):
            try:
                parsed_obj = json.loads(candidate)
            except json.JSONDecodeError as exc:  # noqa: PERF203 (clarity)
                logger.debug(f"Failed to parse ld+json for {slug}: {exc}")
                continue

            if isinstance(parsed_obj, list):
                parsed_obj = parsed_obj[0] if parsed_obj else {}
            if isinstance(parsed_obj, dict):
                aggregate_rating = parsed_obj.get("aggregateRating") or {}
                break

        if not aggregate_rating:
            rating_value_match = re.search(r'"ratingValue"\s*:\s*([0-9.]+)', raw)
            rating_count_match = re.search(r'"ratingCount"\s*:\s*([\d,]+)', raw)
            if rating_value_match or rating_count_match:
                aggregate_rating = {}
                if rating_value_match:
                    aggregate_rating["ratingValue"] = rating_value_match.group(1)
                if rating_count_match:
                    aggregate_rating["ratingCount"] = rating_count_match.group(1).replace(",", "")

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

    is_animation = "animation" in genres
    is_short = False
    if runtime is not None:
        is_short = runtime <= 40
    if not is_short and "short" in genres:
        is_short = True

    # Average rating
    avg_rating = None
    meta = tree.css_first("meta[name='twitter:data2']")
    if meta:
        try:
            avg_rating = float(meta.attributes.get("content", "").split()[0])
        except (ValueError, IndexError):
            pass

    if avg_rating is None and aggregate_rating:
        try:
            avg_rating = float(aggregate_rating.get("ratingValue"))
        except (TypeError, ValueError):
            pass

    # Rating count
    rating_count = None
    ratings_el = tree.css_first("a[href*='/ratings/']")
    if ratings_el:
        rating_count = _parse_rating_count(ratings_el.text(strip=True))

    if rating_count is None and aggregate_rating:
        try:
            rating_count = int(aggregate_rating.get("ratingCount"))
        except (TypeError, ValueError):
            rating_count = None

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

    # External IDs (exact joins to MovieLens/TMDB/IMDb)
    tmdb_id = None
    body = tree.css_first("body")
    raw_tmdb = body.attributes.get("data-tmdb-id") if body else None
    if raw_tmdb and str(raw_tmdb).isdigit() and (body.attributes.get("data-tmdb-type") or "movie") == "movie":
        tmdb_id = int(raw_tmdb)
    imdb_id = None
    imdb_link = tree.css_first("a[data-track-action='IMDb'], a[href*='imdb.com/title/']")
    if imdb_link:
        m = re.search(r"(tt\d+)", imdb_link.attributes.get("href") or "")
        imdb_id = m.group(1) if m else None

    # Fan count shown on the film page ("76K fans"); the old CSI endpoint no longer exists.
    fan_count = None
    fans_link = tree.css_first("a[href$='/fans/']")
    if fans_link:
        fan_count = _parse_fan_count_from_html(fans_link.text())

    return FilmMetadata(
        slug=slug, title=title, year=year, directors=directors,
        genres=genres, cast=cast, themes=themes,
        runtime=runtime, avg_rating=avg_rating, rating_count=rating_count,
        fan_count=fan_count,
        countries=countries, languages=languages, writers=writers,
        cinematographers=cinematographers, composers=composers,
        is_short=is_short, is_animation=is_animation,
        tmdb_id=tmdb_id, imdb_id=imdb_id,
    )


class LetterboxdScraper:
    BASE = "https://letterboxd.com"
    
    def __init__(self, delay: float = 1.0):
        self.cookies = _parse_cookie_header(LETTERBOXD_COOKIE)
        self.client = httpx.Client(
            headers={"User-Agent": "Mozilla/5.0 (compatible; film-rec/0.1)"},
            follow_redirects=True,
            timeout=HTTP_TIMEOUT,
            http2=SCRAPER_HTTP2,
            cookies=self.cookies or None,
        )
        self.client = TimedClient(self.client)
        # URLs whose last fetch failed for a transient/unknown reason (not 404, not private)
        self._failed: set[str] = set()
        self.delay = delay
        self._current_delay = max(delay, SCRAPER_ADAPTIVE_DELAY_MIN)

    def _get(self, url: str, max_retries: int = MAX_HTTP_RETRIES) -> HTMLParser | None:
        self._failed.discard(url)
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
                        self._failed.add(url)
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
                    self._failed.add(url)
                    return None
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error on {url}: {e}")
                if not (e.response.status_code == 403 and _looks_private(e.response.text)):
                    self._failed.add(url)
                return None
            except httpx.HTTPError as e:
                logger.error(f"Request error on {url}: {e}")
                self._failed.add(url)
                return None
        
        self._failed.add(url)
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
            self._failed.add(url)
            return None

        return tree

    def _get_listing_page(self, url: str, attempts: int = 3) -> HTMLParser | None:
        """Fetch one page of a paginated listing, retrying transient failures.

        Returns None only when the page genuinely isn't there (404 / private); if it is still
        failing after the retries it also returns None, but `url` stays in `self._failed` so the
        caller can tell the two apart.
        """
        tree = None
        for attempt in range(attempts):
            tree = self._get_with_soft_block_recovery(url)
            if tree is not None or url not in self._failed:
                return tree
            if attempt < attempts - 1:
                time.sleep(2 * (attempt + 1) ** 2)  # 2s, 8s
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
            films_url = f"{self.BASE}/{username}/films/page/{page}/"
            tree = self._get_listing_page(films_url)
            if not tree:
                # Past-the-end pages come back 200 with an empty grid, so a missing page is a
                # failure. Stopping here would silently keep a truncated profile.
                raise ScrapeIncompleteError(username, films_url)

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
            watch_url = f"{self.BASE}/{username}/watchlist/page/{page}/"
            tree = self._get_listing_page(watch_url)
            if not tree:
                if page > 1 or watch_url in self._failed:
                    raise ScrapeIncompleteError(username, watch_url)
                break  # private or missing watchlist
            
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
            films_url = f"{self.BASE}/{username}/films/page/{page}/"
            tree = self._get_listing_page(films_url)
            if not tree:
                # Past-the-end pages come back 200 with an empty grid, so a missing page is a
                # failure. Stopping here would silently keep a truncated profile.
                raise ScrapeIncompleteError(username, films_url)

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
            watch_url = f"{self.BASE}/{username}/watchlist/page/{page}/"
            tree = self._get_listing_page(watch_url)
            if not tree:
                if page > 1 or watch_url in self._failed:
                    raise ScrapeIncompleteError(username, watch_url)
                break  # private or missing watchlist
            
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
        return meta
    
    def _parse_rating(self, span) -> float | None:
        """Thin wrapper around shared rating parser for backward compatibility."""
        return _parse_rating_span(span)
    
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
            review_items = tree.css("li.film-detail") or tree.css("div.listitem article.production-viewing")
            if not review_items:
                break

            for item in review_items:
                # Extract username from the author link
                author_link = item.css_first("a.avatar") or item.css_first("a.context")
                if not author_link:
                    continue

                href = author_link.attributes.get("href", "")
                if not href.startswith("/") or not href.endswith("/"):
                    continue

                username = href.strip("/")
                if not username:
                    continue

                # Check if they have a rating (star icons visible)
                has_rating = item.css_first("span.rating, span.inline-rating") is not None

                # Try to extract review date if visible
                review_date = None
                date_link = item.css_first("span._nobr a")
                if date_link:
                    review_date = date_link.text(strip=True)
                else:
                    time_el = item.css_first("time")
                    if time_el:
                        review_date = time_el.attributes.get("datetime") or time_el.text(strip=True) or None

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
        if not film_count:
            value_span = tree.css_first(f"a[href='/{username}/films/'] span.value")
            if value_span:
                try:
                    film_count = int(value_span.text(strip=True).replace(",", ""))
                except ValueError:
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

    PAGE_WAVE = 4  # listing pages fetched concurrently once page 1 shows there are more

    def __init__(self, delay: float = 0.2, max_concurrent: int = 5, max_rps: float | None = None,
                 rps_ceiling: float | None = None):
        self.delay = delay
        self._base_delay = delay
        self._max_concurrent = max_concurrent
        # Optional global cap on request *starts* per second, independent of latency.
        self.max_rps = max_rps
        # With a ceiling above max_rps, the rate is probed upward and backed off on throttling.
        self._aimd = AimdRate(start=max_rps, ceiling=rps_ceiling) if (max_rps and rps_ceiling and rps_ceiling > max_rps) else None
        self._pace_lock = asyncio.Lock()
        self._next_start = 0.0
        self._failed: set[str] = set()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.client = None
        self.cookies = _parse_cookie_header(LETTERBOXD_COOKIE)
        # Coordinated rate limiting: when one task hits 429, all tasks pause
        self._rate_limit_event = asyncio.Event()
        self._rate_limit_event.set()  # Start in "not rate limited" state

    @staticmethod
    def _parse_rating(span) -> float | None:
        """Reuse the shared rating parser."""
        return _parse_rating_span(span)

    def _pace_interval(self) -> float:
        """Minimum gap between request starts: the rps cap, or the (adaptive) delay shared across
        the concurrency slots. The delay grows after 429s, so backoff still slows everything."""
        rps = self._aimd.rps if self._aimd else self.max_rps
        cap = (1.0 / rps) if rps else 0.0
        return max(cap, self.delay / max(1, self._max_concurrent))

    async def _pace(self) -> None:
        """Space request starts evenly. Unlike a per-request sleep inside the concurrency slot,
        this keeps the request rate steady regardless of how slow individual responses are."""
        interval = self._pace_interval()
        if interval <= 0:
            return
        async with self._pace_lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            start = max(now, self._next_start)
            self._next_start = start + interval
        if start > now:
            await asyncio.sleep(start - now)

    def _decay_delay(self) -> None:
        """After a successful request, ease the delay back toward its base (mirrors the sync scraper)."""
        self.delay = max(self._base_delay, self.delay * 0.8 + self._base_delay * 0.2)
        if self._aimd:
            self._aimd.on_ok()

    def _note_throttle(self) -> None:
        """A 429 or a blocking 403 was seen: tell the adaptive rate controller (if enabled)."""
        if self._aimd:
            self._aimd.on_throttle()

    def _note_soft_block(self) -> None:
        """A blocked attempt that a retry recovered: a soft signal (cuts the rate only in clusters)."""
        if self._aimd:
            self._aimd.on_soft_block()

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
        self.client = TimedAsyncClient(self.client)
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

        self._failed.discard(url)
        async with self.semaphore:
            await self._pace()

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
                        self._note_throttle()
                        continue

                    resp.raise_for_status()
                    self._decay_delay()
                    return HTMLParser(resp.text)

                except httpx.TimeoutException:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Timeout on {url}, retrying in {wait_time}s (attempt {attempt + 1}/{MAX_HTTP_RETRIES})"
                    )
                    await asyncio.sleep(wait_time)

                except httpx.HTTPStatusError as exc:
                    logger.error(f"HTTP {exc.response.status_code} on {url}: {exc}")
                    if not (exc.response.status_code == 403 and _looks_private(exc.response.text)):
                        self._failed.add(url)
                        if exc.response.status_code == 403:
                            self._note_throttle()
                    return None

                except httpx.HTTPError as exc:
                    logger.error(f"Request error on {url}: {type(exc).__name__}: {exc}")
                    self._failed.add(url)
                    return None

            logger.error(f"Max retries exceeded for {url}")
            self._failed.add(url)
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
            self._failed.add(url)
            return None
        return tree

    async def _get_listing_page(self, url: str, attempts: int = 3) -> HTMLParser | None:
        """Async twin of LetterboxdScraper._get_listing_page: retry transient failures; None with
        `url` left in `self._failed` means it is still failing, plain None means 404/private."""
        tree = None
        for attempt in range(attempts):
            tree = await self._get_with_soft_block_recovery(url)
            if tree is not None or url not in self._failed:
                return tree
            if attempt < attempts - 1:
                await asyncio.sleep(2 * (attempt + 1) ** 2)  # 2s, 8s
        return tree

    async def _iter_pages(self, url_for_page, wave: int | None = None, max_pages: int = 500):
        """Yield (page_number, tree) in order.

        Page 1 is fetched alone, so a one-page listing costs one request. If the consumer keeps
        going, the following pages are fetched `wave` at a time. Pages past the end are cheap empty
        grids and the consumer stops at the first short/empty page, so the speculation wastes at
        most wave-1 requests per listing.
        """
        wave = wave or self.PAGE_WAVE
        yield 1, await self._get_listing_page(url_for_page(1))
        page = 2
        while page <= max_pages:
            numbers = list(range(page, min(page + wave, max_pages + 1)))
            trees = await asyncio.gather(*(self._get_listing_page(url_for_page(n)) for n in numbers))
            for n, tree in zip(numbers, trees):
                yield n, tree
            page += wave

    @staticmethod
    def _slug_of(item) -> str | None:
        react_comp = item.css_first("div.react-component")
        if not react_comp:
            return None
        return validate_slug(react_comp.attributes.get("data-item-slug"))

    @staticmethod
    def _viewing_data(item) -> tuple[float | None, bool]:
        rating, liked = None, False
        viewing_data = item.css_first("p.poster-viewingdata")
        if viewing_data:
            rating_span = viewing_data.css_first("span.rating")
            if rating_span:
                rating = _parse_rating_span(rating_span)
            liked = viewing_data.css_first("span.like") is not None
        return rating, liked

    async def _scrape_watchlist_into(self, username: str, films: dict[str, FilmInteraction]) -> None:
        """Merge the user's watchlist into `films`. Private/missing watchlists are skipped; a page
        that keeps failing raises, so a half-fetched watchlist is never kept."""
        def url_for(p: int) -> str:
            return f"{self.BASE}/{username}/watchlist/page/{p}/"

        seen: set[str] = set()
        async with aclosing(self._iter_pages(url_for)) as pages:
            async for page, tree in pages:
                if not tree:
                    if page > 1 or url_for(page) in self._failed:
                        raise ScrapeIncompleteError(username, url_for(page))
                    break  # private or missing watchlist
                items = tree.css("li.griditem")
                if not items:
                    break

                new_on_page = False
                for item in items:
                    slug = self._slug_of(item)
                    if not slug:
                        continue
                    if slug not in seen:
                        seen.add(slug)
                        new_on_page = True
                    if slug in films:
                        films[slug].watchlisted = True
                    else:
                        films[slug] = FilmInteraction(slug, None, False, True, False)

                # A short page is the last one; a page with nothing new means the site is
                # repeating content past the end.
                if len(items) < SCRAPER_PAGE_SIZE or not new_on_page:
                    break

    async def scrape_user(self, username: str, existing_slugs: set[str] | None = None, stop_on_existing: bool = False) -> list[FilmInteraction]:
        """
        Async variant of user scraping. Uses the shared client/semaphore so requests are globally
        rate limited. After page 1, listing pages are fetched PAGE_WAVE at a time. Any page that
        cannot be fetched raises ScrapeIncompleteError rather than keeping a truncated profile.
        """
        films: dict[str, FilmInteraction] = {}

        logger.info(f"Scraping {username}'s films (async)...")
        consecutive_existing = 0

        def url_for(p: int) -> str:
            return f"{self.BASE}/{username}/films/page/{p}/"

        async with aclosing(self._iter_pages(url_for)) as pages:
            async for page, tree in pages:
                if not tree:
                    # Past-the-end pages are 200 with an empty grid, so a missing page is a failure.
                    raise ScrapeIncompleteError(username, url_for(page))

                items = tree.css("li.griditem")
                if not items:
                    break

                page_had_new_films = False
                for item in items:
                    slug = self._slug_of(item)
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

                    rating, liked = self._viewing_data(item)
                    films[slug] = FilmInteraction(slug, rating, True, False, liked)

                logger.debug(f"  Watched page {page}: {len(items)} films")

                if stop_on_existing and not page_had_new_films and existing_slugs:
                    logger.info(f"  No new films found on page {page}, stopping early (incremental mode)")
                    break
                if len(items) < SCRAPER_PAGE_SIZE:
                    break

        logger.info(f"Scraping {username}'s watchlist (async)...")
        await self._scrape_watchlist_into(username, films)

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

        empty_pages = 0

        def url_for(p: int) -> str:
            return f"{self.BASE}/{username}/films/page/{p}/"

        async with aclosing(self._iter_pages(url_for)) as pages:
            async for page, tree in pages:
                if not tree:
                    raise ScrapeIncompleteError(username, url_for(page))

                items = tree.css("li.griditem")
                if not items:
                    empty_pages += 1
                    if empty_pages >= 2:
                        break
                    continue

                empty_pages = 0
                for item in items:
                    slug = self._slug_of(item)
                    if not slug:
                        continue
                    rating, liked = self._viewing_data(item)
                    films[slug] = FilmInteraction(slug, rating, True, False, liked)

                if len(items) < SCRAPER_PAGE_SIZE * 0.8:
                    break
                if estimated_pages and page + 1 > estimated_pages + 2:
                    break

        await self._scrape_watchlist_into(username, films)

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
            return await self._scrape_batch_with_client(TimedAsyncClient(temp_client), slugs)
    
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
            await self._pace()

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
                        self._note_throttle()
                        continue

                    resp.raise_for_status()
                    self._decay_delay()
                    tree = HTMLParser(resp.text)
                    meta = parse_film_page(tree, slug)
                    return meta

                except httpx.TimeoutException:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Timeout on {slug}, retrying in {wait_time}s (attempt {attempt + 1}/{MAX_HTTP_RETRIES})"
                    )
                    await asyncio.sleep(wait_time)

                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP {e.response.status_code} on {slug}: {e}")
                    if e.response.status_code == 403 and not _looks_private(e.response.text):
                        self._note_throttle()
                    return None

                except httpx.HTTPError as e:
                    logger.error(f"Request error on {slug}: {type(e).__name__}: {e}")
                    return None

            logger.error(f"Max retries exceeded for {slug}")
            return None
