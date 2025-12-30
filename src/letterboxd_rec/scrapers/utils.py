"""Utility functions for scraping."""

import json
import logging
import re

from selectolax.parser import HTMLParser

from ..config import SCRAPER_MAX_CAST
from .types import FilmMetadata

logger = logging.getLogger(__name__)


def parse_cookie_header(cookie_header: str | None) -> dict[str, str]:
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


def parse_rating_span(span) -> float | None:
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


def parse_rating_count(text: str) -> int | None:
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


def parse_fan_count_from_html(html: str) -> int | None:
    """
    Extract fan count from the ratings-summary CSI snippet.
    Looks for patterns like '26K fans' or '4,876,723 fans'.
    """
    match = re.search(r"([0-9][0-9,.\u202fKMB]+)\s+fans", html, re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1)
    cleaned = raw.replace("\u202f", "").replace(" ", "")
    return parse_rating_count(cleaned)


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
        rating_count = parse_rating_count(ratings_el.text(strip=True))

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

    return FilmMetadata(
        slug=slug, title=title, year=year, directors=directors,
        genres=genres, cast=cast, themes=themes,
        runtime=runtime, avg_rating=avg_rating, rating_count=rating_count,
        fan_count=None,
        countries=countries, languages=languages, writers=writers,
        cinematographers=cinematographers, composers=composers,
        is_short=is_short, is_animation=is_animation,
    )
