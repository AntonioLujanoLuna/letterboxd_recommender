"""Scrapers package for Letterboxd data collection.

This package provides synchronous and asynchronous scrapers for
fetching user data and film metadata from Letterboxd.
"""

from .types import FilmInteraction, FilmMetadata
from .utils import (
    parse_cookie_header,
    validate_slug,
    parse_rating_span,
    parse_rating_count,
    parse_fan_count_from_html,
    parse_film_page,
)
from .sync import LetterboxdScraper
from .async_scraper import AsyncLetterboxdScraper

# Backward compatibility aliases
_parse_cookie_header = parse_cookie_header
_parse_rating_span = parse_rating_span
_parse_rating_count = parse_rating_count
_parse_fan_count_from_html = parse_fan_count_from_html

__all__ = [
    # Types
    "FilmInteraction",
    "FilmMetadata",
    # Utilities
    "parse_cookie_header",
    "validate_slug",
    "parse_rating_span",
    "parse_rating_count",
    "parse_fan_count_from_html",
    "parse_film_page",
    # Backward compatibility
    "_parse_cookie_header",
    "_parse_rating_span",
    "_parse_rating_count",
    "_parse_fan_count_from_html",
    # Scrapers
    "LetterboxdScraper",
    "AsyncLetterboxdScraper",
]
