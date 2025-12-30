"""Data types for film scraping."""

from dataclasses import dataclass


@dataclass
class FilmInteraction:
    """Represents a user's interaction with a film."""
    film_slug: str
    rating: float | None
    watched: bool
    watchlisted: bool
    liked: bool


@dataclass
class FilmMetadata:
    """Metadata for a film scraped from Letterboxd."""
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
