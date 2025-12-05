"""
Configuration constants for the Letterboxd recommender system.

This module centralizes all magic numbers and configurable parameters.
Values can be overridden via environment variables or configuration files.
"""
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_float_env(key: str, default: float, min_val: float = 0) -> float:
    """
    Safely parse float from environment variable with validation.

    Args:
        key: Environment variable name
        default: Default value if not set or invalid
        min_val: Minimum allowed value

    Returns:
        Validated float value
    """
    try:
        val = float(os.environ.get(key, default))
        if val < min_val:
            logger.warning(f"{key}={val} is below minimum {min_val}, using {min_val}")
            return min_val
        return val
    except ValueError:
        logger.warning(f"Invalid {key}='{os.environ.get(key)}', using default {default}")
        return default


def _get_int_env(key: str, default: int, min_val: int = 1) -> int:
    """
    Safely parse integer from environment variable with validation.

    Args:
        key: Environment variable name
        default: Default value if not set or invalid
        min_val: Minimum allowed value

    Returns:
        Validated integer value
    """
    try:
        val = int(os.environ.get(key, default))
        if val < min_val:
            logger.warning(f"{key}={val} is below minimum {min_val}, using {min_val}")
            return min_val
        return val
    except ValueError:
        logger.warning(f"Invalid {key}='{os.environ.get(key)}', using default {default}")
        return default


# Database Configuration
DB_PATH = Path(os.environ.get("LETTERBOXD_DB", "data/letterboxd.db"))

# Scraper Configuration
DEFAULT_SCRAPER_DELAY = _get_float_env("LETTERBOXD_SCRAPER_DELAY", 1.0, min_val=0.1)
DEFAULT_ASYNC_DELAY = _get_float_env("LETTERBOXD_ASYNC_DELAY", 0.2, min_val=0.0)
DEFAULT_MAX_CONCURRENT = _get_int_env("LETTERBOXD_MAX_CONCURRENT", 5, min_val=1)

# Scraper Limits
SCRAPER_MAX_CAST = 10  # Cast members to scrape per film (more than we score, for future use)
HTTP_TIMEOUT = 30.0  # HTTP request timeout in seconds
MAX_CONSECUTIVE_EXISTING = 20  # For incremental scraping early termination

# Retry and Rate Limiting
MAX_HTTP_RETRIES = 3
MAX_429_RETRY_SECONDS = 300  # Maximum total time to wait for 429 responses
DEFAULT_RETRY_AFTER = 60  # Default wait time if Retry-After header missing

# Recommender Configuration
MIN_COMMON_FILMS = 5  # Minimum overlap for collaborative filtering
DEFAULT_MIN_NEIGHBORS = 3
DEFAULT_K_NEIGHBORS = 10
DEFAULT_MAX_PER_DIRECTOR = 2

# Batch Processing
DEFAULT_MAX_PER_BATCH = 100

# Popularity Thresholds
POPULARITY_HIGH_THRESHOLD = 10000
POPULARITY_MED_THRESHOLD = 1000

# Profile Weights - centralized from profile.py
WEIGHT_LOVED = 2.0        # Rating 4.5-5.0
WEIGHT_LIKED = 1.0        # Rating 3.5-4.0
WEIGHT_NEUTRAL = 0.3      # Rating 3.0
WEIGHT_DISLIKED = -0.5    # Rating 2.0-2.5
WEIGHT_HATED = -1.5       # Rating 0.5-1.5
WEIGHT_LIKED_NO_RATING = 1.5
WEIGHT_WATCHED_ONLY = 0.4
WEIGHT_WATCHLISTED = 0.2

# List multipliers
LIST_MULTIPLIER_FAVORITES = 3.0
LIST_MULTIPLIER_TOP_10 = 2.0
LIST_MULTIPLIER_TOP_30 = 1.5
LIST_MULTIPLIER_RANKED_OTHER = 1.2
LIST_MULTIPLIER_CURATED = 1.3

# Profile configuration
MAX_CAST_CONSIDERED = 5
MAX_THEMES_CONSIDERED = 10
SECONDARY_COUNTRY_WEIGHT = 0.3

# Normalization exponents (0 = no normalization, 0.5 = sqrt, 1.0 = full count normalization)
NORM_EXPONENT_DEFAULT = 0.5
NORM_EXPONENT_ACTORS = 0.3

# Recommender Weights
WEIGHTS = {
    'genre': 1.0,
    'director': 3.0,
    'actor': 0.5,
    'theme': 0.4,
    'decade': 0.3,
    'community_rating': 0.8,
    'popularity': 0.2,
    'country': 1.5,
    'language': 1.0,
    'writer': 2.0,
    'cinematographer': 1.0,
    'composer': 0.8,
    'genre_pair': 0.6,  # Genre co-occurrence (supplementary signal)
}

# Serendipity Configuration
SERENDIPITY_FACTOR = 0.15
SERENDIPITY_MIN_RATING = 3.5
SERENDIPITY_POPULARITY_CAP = 50000

# Match Thresholds
MATCH_THRESHOLD_GENRE = 0.5
MATCH_THRESHOLD_ACTOR = 0.5
MATCH_THRESHOLD_LANGUAGE = 0.5
MATCH_THRESHOLD_DIRECTOR = 1.0
MATCH_THRESHOLD_WRITER = 1.0
MATCH_THRESHOLD_CINE = 0.8
MATCH_THRESHOLD_COMPOSER = 0.8

# Rating Differences
RATING_DIFF_HIGH = 0.3
RATING_DIFF_MED = 0.5

# Similar Film Scoring
SIMILAR_DIRECTOR_BONUS = 5.0
SIMILAR_CAST_SCORE = 0.5
SIMILAR_DECADE_SCORE = 0.5

# Negative Penalty Configuration
NEGATIVE_PENALTY_MULTIPLIER = 1.5  # Amplify negative matches

# Negative thresholds (when to surface negative matches in warnings)
NEGATIVE_THRESHOLD_DIRECTOR = -0.8
NEGATIVE_THRESHOLD_GENRE = -0.3
NEGATIVE_THRESHOLD_ACTOR = -0.3
NEGATIVE_THRESHOLD_WRITER = -0.8
NEGATIVE_THRESHOLD_CINE = -0.6
NEGATIVE_THRESHOLD_COMPOSER = -0.6

# Confidence weighting based on sample size
CONFIDENCE_MIN_SAMPLES = {
    'director': 3,     # Directors need fewer samples (distinctive signal)
    'genre': 5,        # Genres are common, need more observations
    'actor': 4,        # Actors somewhere in between
    'theme': 5,        # Themes need decent sample
    'country': 5,      # Countries need decent sample
    'language': 5,     # Languages need decent sample
    'writer': 2,       # Writers are rare, accept fewer samples
    'cinematographer': 2,  # Cinematographers are rare
    'composer': 2,     # Composers are rare
}

# IDF (Inverse Document Frequency) weighting for rarity
USE_IDF_WEIGHTING = True  # Enable IDF weighting for attributes
IDF_DISTINCTIVE_THRESHOLD = 2.0  # IDF above this is "distinctive taste"

# Profile Schema Versioning
# Increment this when:
# - UserProfile fields are added/removed
# - Weight constants change (WEIGHT_*, WEIGHTS)
# - Normalization exponents change (NORM_EXPONENT_*)
# - List multipliers change (LIST_MULTIPLIER_*)
# - Confidence min samples change (CONFIDENCE_MIN_SAMPLES)
PROFILE_SCHEMA_VERSION = 1

# Discovery Priority Configuration
# Priority scores for different user discovery sources
DISCOVERY_PRIORITY_MAP = {
    'film_reviews': 100,  # Engaged reviewers (highest quality)
    'followers': 80,      # Social connections
    'following': 80,      # Social connections
    'popular': 70,        # Popular members
    'film': 50,           # Film fans (default)
}

# Temporal Decay Configuration
# Weight recent interactions more heavily to account for evolving tastes
TEMPORAL_DECAY_ENABLED = True
TEMPORAL_DECAY_HALF_LIFE_DAYS = 365 * 2  # 2 years: weight halves every 2 years
TEMPORAL_DECAY_MIN_WEIGHT = 0.1  # Floor to prevent old ratings from vanishing completely
