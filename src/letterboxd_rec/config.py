"""
Configuration constants for the Letterboxd recommender system.

This module centralizes all magic numbers and configurable parameters.
Values can be overridden via environment variables or configuration files.
"""
import os
from pathlib import Path

# Database Configuration
DB_PATH = Path(os.environ.get("LETTERBOXD_DB", "data/letterboxd.db"))

# Scraper Configuration
DEFAULT_SCRAPER_DELAY = float(os.environ.get("LETTERBOXD_SCRAPER_DELAY", "1.0"))
DEFAULT_ASYNC_DELAY = float(os.environ.get("LETTERBOXD_ASYNC_DELAY", "0.2"))
DEFAULT_MAX_CONCURRENT = int(os.environ.get("LETTERBOXD_MAX_CONCURRENT", "5"))

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
}

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
