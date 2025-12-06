import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, pstdev
from .database import load_json, parse_timestamp_naive
from .config import (
    MAX_CAST_CONSIDERED,
    MAX_THEMES_CONSIDERED,
    SECONDARY_COUNTRY_WEIGHT,
    BLENDED_CONFIDENCE_BASE,
    BLENDED_CONFIDENCE_MAX_RATINGS,
    BLENDED_CONFIDENCE_SPAN,
    WEIGHT_LOVED,
    WEIGHT_LIKED,
    WEIGHT_NEUTRAL,
    WEIGHT_DISLIKED,
    WEIGHT_HATED,
    WEIGHT_LIKED_NO_RATING,
    WEIGHT_WATCHED_ONLY,
    WEIGHT_WATCHLISTED,
    LIST_MULTIPLIER_FAVORITES,
    LIST_MULTIPLIER_TOP_10,
    LIST_MULTIPLIER_TOP_30,
    LIST_MULTIPLIER_RANKED_OTHER,
    LIST_MULTIPLIER_CURATED,
    NORM_EXPONENT_DEFAULT,
    NORM_EXPONENT_ACTORS,
    TEMPORAL_DECAY_ENABLED,
    TEMPORAL_DECAY_HALF_LIFE_DAYS,
    TEMPORAL_DECAY_MIN_WEIGHT,
)

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """Aggregated user preferences from their film interactions."""
    n_films: int = 0
    n_rated: int = 0
    n_liked: int = 0
    avg_liked_rating: float | None = None
    rating_mean: float | None = None
    rating_std: float | None = None
    weighting_mode: str = "absolute"

    genres: dict[str, float] = field(default_factory=dict)
    directors: dict[str, float] = field(default_factory=dict)
    actors: dict[str, float] = field(default_factory=dict)
    themes: dict[str, float] = field(default_factory=dict)
    decades: dict[int, float] = field(default_factory=dict)
    countries: dict[str, float] = field(default_factory=dict)
    languages: dict[str, float] = field(default_factory=dict)
    writers: dict[str, float] = field(default_factory=dict)
    cinematographers: dict[str, float] = field(default_factory=dict)
    composers: dict[str, float] = field(default_factory=dict)

    # Observation counts for confidence weighting
    genre_counts: dict[str, int] = field(default_factory=dict)
    director_counts: dict[str, int] = field(default_factory=dict)
    actor_counts: dict[str, int] = field(default_factory=dict)
    theme_counts: dict[str, int] = field(default_factory=dict)
    country_counts: dict[str, int] = field(default_factory=dict)
    language_counts: dict[str, int] = field(default_factory=dict)
    writer_counts: dict[str, int] = field(default_factory=dict)
    cinematographer_counts: dict[str, int] = field(default_factory=dict)
    composer_counts: dict[str, int] = field(default_factory=dict)

    # Genre co-occurrence preferences (e.g., horror+comedy vs separate)
    genre_pairs: dict[str, float] = field(default_factory=dict)


def _compute_temporal_weight(
    scraped_at: str | None,
    reference_time: datetime | None = None,
    half_life_days: int = TEMPORAL_DECAY_HALF_LIFE_DAYS,
    min_weight: float = TEMPORAL_DECAY_MIN_WEIGHT
) -> float:
    """
    Compute temporal decay weight based on age.

    Uses exponential decay: weight = 2^(-age / half_life)

    Args:
        scraped_at: ISO format timestamp string (when the interaction was recorded)
        reference_time: Reference point for age calculation (default: now)
        half_life_days: Days after which weight is halved
        min_weight: Minimum weight floor

    Returns:
        Weight between min_weight and 1.0
    """
    if not scraped_at:
        return 1.0  # No timestamp = assume recent

    if reference_time is None:
        reference_time = datetime.now()

    try:
        interaction_time = parse_timestamp_naive(scraped_at)
        age_days = (reference_time - interaction_time).days

        if age_days <= 0:
            return 1.0

        # Exponential decay
        decay = math.pow(2, -age_days / half_life_days)

        return max(decay, min_weight)

    except (ValueError, TypeError):
        return 1.0


def _compute_weight_absolute(rating: float) -> float:
    """Extract weight from rating alone (absolute scale)."""
    if rating >= 4.5:
        return WEIGHT_LOVED
    elif rating >= 3.5:
        return WEIGHT_LIKED
    elif rating >= 3.0:
        return WEIGHT_NEUTRAL
    elif rating >= 2.0:
        return WEIGHT_DISLIKED
    else:
        return WEIGHT_HATED


def _normalize_scores(scores: dict, counts: dict, exponent: float = NORM_EXPONENT_DEFAULT) -> dict:
    """
    Normalize accumulated scores by count with configurable exponent.

    exponent=0.0: no normalization (raw scores)
    exponent=0.5: sqrt normalization (balance frequency and magnitude)
    exponent=1.0: full mean normalization
    """
    if exponent == 0:
        return dict(scores)
    return {k: v / (counts[k] ** exponent) for k, v in scores.items() if counts[k] > 0}


def _z_to_weight(z_score: float) -> float:
    """Map z-score to weight constant based on personal rating distribution."""
    if z_score >= 1.5:
        return WEIGHT_LOVED
    elif z_score >= 0.5:
        return WEIGHT_LIKED
    elif z_score >= -0.5:
        return WEIGHT_NEUTRAL
    elif z_score >= -1.5:
        return WEIGHT_DISLIKED
    else:
        return WEIGHT_HATED


def _compute_weight_normalized(
    uf: dict,
    user_mean: float | None,
    user_std: float
) -> float:
    """Weight using user's personal rating distribution (z-score)."""
    rating = uf.get('rating')
    liked = uf.get('liked', False)
    watched = uf.get('watched', False)
    watchlisted = uf.get('watchlisted', False)

    if rating is not None and user_mean is not None and user_std > 0:
        z_score = (rating - user_mean) / user_std
        return _z_to_weight(z_score)

    # Fallbacks
    if rating is not None:
        return _compute_weight_absolute(rating)
    if liked:
        return WEIGHT_LIKED_NO_RATING
    if watched:
        return WEIGHT_WATCHED_ONLY
    if watchlisted:
        return WEIGHT_WATCHLISTED
    return 0.0


def _compute_weight_blended(
    uf: dict,
    user_mean: float | None,
    user_std: float,
    n_ratings: int
) -> float:
    """
    Blend absolute and normalized weights based on confidence in user's scale.

    With few ratings, trust absolute scale more.
    With many ratings and stable std, trust personalized scale more.
    """
    rating = uf.get('rating')
    if rating is None:
        return _compute_weight(uf)

    absolute_weight = _compute_weight_absolute(rating)

    if user_mean is not None and user_std > 0.2 and n_ratings >= 10:
        z_score = (rating - user_mean) / user_std
        normalized_weight = _z_to_weight(z_score)
    else:
        normalized_weight = absolute_weight

    # Confidence ramp is configurable to avoid hard-coded convergence speed
    confidence = min(1.0, n_ratings / BLENDED_CONFIDENCE_MAX_RATINGS)
    blend_factor = BLENDED_CONFIDENCE_BASE + (confidence * BLENDED_CONFIDENCE_SPAN)

    return (blend_factor * normalized_weight) + ((1 - blend_factor) * absolute_weight)


def build_profile(
    user_films: list[dict],
    film_metadata: dict[str, dict],
    user_lists: list[dict] | None = None,
    username: str | None = None,
    use_cache: bool = True,
    use_temporal_decay: bool = TEMPORAL_DECAY_ENABLED,
    reference_time: datetime | None = None,
    weighting_mode: str = "absolute",
) -> UserProfile:
    """
    Build preference profile from user's film interactions and lists.

    If username provided and use_cache=True, checks for cached profile first.
    Cache is invalidated after 7 days or when new films are scraped.

    Args:
        user_films: List of user film interactions (with 'scraped_at' for temporal decay)
        film_metadata: Dict mapping slug -> film metadata
        user_lists: Optional list of user list entries
        username: Optional username for caching
        use_cache: Whether to use cached profiles
        use_temporal_decay: Whether to apply temporal decay weighting
        reference_time: Reference time for decay calculation (default: now)

    Weighting strategy:
    - Rating 4.5-5.0: +2.0 (loved it)
    - Rating 3.5-4.0: +1.0 (liked it)
    - Rating 3.0:     +0.3 (neutral-positive)
    - Rating 2.0-2.5: -0.5 (disliked)
    - Rating 0.5-1.5: -1.5 (hated)
    - Liked (heart):  +1.5
    - Watched only:   +0.4 (mild positive)
    - Watchlisted:    +0.2 (interest signal)

    List multipliers:
    - Favorites:      3.0x (strongest signal)
    - Ranked top 10:  2.0x
    - Ranked 11-30:   1.5x
    - Ranked 31+:     1.2x
    - Curated list:   1.3x

    Temporal decay (if enabled):
    - Recent interactions weighted more heavily
    - Half-life of 2 years by default (configurable)
    - Minimum weight of 0.1 prevents old favorites from vanishing
    """
    # Try to load from cache if username provided
    if username and use_cache:
        from .database import load_cached_profile
        cached = load_cached_profile(username, max_age_days=7)
        if cached:
            logger.debug(f"Using cached profile for {username}")
            # Reconstruct UserProfile from cached dict
            profile = UserProfile(**cached)
            return profile
    
    profile = UserProfile()

    # Build list weight lookup
    list_weights = _build_list_weights(user_lists)

    # Precompute temporal weights if enabled
    temporal_weights = {}
    if use_temporal_decay:
        ref_time = reference_time or datetime.now()
        for uf in user_films:
            slug = uf['slug']
            scraped_at = uf.get('scraped_at')
            temporal_weights[slug] = _compute_temporal_weight(scraped_at, ref_time)

    # Rating distribution stats for personalized weighting
    ratings = [uf['rating'] for uf in user_films if uf.get('rating') is not None]
    n_ratings = len(ratings)
    user_mean = mean(ratings) if ratings else None
    user_std = pstdev(ratings) if len(ratings) > 1 else 0.0

    def _weight_for_interaction(uf: dict) -> float:
        mode = (weighting_mode or "absolute").lower()
        if mode == "normalized":
            return _compute_weight_normalized(uf, user_mean, user_std)
        if mode == "blended":
            return _compute_weight_blended(uf, user_mean, user_std, n_ratings)
        return _compute_weight(uf)

    # Score accumulators
    scores = {
        'genre': defaultdict(float),
        'director': defaultdict(float),
        'actor': defaultdict(float),
        'theme': defaultdict(float),
        'decade': defaultdict(float),
        'country': defaultdict(float),
        'language': defaultdict(float),
        'writer': defaultdict(float),
        'cinematographer': defaultdict(float),
        'composer': defaultdict(float),
    }

    counts = {k: defaultdict(int) for k in scores}
    rated_films = []

    # Genre pair tracking (combined with main loop for efficiency)
    genre_pair_scores = defaultdict(float)
    genre_pair_counts = defaultdict(int)

    # Single iteration over user_films for all scoring
    for uf in user_films:
        slug = uf['slug']
        meta = film_metadata.get(slug)
        if not meta:
            continue

        # Determine base weight for this film
        base_weight = _weight_for_interaction(uf)

        # Apply temporal decay if enabled
        if use_temporal_decay:
            temporal_factor = temporal_weights.get(slug, 1.0)
            weight = base_weight * temporal_factor
        else:
            weight = base_weight

        # Apply list multiplier if film is in any lists
        if slug in list_weights:
            weight *= list_weights[slug]

        if weight == 0:
            continue

        # Extract and accumulate scores for each attribute type
        film_genres = load_json(meta.get('genres'))
        _accumulate_list_scores(
            film_genres,
            weight, scores['genre'], counts['genre']
        )
        _accumulate_list_scores(
            load_json(meta.get('directors')),
            weight, scores['director'], counts['director']
        )
        _accumulate_list_scores(
            load_json(meta.get('cast', []))[:MAX_CAST_CONSIDERED],
            weight * 0.7, scores['actor'], counts['actor']
        )
        _accumulate_list_scores(
            load_json(meta.get('themes', []))[:MAX_THEMES_CONSIDERED],
            weight * 0.5, scores['theme'], counts['theme']
        )
        _accumulate_list_scores(
            load_json(meta.get('languages', [])),
            weight, scores['language'], counts['language']
        )
        _accumulate_list_scores(
            load_json(meta.get('writers', [])),
            weight, scores['writer'], counts['writer']
        )
        _accumulate_list_scores(
            load_json(meta.get('cinematographers', [])),
            weight, scores['cinematographer'], counts['cinematographer']
        )
        _accumulate_list_scores(
            load_json(meta.get('composers', [])),
            weight, scores['composer'], counts['composer']
        )

        # Countries: primary gets full weight, secondary reduced
        countries = load_json(meta.get('countries', []))
        for i, country in enumerate(countries):
            country_weight = weight if i == 0 else weight * SECONDARY_COUNTRY_WEIGHT
            scores['country'][country] += country_weight
            counts['country'][country] += 1

        # Decades
        year = meta.get('year')
        if year:
            decade = (year // 10) * 10
            scores['decade'][decade] += weight
            counts['decade'][decade] += 1

        # Genre pair preferences (co-occurrence modeling) - combined in same loop
        for i, g1 in enumerate(film_genres):
            for g2 in film_genres[i+1:]:
                pair = "|".join(sorted([g1, g2]))
                genre_pair_scores[pair] += weight
                genre_pair_counts[pair] += 1

        # Track rated films for average (with recency info)
        if uf.get('rating'):
            rated_films.append({
                'rating': uf['rating'],
                'temporal_weight': temporal_weights.get(slug, 1.0) if use_temporal_decay else 1.0
            })

    # Normalize all scores consistently
    profile.genres = _normalize_scores(scores['genre'], counts['genre'])
    profile.directors = _normalize_scores(scores['director'], counts['director'])
    profile.actors = _normalize_scores(scores['actor'], counts['actor'], NORM_EXPONENT_ACTORS)
    profile.themes = _normalize_scores(scores['theme'], counts['theme'])
    profile.decades = _normalize_scores(scores['decade'], counts['decade'])
    profile.countries = _normalize_scores(scores['country'], counts['country'])
    profile.languages = _normalize_scores(scores['language'], counts['language'])
    profile.writers = _normalize_scores(scores['writer'], counts['writer'])
    profile.cinematographers = _normalize_scores(scores['cinematographer'], counts['cinematographer'])
    profile.composers = _normalize_scores(scores['composer'], counts['composer'])

    # Store observation counts for confidence weighting
    profile.genre_counts = dict(counts['genre'])
    profile.director_counts = dict(counts['director'])
    profile.actor_counts = dict(counts['actor'])
    profile.theme_counts = dict(counts['theme'])
    profile.country_counts = dict(counts['country'])
    profile.language_counts = dict(counts['language'])
    profile.writer_counts = dict(counts['writer'])
    profile.cinematographer_counts = dict(counts['cinematographer'])
    profile.composer_counts = dict(counts['composer'])

    # Normalize pairs (require at least 2 observations)
    profile.genre_pairs = {
        pair: score / (count ** 0.5)
        for pair, score in genre_pair_scores.items()
        if (count := genre_pair_counts[pair]) >= 2
    }

    # Aggregate counts
    profile.n_films = len(user_films)
    profile.n_rated = len(rated_films)
    profile.n_liked = sum(1 for f in user_films if f.get('liked'))
    profile.rating_mean = user_mean
    profile.rating_std = user_std if n_ratings > 1 else 0.0
    profile.weighting_mode = weighting_mode

    # Compute temporally-weighted average rating
    if rated_films:
        if use_temporal_decay:
            weighted_sum = sum(rf['rating'] * rf['temporal_weight'] for rf in rated_films)
            weight_sum = sum(rf['temporal_weight'] for rf in rated_films)
            profile.avg_liked_rating = weighted_sum / weight_sum if weight_sum > 0 else None
        else:
            profile.avg_liked_rating = sum(rf['rating'] for rf in rated_films) / len(rated_films)
    else:
        profile.avg_liked_rating = None
    
    # Save to cache if username provided
    if username:
        from .database import save_user_profile
        from dataclasses import asdict
        save_user_profile(username, asdict(profile))
        logger.debug(f"Cached profile for {username}")
    
    return profile


def _build_list_weights(user_lists: list[dict] | None) -> dict[str, float]:
    """Build lookup of film slug -> list weight multiplier."""
    if not user_lists:
        return {}
    
    list_weights = {}
    for entry in user_lists:
        slug = entry['film_slug']
        
        if entry.get('is_favorites'):
            multiplier = LIST_MULTIPLIER_FAVORITES
        elif entry.get('is_ranked') and entry.get('position'):
            position = entry['position']
            if position <= 10:
                multiplier = LIST_MULTIPLIER_TOP_10
            elif position <= 30:
                multiplier = LIST_MULTIPLIER_TOP_30
            else:
                multiplier = LIST_MULTIPLIER_RANKED_OTHER
        else:
            multiplier = LIST_MULTIPLIER_CURATED
        
        # Keep highest multiplier if film is in multiple lists
        if slug not in list_weights or multiplier > list_weights[slug]:
            list_weights[slug] = multiplier
    
    return list_weights


def _accumulate_list_scores(
    items: list, 
    weight: float, 
    scores: dict, 
    counts: dict
) -> None:
    """Accumulate scores for a list of items."""
    for item in items:
        scores[item] += weight
        counts[item] += 1


def _compute_weight(uf: dict) -> float:
    """Compute preference weight for a single film interaction."""
    rating = uf.get('rating')
    liked = uf.get('liked', False)
    watched = uf.get('watched', False)
    watchlisted = uf.get('watchlisted', False)

    # Explicit rating takes precedence
    if rating is not None:
        if rating >= 4.5:
            return WEIGHT_LOVED
        elif rating >= 3.5:
            return WEIGHT_LIKED
        elif rating >= 3.0:
            return WEIGHT_NEUTRAL
        elif rating >= 2.0:
            return WEIGHT_DISLIKED
        else:
            return WEIGHT_HATED
    
    # Liked without rating
    if liked:
        return WEIGHT_LIKED_NO_RATING
    
    # Just watched
    if watched:
        return WEIGHT_WATCHED_ONLY
    
    # Just watchlisted
    if watchlisted:
        return WEIGHT_WATCHLISTED
    
    return 0.0