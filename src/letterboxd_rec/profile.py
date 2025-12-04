import logging
from collections import defaultdict
from dataclasses import dataclass, field
from .database import load_json
from .config import (
    MAX_CAST_CONSIDERED,
    MAX_THEMES_CONSIDERED,
    SECONDARY_COUNTRY_WEIGHT,
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
)

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """Aggregated user preferences from their film interactions."""
    n_films: int = 0
    n_rated: int = 0
    n_liked: int = 0
    avg_liked_rating: float | None = None

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


def build_profile(
    user_films: list[dict], 
    film_metadata: dict[str, dict],
    user_lists: list[dict] | None = None,
    username: str | None = None,
    use_cache: bool = True
) -> UserProfile:
    """
    Build preference profile from user's film interactions and lists.
    
    If username provided and use_cache=True, checks for cached profile first.
    Cache is invalidated after 7 days or when new films are scraped.
    
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

        # Determine weight for this film
        weight = _compute_weight(uf)

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

        # Track rated films for average
        if uf.get('rating'):
            rated_films.append(uf['rating'])

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
    profile.avg_liked_rating = sum(rated_films) / len(rated_films) if rated_films else None
    
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