"""Attribute configuration and scoring rules for metadata recommendations."""

from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING

from ..config import (
    WEIGHTS,
    MATCH_THRESHOLD_GENRE,
    MATCH_THRESHOLD_ACTOR,
    MATCH_THRESHOLD_LANGUAGE,
    MATCH_THRESHOLD_DIRECTOR,
    MATCH_THRESHOLD_WRITER,
    MATCH_THRESHOLD_CINE,
    MATCH_THRESHOLD_COMPOSER,
    RATING_DIFF_HIGH,
    RATING_DIFF_MED,
    POPULARITY_HIGH_THRESHOLD,
    POPULARITY_MED_THRESHOLD,
    NEGATIVE_PENALTY_MULTIPLIER,
    NEGATIVE_PENALTY_MULTIPLIERS,
    NEGATIVE_THRESHOLD_DIRECTOR,
    NEGATIVE_THRESHOLD_GENRE,
    NEGATIVE_THRESHOLD_ACTOR,
    NEGATIVE_THRESHOLD_WRITER,
    NEGATIVE_THRESHOLD_CINE,
    NEGATIVE_THRESHOLD_COMPOSER,
    CONFIDENCE_MIN_SAMPLES,
    LONG_TAIL_BOOST,
    LONG_TAIL_RATING_COUNT,
    MOMENTUM_THRESHOLD_POSITIVE,
    MOMENTUM_THRESHOLD_NEGATIVE,
    MOMENTUM_WEIGHT_DIRECTOR,
    MOMENTUM_WEIGHT_GENRE,
    SECONDARY_COUNTRY_WEIGHT,
)

if TYPE_CHECKING:
    from ..profile import UserProfile


@dataclass
class AttributeConfig:
    """Configuration for scoring a film attribute."""
    name: str                          # e.g., "genre", "director"
    film_field: str                    # JSON field in film dict, e.g., "genres"
    profile_attr: str                  # attribute name on UserProfile, e.g., "genres"
    counts_attr: str                   # counts attribute, e.g., "genre_counts"
    weight: float                      # from WEIGHTS config
    match_threshold: float             # minimum score to report as reason
    negative_threshold: float          # threshold to report as warning (set to 0 to disable)
    max_items: Optional[int]           # limit items considered (e.g., 5 for cast)
    idf_type: Optional[str]            # key in IDF dict, or None to skip IDF
    reason_template: str               # e.g., "Genre: {}" or "Director: {}"
    warning_template: str              # e.g., "Genre: {} (disliked)"
    distinctive_reason_template: str   # e.g., "Genre: {} (distinctive taste)"
    secondary_weight: Optional[float] = None  # Optional multiplier for secondary values


# Type alias for scoring rule functions
RuleFunc = Callable[
    ["MetadataRecommender", dict, "UserProfile", list[str], list[str]],
    tuple[float, list[str], list[str]],
]


def _confidence_weight(count: int, min_for_full_confidence: int = 5) -> float:
    """
    Returns weight between 0.0 and 1.0 based on sample size.

    Reaches 1.0 at min_for_full_confidence observations.
    Uses sqrt scaling for smooth ramp-up.
    """
    if count >= min_for_full_confidence:
        return 1.0
    return (count / min_for_full_confidence) ** 0.5


def _director_confidence_rule(
    recommender,
    film: dict,
    profile: "UserProfile",
    reasons: list[str],
    warnings: list[str],
) -> tuple[float, list[str], list[str]]:
    """Annotate director matches with sample size to make low-confidence signals transparent."""
    film_directors = recommender._get_list(film, 'directors')
    for d in film_directors:
        if d in profile.directors and profile.directors[d] > MATCH_THRESHOLD_DIRECTOR:
            count = profile.director_counts.get(d, 1)
            confidence = _confidence_weight(count, CONFIDENCE_MIN_SAMPLES['director'])
            if confidence < 0.7:
                for i, reason in enumerate(reasons):
                    if reason.startswith(f"Director: {d}"):
                        plural = 's' if count > 1 else ''
                        reasons[i] = f"Director: {d} (based on {count} film{plural})"
                        break
    return 0.0, [], []


def _genre_pair_rule(
    recommender,
    film: dict,
    profile: "UserProfile",
    reasons: list[str],
    warnings: list[str],
) -> tuple[float, list[str], list[str]]:
    """Capture genre co-occurrence preferences as an explicit rule."""
    film_genres = recommender._get_list(film, 'genres')
    pair_score = 0.0
    matched_pairs: list[str] = []
    pair_warnings: list[str] = []

    for i, g1 in enumerate(film_genres):
        for g2 in film_genres[i + 1:]:
            pair = "|".join(sorted([g1, g2]))
            pair_value = None
            if pair in getattr(profile, "genre_interactions", {}):
                pair_value = profile.genre_interactions[pair]
            elif pair in profile.genre_pairs:
                pair_value = profile.genre_pairs[pair]

            if pair_value is None:
                continue

            if pair_value < 0:
                pair_penalty = NEGATIVE_PENALTY_MULTIPLIERS.get('genre_pair', NEGATIVE_PENALTY_MULTIPLIER)
                pair_score += pair_value * pair_penalty
                if pair_value < -0.5:
                    pair_warnings.append(f"Genre combo: {g1}+{g2} (disliked)")
            else:
                pair_score += pair_value
                if pair_value > 0.5:
                    matched_pairs.append(f"{g1}+{g2}")

    pair_reasons = [f"Genre combo: {matched_pairs[0]}"] if matched_pairs else []
    return pair_score * WEIGHTS.get('genre_pair', 0.6), pair_reasons, pair_warnings


def _decade_rule(
    recommender,
    film: dict,
    profile: "UserProfile",
    reasons: list[str],
    warnings: list[str],
) -> tuple[float, list[str], list[str]]:
    """Decade affinity scoring."""
    year = film.get('year')
    if not year:
        return 0.0, [], []

    decade = (year // 10) * 10
    if decade in profile.decades:
        learned_weight = (
            recommender.feature_weights.factor("decade", decade)
            if getattr(recommender, "feature_weights", None)
            else 1.0
        )
        return profile.decades[decade] * WEIGHTS['decade'] * learned_weight, [], []
    return 0.0, [], []


def _community_rating_rule(
    recommender,
    film: dict,
    profile: "UserProfile",
    reasons: list[str],
    warnings: list[str],
) -> tuple[float, list[str], list[str]]:
    """Favor films whose community rating aligns to the user's sweet spot."""
    avg = film.get('avg_rating')
    if avg and profile.avg_liked_rating:
        rating_diff = abs(avg - profile.avg_liked_rating)
        if rating_diff < RATING_DIFF_HIGH:
            return 1.0 * WEIGHTS['community_rating'], [f"Highly rated ({avg:.1f}*)"], []
        elif rating_diff < RATING_DIFF_MED:
            return 0.5 * WEIGHTS['community_rating'], [], []
    return 0.0, [], []


def _popularity_rule(
    recommender,
    film: dict,
    profile: "UserProfile",
    reasons: list[str],
    warnings: list[str],
) -> tuple[float, list[str], list[str]]:
    """Gentle popularity shaping plus long-tail boost."""
    _ = profile  # unused in this rule
    count = film.get('rating_count') or 0
    score = 0.0
    popularity_reasons: list[str] = []

    if count > POPULARITY_HIGH_THRESHOLD:
        score += 0.3 * WEIGHTS['popularity']
    elif count > POPULARITY_MED_THRESHOLD:
        score += 0.1 * WEIGHTS['popularity']
    elif count and count < LONG_TAIL_RATING_COUNT:
        score += LONG_TAIL_BOOST
        popularity_reasons.append("Underseen gem")

    return score, popularity_reasons, []


def _momentum_rule(
    recommender,
    film: dict,
    profile: "UserProfile",
    reasons: list[str],
    warnings: list[str],
) -> tuple[float, list[str], list[str]]:
    """
    Boost films matching strengthening preferences, penalize waning ones.

    Uses preference_momentum from profile to detect evolving taste.
    """
    momentum_data = getattr(profile, "preference_momentum", {})
    if not momentum_data:
        return 0.0, [], []

    score_delta = 0.0
    momentum_reasons: list[str] = []
    momentum_warnings: list[str] = []

    # Check directors
    for director in recommender._get_list(film, 'directors'):
        key = f"director:{director}"
        if key in momentum_data:
            momentum = momentum_data[key]
            if momentum > MOMENTUM_THRESHOLD_POSITIVE:
                score_delta += momentum * (MOMENTUM_WEIGHT_DIRECTOR * 1.5)
                momentum_reasons.append(f"Rising interest: {director}")
            elif momentum < MOMENTUM_THRESHOLD_NEGATIVE:
                score_delta += momentum * max(MOMENTUM_WEIGHT_DIRECTOR, 0.5)
                momentum_warnings.append(f"Waning interest: {director}")

    # Check genres
    for genre in recommender._get_list(film, 'genres'):
        key = f"genre:{genre}"
        if key in momentum_data:
            momentum = momentum_data[key]
            if momentum > MOMENTUM_THRESHOLD_POSITIVE:
                score_delta += momentum * (MOMENTUM_WEIGHT_GENRE * 1.3)
                momentum_reasons.append(f"Growing {genre} interest")
            elif momentum < MOMENTUM_THRESHOLD_NEGATIVE:
                score_delta += momentum * max(MOMENTUM_WEIGHT_GENRE, 0.4)

    return score_delta, momentum_reasons[:2], momentum_warnings[:2]


# Default scoring rules applied during recommendation
DEFAULT_SCORING_RULES: list[RuleFunc] = [
    _director_confidence_rule,
    _genre_pair_rule,
    _decade_rule,
    _community_rating_rule,
    _popularity_rule,
    _momentum_rule,
]


# Attribute configurations for metadata scoring
ATTRIBUTE_CONFIGS = [
    AttributeConfig(
        name='genre',
        film_field='genres',
        profile_attr='genres',
        counts_attr='genre_counts',
        weight=WEIGHTS['genre'],
        match_threshold=MATCH_THRESHOLD_GENRE,
        negative_threshold=NEGATIVE_THRESHOLD_GENRE,
        max_items=None,
        idf_type='genre',
        reason_template='Genre: {}',
        warning_template='Genre: {} (disliked)',
        distinctive_reason_template='Genre: {} (distinctive taste)'
    ),
    AttributeConfig(
        name='director',
        film_field='directors',
        profile_attr='directors',
        counts_attr='director_counts',
        weight=WEIGHTS['director'],
        match_threshold=MATCH_THRESHOLD_DIRECTOR,
        negative_threshold=NEGATIVE_THRESHOLD_DIRECTOR,
        max_items=None,
        idf_type='director',
        reason_template='Director: {}',
        warning_template='Director: {} (disliked)',
        distinctive_reason_template='Director: {} (distinctive)'
    ),
    AttributeConfig(
        name='actor',
        film_field='cast',
        profile_attr='actors',
        counts_attr='actor_counts',
        weight=WEIGHTS['actor'],
        match_threshold=MATCH_THRESHOLD_ACTOR,
        negative_threshold=NEGATIVE_THRESHOLD_ACTOR,
        max_items=5,
        idf_type=None,
        reason_template='Cast: {}',
        warning_template='Actor: {} (disliked)',
        distinctive_reason_template='Cast: {}'
    ),
    AttributeConfig(
        name='theme',
        film_field='themes',
        profile_attr='themes',
        counts_attr='theme_counts',
        weight=WEIGHTS['theme'],
        match_threshold=0.0,
        negative_threshold=0.0,
        max_items=None,
        idf_type=None,
        reason_template='Theme: {}',
        warning_template='Theme: {} (disliked)',
        distinctive_reason_template='Theme: {}'
    ),
    AttributeConfig(
        name='language',
        film_field='languages',
        profile_attr='languages',
        counts_attr='language_counts',
        weight=WEIGHTS['language'],
        match_threshold=MATCH_THRESHOLD_LANGUAGE,
        negative_threshold=0.0,
        max_items=None,
        idf_type=None,
        reason_template='Language: {}',
        warning_template='Language: {} (disliked)',
        distinctive_reason_template='Language: {}'
    ),
    AttributeConfig(
        name='country',
        film_field='countries',
        profile_attr='countries',
        counts_attr='country_counts',
        weight=WEIGHTS['country'],
        match_threshold=0.2,
        negative_threshold=0.0,
        max_items=3,
        idf_type='country',
        reason_template='Country: {}',
        warning_template='Country: {} (disliked)',
        distinctive_reason_template='Country: {}',
        secondary_weight=SECONDARY_COUNTRY_WEIGHT,
    ),
    AttributeConfig(
        name='writer',
        film_field='writers',
        profile_attr='writers',
        counts_attr='writer_counts',
        weight=WEIGHTS['writer'],
        match_threshold=MATCH_THRESHOLD_WRITER,
        negative_threshold=NEGATIVE_THRESHOLD_WRITER,
        max_items=None,
        idf_type=None,
        reason_template='Writer: {}',
        warning_template='Writer: {} (disliked)',
        distinctive_reason_template='Writer: {}'
    ),
    AttributeConfig(
        name='cinematographer',
        film_field='cinematographers',
        profile_attr='cinematographers',
        counts_attr='cinematographer_counts',
        weight=WEIGHTS['cinematographer'],
        match_threshold=MATCH_THRESHOLD_CINE,
        negative_threshold=NEGATIVE_THRESHOLD_CINE,
        max_items=None,
        idf_type=None,
        reason_template='Cinematography: {}',
        warning_template='Cinematographer: {} (disliked)',
        distinctive_reason_template='Cinematography: {}'
    ),
    AttributeConfig(
        name='composer',
        film_field='composers',
        profile_attr='composers',
        counts_attr='composer_counts',
        weight=WEIGHTS['composer'],
        match_threshold=MATCH_THRESHOLD_COMPOSER,
        negative_threshold=NEGATIVE_THRESHOLD_COMPOSER,
        max_items=None,
        idf_type=None,
        reason_template='Composer: {}',
        warning_template='Composer: {} (disliked)',
        distinctive_reason_template='Composer: {}'
    ),
]
