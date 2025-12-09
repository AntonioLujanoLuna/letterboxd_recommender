from dataclasses import dataclass, field
import logging
from typing import Callable, Optional
import math
import json
from pathlib import Path
from .profile import UserProfile, build_profile
from .database import load_json
from .feature_weights import FeatureWeights, load_feature_weights
from .config import (
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
    SIMILAR_DIRECTOR_BONUS,
    SIMILAR_CAST_SCORE,
    SIMILAR_DECADE_SCORE,
    NEGATIVE_PENALTY_MULTIPLIER,
    NEGATIVE_PENALTY_MULTIPLIERS,
    NEGATIVE_THRESHOLD_DIRECTOR,
    NEGATIVE_THRESHOLD_GENRE,
    NEGATIVE_THRESHOLD_ACTOR,
    NEGATIVE_THRESHOLD_WRITER,
    NEGATIVE_THRESHOLD_CINE,
    NEGATIVE_THRESHOLD_COMPOSER,
    CONFIDENCE_MIN_SAMPLES,
    USE_IDF_WEIGHTING,
    IDF_DISTINCTIVE_THRESHOLD,
    SERENDIPITY_FACTOR,
    SERENDIPITY_MIN_RATING,
    SERENDIPITY_POPULARITY_CAP,
    SERENDIPITY_PERCENTILE_WINDOW,
    SERENDIPITY_MIN_RANK,
    SERENDIPITY_MAX_RANK,
    SERENDIPITY_RELATIVE_NOVELTY_WEIGHT,
    ATTRIBUTE_CAPS,
    LONG_TAIL_BOOST,
    LONG_TAIL_RATING_COUNT,
    MOMENTUM_THRESHOLD_POSITIVE,
    MOMENTUM_THRESHOLD_NEGATIVE,
    MOMENTUM_WEIGHT_DIRECTOR,
    MOMENTUM_WEIGHT_GENRE,
    TFIDF_FIELDS,
    TFIDF_MIN_DF,
    TFIDF_MAX_FEATURES,
    COLLAB_SHRINKAGE,
    COLLAB_IMPLICIT_WEIGHTS,
    COLLAB_POPULARITY_DEBIAS,
    ITEM_SIM_CACHE_PATH,
    ITEM_SIM_MIN_RATINGS,
    ITEM_SIM_MAX_ITEMS,
)

logger = logging.getLogger(__name__)


def _fuse_normalized(
    meta_results: list["Recommendation"],
    collab_results: list["Recommendation"],
    weight_meta: float = 0.6,
    weight_collab: float = 0.4,
) -> list[tuple[str, float, list[str]]]:
    """
    Score-based fusion with min-max normalization.

    Preserves relative score differences within each strategy.
    """

    def normalize_scores(recs: list["Recommendation"]) -> dict[str, float]:
        if not recs:
            return {}
        scores = [r.score for r in recs]
        min_s, max_s = min(scores), max(scores)
        range_s = max_s - min_s if max_s > min_s else 1.0
        return {r.slug: (r.score - min_s) / range_s for r in recs}

    meta_norm = normalize_scores(meta_results)
    collab_norm = normalize_scores(collab_results)

    # Collect all slugs
    all_slugs = set(meta_norm.keys()) | set(collab_norm.keys())

    fused: dict[str, float] = {}
    reasons_map: dict[str, list[str]] = {}
    for slug in all_slugs:
        m_score = meta_norm.get(slug, 0.0) * weight_meta
        c_score = collab_norm.get(slug, 0.0) * weight_collab

        # Bonus for appearing in both (consensus signal)
        consensus_bonus = 0.1 if slug in meta_norm and slug in collab_norm else 0.0

        fused[slug] = m_score + c_score + consensus_bonus

        # Merge reasons
        reasons = []
        for r in meta_results:
            if r.slug == slug:
                reasons.extend(r.reasons)
                reasons.extend(r.warnings)
        for r in collab_results:
            if r.slug == slug:
                reasons.extend(r.reasons)
                reasons.extend(getattr(r, "warnings", []))
        reasons_map[slug] = list(dict.fromkeys(reasons))  # dedupe preserving order

    return [
        (slug, score, reasons_map.get(slug, []))
        for slug, score in sorted(fused.items(), key=lambda x: -x[1])
    ]

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
    warning_template: str              # e.g., "⚠️ Genre: {} (disliked)"
    distinctive_reason_template: str   # e.g., "Genre: {} (distinctive taste)"


RuleFunc = Callable[
    ["MetadataRecommender", dict, UserProfile, list[str], list[str]],
    tuple[float, list[str], list[str]],
]


class ScoringEngine:
    """Composable scoring pipeline for metadata recommendations."""

    def __init__(self, attribute_configs: list[AttributeConfig], rules: list["RuleFunc"] | None = None):
        self.attribute_configs = attribute_configs
        self.rules = rules or []

    def score(
        self,
        recommender: "MetadataRecommender",
        film: dict,
        profile: UserProfile,
    ) -> tuple[float, list[str], list[str]]:
        score = 0.0
        reasons: list[str] = []
        warnings: list[str] = []

        for config in self.attribute_configs:
            attr_score, attr_reasons, attr_warnings = recommender._score_attribute(film, profile, config)
            score += attr_score
            reasons.extend(attr_reasons)
            warnings.extend(attr_warnings)

        for rule in self.rules:
            delta, extra_reasons, extra_warnings = rule(recommender, film, profile, reasons, warnings)
            score += delta
            if extra_reasons:
                reasons.extend(extra_reasons)
            if extra_warnings:
                warnings.extend(extra_warnings)

        return score, reasons, warnings


def _confidence_weight(count: int, min_for_full_confidence: int = 5) -> float:
    """
    Returns weight between 0.0 and 1.0 based on sample size.

    Reaches 1.0 at min_for_full_confidence observations.
    Uses sqrt scaling for smooth ramp-up.

    Examples:
    - 1 observation → 0.45 confidence
    - 2 observations → 0.63 confidence
    - 3 observations → 0.77 confidence
    - 5+ observations → 1.0 confidence
    """
    if count >= min_for_full_confidence:
        return 1.0
    return (count / min_for_full_confidence) ** 0.5


def _director_confidence_rule(
    recommender: "MetadataRecommender",
    film: dict,
    profile: UserProfile,
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
    recommender: "MetadataRecommender",
    film: dict,
    profile: UserProfile,
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
            # Prefer interaction effects when available, otherwise fall back to co-occurrence counts
            if hasattr(profile, "genre_interactions") and pair in getattr(profile, "genre_interactions", {}):
                pair_value = profile.genre_interactions[pair]
            elif pair in profile.genre_pairs:
                pair_value = profile.genre_pairs[pair]

            if pair_value is None:
                continue

            if pair_value < 0:
                pair_penalty = NEGATIVE_PENALTY_MULTIPLIERS.get('genre_pair', NEGATIVE_PENALTY_MULTIPLIER)
                pair_score += pair_value * pair_penalty
                if pair_value < -0.5:
                    pair_warnings.append(f"⚠️ Genre combo: {g1}+{g2} (disliked)")
            else:
                pair_score += pair_value
                if pair_value > 0.5:
                    matched_pairs.append(f"{g1}+{g2}")

    pair_reasons = [f"Genre combo: {matched_pairs[0]}"] if matched_pairs else []
    return pair_score * WEIGHTS.get('genre_pair', 0.6), pair_reasons, pair_warnings


def _decade_rule(
    recommender: "MetadataRecommender",
    film: dict,
    profile: UserProfile,
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


def _country_rule(
    recommender: "MetadataRecommender",
    film: dict,
    profile: UserProfile,
    reasons: list[str],
    warnings: list[str],
) -> tuple[float, list[str], list[str]]:
    """Country affinity with primary/secondary weighting and confidence."""
    film_countries = recommender._get_list(film, 'countries')
    country_reasons: list[str] = []
    country_score_total = 0.0

    for i, country in enumerate(film_countries):
        if country in profile.countries:
            country_score = profile.countries[country]
            count = profile.country_counts.get(country, 1)
            confidence = _confidence_weight(count, CONFIDENCE_MIN_SAMPLES['country'])

            country_weight = WEIGHTS['country'] if i == 0 else WEIGHTS['country'] * recommender.COUNTRY_SECONDARY_WEIGHT
            country_score_total += country_score * country_weight * confidence
            if i == 0 and country_score > 0.5:
                country_reasons.append(f"Country: {country}")

    return country_score_total, country_reasons, []


def _community_rating_rule(
    recommender: "MetadataRecommender",
    film: dict,
    profile: UserProfile,
    reasons: list[str],
    warnings: list[str],
) -> tuple[float, list[str], list[str]]:
    """Favor films whose community rating aligns to the user's sweet spot."""
    avg = film.get('avg_rating')
    if avg and profile.avg_liked_rating:
        rating_diff = abs(avg - profile.avg_liked_rating)
        if rating_diff < RATING_DIFF_HIGH:
            return 1.0 * WEIGHTS['community_rating'], [f"Highly rated ({avg:.1f}★)"], []
        elif rating_diff < RATING_DIFF_MED:
            return 0.5 * WEIGHTS['community_rating'], [], []
    return 0.0, [], []


def _popularity_rule(
    recommender: "MetadataRecommender",
    film: dict,
    profile: UserProfile,
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
    recommender: "MetadataRecommender",
    film: dict,
    profile: UserProfile,
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
                score_delta += momentum * MOMENTUM_WEIGHT_DIRECTOR
                momentum_reasons.append(f"Rising interest: {director}")
            elif momentum < MOMENTUM_THRESHOLD_NEGATIVE:
                # Negative momentum penalizes matching films
                score_delta += momentum * 0.3
                momentum_warnings.append(f"Waning interest: {director}")

    # Check genres
    for genre in recommender._get_list(film, 'genres'):
        key = f"genre:{genre}"
        if key in momentum_data:
            momentum = momentum_data[key]
            if momentum > MOMENTUM_THRESHOLD_POSITIVE:
                score_delta += momentum * MOMENTUM_WEIGHT_GENRE
                if not momentum_reasons:
                    momentum_reasons.append(f"Growing {genre} interest")
            elif momentum < MOMENTUM_THRESHOLD_NEGATIVE:
                score_delta += momentum * 0.2

    return score_delta, momentum_reasons[:1], momentum_warnings[:1]


DEFAULT_SCORING_RULES: list[RuleFunc] = [
    _director_confidence_rule,
    _genre_pair_rule,
    _decade_rule,
    _country_rule,
    _community_rating_rule,
    _popularity_rule,
    _momentum_rule,
]


# Fields we repeatedly parse from JSON-encoded strings
PARSED_FIELDS = (
    'genres',
    'directors',
    'cast',
    'themes',
    'countries',
    'languages',
    'writers',
    'cinematographers',
    'composers',
)


class TfidfEmbedder:
    """
    Lightweight TF-IDF embedder for metadata-based cold-start similarity.

    Keeps everything in simple Python dicts to avoid heavy dependencies.
    """

    def __init__(self, films: dict[str, dict]):
        self.films = films
        self.idf: dict[str, float] = {}
        self.vectors: dict[str, dict[str, float]] = {}
        self._build()

    def _tokenize(self, film: dict) -> list[str]:
        tokens = []
        parsed = film.get('_parsed', {})
        for field in TFIDF_FIELDS:
            values = parsed.get(field)
            if values is None:
                values = load_json(film.get(field, []))
            tokens.extend([str(v).lower() for v in (values or [])])
        return tokens

    def _build(self):
        df_counts = {}
        corpus = {}
        for slug, film in self.films.items():
            tokens = self._tokenize(film)
            corpus[slug] = tokens
            unique_tokens = set(tokens)
            for tok in unique_tokens:
                df_counts[tok] = df_counts.get(tok, 0) + 1

        n_docs = max(1, len(corpus))
        # Filter rare and extremely common tokens
        for tok, df in df_counts.items():
            if df < TFIDF_MIN_DF:
                continue
            idf = math.log((n_docs + 1) / (df + 1)) + 1
            self.idf[tok] = idf

        # Limit feature size by top IDF weights to avoid memory blow-up
        if len(self.idf) > TFIDF_MAX_FEATURES:
            # Keep top features
            top_tokens = sorted(self.idf.items(), key=lambda x: -x[1])[:TFIDF_MAX_FEATURES]
            self.idf = dict(top_tokens)

        for slug, tokens in corpus.items():
            tf = {}
            for tok in tokens:
                if tok not in self.idf:
                    continue
                tf[tok] = tf.get(tok, 0) + 1
            if not tf:
                continue
            vec = {tok: (count / len(tokens)) * self.idf[tok] for tok, count in tf.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.vectors[slug] = {tok: val / norm for tok, val in vec.items()}

    def similarity(self, slug_a: str, slug_b: str) -> float:
        va = self.vectors.get(slug_a)
        vb = self.vectors.get(slug_b)
        if not va or not vb:
            return 0.0
        # Cosine on sparse dicts
        if len(va) > len(vb):
            va, vb = vb, va
        return sum(weight * vb.get(tok, 0.0) for tok, weight in va.items())

    def rank_against(self, slug: str, candidates: list[str], top_k: int = 50) -> list[tuple[str, float]]:
        scores = []
        for other in candidates:
            if other == slug:
                continue
            sim = self.similarity(slug, other)
            if sim > 0:
                scores.append((other, sim))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def score_to_centroid(self, anchor_slugs: list[str], candidates: list[str], top_k: int = 50) -> list[tuple[str, float]]:
        if not anchor_slugs:
            return []
        centroid = {}
        for slug in anchor_slugs:
            vec = self.vectors.get(slug)
            if not vec:
                continue
            for tok, weight in vec.items():
                centroid[tok] = centroid.get(tok, 0.0) + weight
        if not centroid:
            return []
        norm = math.sqrt(sum(v * v for v in centroid.values())) or 1.0
        centroid = {k: v / norm for k, v in centroid.items()}

        scores = []
        for slug in candidates:
            vec = self.vectors.get(slug)
            if not vec:
                continue
            if len(vec) > len(centroid):
                vec, centroid = centroid, vec  # swap for faster loop
            sim = sum(weight * centroid.get(tok, 0.0) for tok, weight in vec.items())
            if sim > 0:
                scores.append((slug, sim))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

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
        warning_template='⚠️ Genre: {} (disliked)',
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
        warning_template='⚠️ Director: {} (disliked)',
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
        idf_type=None,  # Actors don't use IDF currently
        reason_template='Cast: {}',
        warning_template='⚠️ Actor: {} (disliked)',
        distinctive_reason_template='Cast: {}'
    ),
    AttributeConfig(
        name='theme',
        film_field='themes',
        profile_attr='themes',
        counts_attr='theme_counts',
        weight=WEIGHTS['theme'],
        match_threshold=0.0,  # Themes don't generate reasons currently
        negative_threshold=0.0,
        max_items=None,
        idf_type=None,
        reason_template='Theme: {}',
        warning_template='⚠️ Theme: {} (disliked)',
        distinctive_reason_template='Theme: {}'
    ),
    AttributeConfig(
        name='language',
        film_field='languages',
        profile_attr='languages',
        counts_attr='language_counts',
        weight=WEIGHTS['language'],
        match_threshold=MATCH_THRESHOLD_LANGUAGE,
        negative_threshold=0.0,  # Languages don't have negative thresholds currently
        max_items=None,
        idf_type=None,
        reason_template='Language: {}',
        warning_template='⚠️ Language: {} (disliked)',
        distinctive_reason_template='Language: {}'
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
        warning_template='⚠️ Writer: {} (disliked)',
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
        warning_template='⚠️ Cinematographer: {} (disliked)',
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
        warning_template='⚠️ Composer: {} (disliked)',
        distinctive_reason_template='Composer: {}'
    ),
]


@dataclass
class Recommendation:
    slug: str
    title: str
    year: int | None
    score: float
    reasons: list[str]
    warnings: list[str] = field(default_factory=list)

class MetadataRecommender:
    """
    Score films by metadata match to user profile.
    No embeddings—just weighted feature matching.
    """

    COUNTRY_SECONDARY_WEIGHT = 0.3

    def __init__(
        self,
        all_films: list[dict],
        use_idf: bool = USE_IDF_WEIGHTING,
        feature_weights: FeatureWeights | None = None,
        feature_weights_path: str | Path | None = None,
    ):
        # Pre-parse JSON fields once to avoid repeated load_json calls during scoring
        self.films = {f['slug']: f for f in all_films}
        for film in self.films.values():
            parsed = film.get('_parsed', {})
            for field in PARSED_FIELDS:
                if field not in parsed:
                    parsed[field] = load_json(film.get(field, []))
            film['_parsed'] = parsed
        self.use_idf = use_idf

        # Load IDF scores if enabled
        if self.use_idf:
            from .database import load_idf
            try:
                self.idf = load_idf()
                if not self.idf:
                    logger.warning("IDF table is empty. Run 'python main.py rebuild-idf' to compute IDF scores.")
                    self.idf = {}
            except Exception as e:
                logger.warning(f"Failed to load IDF scores: {e}. Continuing without IDF weighting.")
                self.idf = {}
        else:
            self.idf = {}
        self._tfidf: TfidfEmbedder | None = None
        self.feature_weights = feature_weights or load_feature_weights(feature_weights_path)
        self.scoring_engine = ScoringEngine(ATTRIBUTE_CONFIGS, DEFAULT_SCORING_RULES)

    def _get_list(self, film: dict, field: str, limit: int | None = None) -> list:
        """Return a parsed list for a film field (uses pre-parsed cache when available)."""
        parsed = film.get('_parsed')
        values = parsed.get(field) if parsed else None
        if values is None:
            values = load_json(film.get(field, []))
        values = values or []
        return values[:limit] if limit is not None else values

    def _ensure_tfidf(self):
        if self._tfidf is None:
            self._tfidf = TfidfEmbedder(self.films)

    def _score_attribute(
        self,
        film: dict,
        profile: UserProfile,
        config: AttributeConfig
    ) -> tuple[float, list[str], list[str]]:
        """
        Generic attribute scoring method.

        Returns:
            (score, reasons, warnings) tuple
        """
        # Load film attribute values (pre-parsed when available)
        film_values = self._get_list(film, config.film_field)

        # Apply max_items limit if specified
        if config.max_items is not None:
            film_values = film_values[:config.max_items]

        # Get profile scores and counts
        profile_scores = getattr(profile, config.profile_attr)
        profile_counts = getattr(profile, config.counts_attr)

        total_score = 0.0
        matched_items = []
        distinctive_items = []
        warnings = []

        for value in film_values:
            if value not in profile_scores:
                continue

            value_score = profile_scores[value]
            count = profile_counts.get(value, 1)
            learned_weight = (
                self.feature_weights.factor(config.name, value)
                if self.feature_weights
                else 1.0
            )
            adjusted_score = value_score * learned_weight

            # Apply confidence weighting
            confidence = _confidence_weight(count, CONFIDENCE_MIN_SAMPLES.get(config.name, 5))

            # Apply IDF weighting if enabled and configured
            idf_weight = 1.0
            if self.use_idf and config.idf_type and config.idf_type in self.idf:
                idf_weight = self.idf[config.idf_type].get(value, 1.0)

            # Handle negative scores with amplified penalty
            if adjusted_score < 0:
                penalty_multiplier = NEGATIVE_PENALTY_MULTIPLIERS.get(config.name, NEGATIVE_PENALTY_MULTIPLIER)
                total_score += adjusted_score * penalty_multiplier * confidence * idf_weight
                if config.negative_threshold != 0.0 and adjusted_score < config.negative_threshold:
                    warnings.append(config.warning_template.format(value))
            else:
                total_score += adjusted_score * confidence * idf_weight

                # Track positive matches for reasons
                if adjusted_score > config.match_threshold:
                    # Check if distinctive (high IDF)
                    if self.use_idf and config.idf_type and idf_weight > IDF_DISTINCTIVE_THRESHOLD:
                        distinctive_items.append(value)
                    else:
                        matched_items.append(value)

        # Soft cap to avoid any single attribute dominating
        cap = ATTRIBUTE_CAPS.get(config.name)
        if cap:
            total_score = max(min(total_score, cap), -cap)

        # Build reasons list
        reasons = []
        if distinctive_items:
            reasons.append(config.distinctive_reason_template.format(distinctive_items[0]))
        elif matched_items:
            # For most attributes, show first match or first two
            if config.name == 'actor':
                reasons.append(config.reason_template.format(', '.join(matched_items[:2])))
            elif config.name == 'genre':
                reasons.append(config.reason_template.format(', '.join(matched_items[:2])))
            else:
                reasons.append(config.reason_template.format(matched_items[0]))

        return total_score * config.weight, reasons, warnings

    def _compute_negative_profile(self, user_films: list[dict], film_metadata: dict[str, dict]) -> dict[str, set]:
        """
        Build explicit negative preferences from low-rated films.

        Helps avoid recommending films similar to ones the user disliked.
        """
        negatives: dict[str, set] = {
            'directors': set(),
            'genres': set(),
            'actors': set(),
            'themes': set(),
        }

        for uf in user_films:
            rating = uf.get('rating')
            if rating is None or rating >= 2.5:
                continue

            film = film_metadata.get(uf['slug'])
            if not film:
                continue

            # Strong negative signal (1-2 stars)
            if rating <= 2.0:
                negatives['directors'].update(self._get_list(film, 'directors'))
                negatives['genres'].update(self._get_list(film, 'genres'))

            # Mild negative (2-2.5 stars) - track top-billed actors/themes
            if rating <= 2.5:
                negatives['actors'].update(self._get_list(film, 'cast', limit=3))
                negatives['themes'].update(self._get_list(film, 'themes'))

        return negatives

    def _apply_negative_filter(
        self,
        candidates: list[tuple[str, float, list[str], list[str]]],
        negatives: dict[str, set],
        hard_filter: bool = False
    ) -> list[tuple[str, float, list[str], list[str]]]:
        """
        Penalize or filter candidates matching negative preferences.
        """
        if not candidates or not negatives:
            return candidates

        filtered: list[tuple[str, float, list[str], list[str]]] = []
        for slug, score, reasons, warnings in candidates:
            film = self.films.get(slug)
            if not film:
                continue

            penalty = 0.0
            warning_list = list(warnings)

            film_directors = set(self._get_list(film, 'directors'))
            film_genres = set(self._get_list(film, 'genres'))

            if film_directors & negatives['directors']:
                if hard_filter:
                    continue
                penalty += 2.0
                warning_list.append("⚠️ Director you've disliked")

            genre_overlap = film_genres & negatives['genres']
            if len(genre_overlap) >= 2:
                penalty += 0.5 * len(genre_overlap)

            adjusted_score = score - penalty
            if adjusted_score > 0:
                filtered.append((slug, adjusted_score, reasons, warning_list))

        return filtered

    def inject_serendipity(
        self,
        ranked_candidates: list[tuple[str, float, list[str], list[str]]],
        profile: UserProfile,
        n: int,
        serendipity_factor: float = SERENDIPITY_FACTOR
    ) -> list[tuple[str, float, list[str], list[str]]]:
        """
        Replace some top recommendations with high-quality surprises.
        
        Serendipitous picks are films that:
        1. Score moderately (not hated, but not obvious picks)
        2. Have high community ratings (quality floor)
        3. Introduce underexplored attributes (genres, countries, decades user hasn't seen much of)
        """
        # If we don't have enough candidates, skip serendipity to avoid truncation.
        if not ranked_candidates or n <= 0:
            return []

        total_candidates = len(ranked_candidates)
        n_serendipitous = max(1, int(n * serendipity_factor)) if serendipity_factor > 0 else 0
        n_core = n - n_serendipitous
        
        # Take top core recommendations
        core_recs = ranked_candidates[:n_core]
        core_slugs = {c[0] for c in core_recs}
        
        # Find serendipitous candidates from mid-tier based on percentile window with rank guards
        start_pct, end_pct = SERENDIPITY_PERCENTILE_WINDOW
        start_idx = max(n_core, int(total_candidates * start_pct), SERENDIPITY_MIN_RANK)
        end_idx = min(int(total_candidates * end_pct), SERENDIPITY_MAX_RANK, total_candidates)
        if end_idx <= start_idx:
            start_idx = min(start_idx, total_candidates)
            end_idx = total_candidates
        serendipity_pool = [
            c for c in ranked_candidates[start_idx:end_idx]
            if c[0] not in core_slugs
        ]

        # Nothing to inject? Just return the top-N as-is.
        if not serendipity_pool:
            return ranked_candidates[:n]

        # Cap serendipity picks by available pool and requested size.
        n_serendipitous = min(n_serendipitous, len(serendipity_pool), max(0, n))
        n_core = max(0, n - n_serendipitous)
        core_recs = ranked_candidates[:n_core]
        core_slugs = {c[0] for c in core_recs}

        # Precompute averages to reward relative novelty for broad-taste users
        avg_genre_count = (
            sum(profile.genre_counts.values()) / len(profile.genre_counts)
            if profile.genre_counts else 0.0
        )
        avg_country_count = (
            sum(profile.country_counts.values()) / len(profile.country_counts)
            if profile.country_counts else 0.0
        )
        
        # Score for serendipity value
        serendipity_scored = []
        for slug, score, reasons, warnings in serendipity_pool:
            film = self.films.get(slug)
            if not film:
                continue
            
            # Quality floor
            avg_rating = film.get('avg_rating') or 0
            if avg_rating < SERENDIPITY_MIN_RATING:
                continue
            
            # Compute novelty: how different is this from user's typical viewing?
            novelty = 0.0
            
            # Genre novelty
            film_genres = set(self._get_list(film, 'genres'))
            for genre in film_genres:
                count = profile.genre_counts.get(genre, 0)
                if count < 3:  # Underexplored genre
                    novelty += 1.0
                elif count < 10:
                    novelty += 0.3
                elif avg_genre_count:
                    relative_gap = max(0.0, 1 - (count / avg_genre_count))
                    novelty += relative_gap * SERENDIPITY_RELATIVE_NOVELTY_WEIGHT
            
            # Country novelty
            film_countries = self._get_list(film, 'countries')
            for country in film_countries[:1]:  # Primary country
                if profile.country_counts.get(country, 0) < 5:
                    novelty += 1.5
                elif avg_country_count:
                    relative_gap = max(0.0, 1 - (profile.country_counts.get(country, 0) / avg_country_count))
                    novelty += relative_gap * SERENDIPITY_RELATIVE_NOVELTY_WEIGHT
            
            # Decade novelty
            year = film.get('year')
            if year:
                decade = (year // 10) * 10
                if profile.decades.get(decade, 0) < 0.5:
                    novelty += 0.5
            
            # Prefer less popular (hidden gems)
            rating_count = film.get('rating_count') or 0
            if rating_count < SERENDIPITY_POPULARITY_CAP:
                novelty += 0.5
            
            serendipity_scored.append((slug, score, reasons + ["🎲 Discovery pick"], warnings, novelty))
        
        # Select diverse serendipitous picks
        serendipity_scored.sort(key=lambda x: -x[4])  # Sort by novelty
        serendipity_picks = [(s[0], s[1], s[2], s[3]) for s in serendipity_scored[:n_serendipitous]]
        
        # Interleave serendipitous picks throughout results
        result = list(core_recs)
        for i, pick in enumerate(serendipity_picks):
            # Insert at positions 5, 10, 15... to mix with core recommendations
            insert_pos = min((i + 1) * 5, len(result))
            result.insert(insert_pos, pick)

        # Backfill if we still have fewer than requested (can happen with small pools)
        if len(result) < n:
            used = {slug for slug, *_ in result}
            for slug, score, reasons, warnings in ranked_candidates:
                if slug in used:
                    continue
                result.append((slug, score, reasons, warnings))
                used.add(slug)
                if len(result) >= n:
                    break

        return result[:n]

    def recommend(
        self,
        user_films: list[dict],
        n: int = 20,
        min_year: int | None = None,
        max_year: int | None = None,
        genres: list[str] | None = None,
        exclude_genres: list[str] | None = None,
        min_rating: float | None = None,
        diversity: bool = False,
        max_per_director: int = 2,
        username: str | None = None,
        user_lists: list[dict] | None = None,
        profile: UserProfile | None = None,
        serendipity_factor: float | None = SERENDIPITY_FACTOR,
        weighting_mode: str = "absolute",
    ) -> list[Recommendation]:
        """Generate recommendations."""

        # Build user profile
        if profile is None:
            profile = build_profile(
                user_films,
                self.films,
                user_lists=user_lists,
                username=username,
                weighting_mode=weighting_mode,
            )
        
        # Get seen films
        seen = {f['slug'] for f in user_films}
        negatives = self._compute_negative_profile(user_films, self.films)
        
        # Score all unseen films
        candidates = []
        for slug, film in self.films.items():
            if slug in seen:
                continue
            
            # Apply hard filters
            year = film.get('year')
            if min_year and year and year < min_year:
                continue
            if max_year and year and year > max_year:
                continue
            
            film_genres = self._get_list(film, 'genres')
            # Genres are now stored lowercase, so normalize user input for comparison
            if genres:
                genres_lower = [g.lower() for g in genres]
                if not any(g in film_genres for g in genres_lower):
                    continue
            if exclude_genres:
                exclude_genres_lower = [g.lower() for g in exclude_genres]
                if any(g in film_genres for g in exclude_genres_lower):
                    continue
            
            if min_rating and film.get('avg_rating') and film['avg_rating'] < min_rating:
                continue
            
            # Score the film
            score, reasons, warnings = self._score_film(film, profile)

            if score > 0:
                candidates.append((slug, score, reasons, warnings))

        # Apply explicit negative preferences
        candidates = self._apply_negative_filter(candidates, negatives)

        # Sort by score
        candidates.sort(key=lambda x: -x[1])

        ranked_candidates = candidates

        # Optionally mix in serendipitous picks to add novelty
        if serendipity_factor and serendipity_factor > 0:
            ranked_candidates = self.inject_serendipity(
                ranked_candidates, profile, n, serendipity_factor
            )

        # Apply diversity if requested
        if diversity:
            return self._diversify(ranked_candidates, n, max_per_director)

        # Build results (standard mode)
        results = []
        for slug, score, reasons, warnings in ranked_candidates[:n]:
            film = self.films[slug]
            results.append(Recommendation(
                slug=slug,
                title=film.get('title', slug),
                year=film.get('year'),
                score=score,
                reasons=reasons[:3],  # top 3 reasons
                warnings=warnings[:2]  # top 2 warnings
            ))
        
        return results
    
    def recommend_from_candidates(
        self,
        user_films: list[dict],
        candidates: list[str],
        n: int = 20,
        profile: UserProfile | None = None,
        weighting_mode: str = "absolute",
    ) -> list[Recommendation]:
        """
        Score and rank a specific list of films (e.g. watchlist).

        Args:
            user_films: User's film interactions
            candidates: List of film slugs to score
            n: Number of recommendations to return
            profile: Optional pre-built UserProfile to avoid rebuilding
        """
        # Build or use provided profile
        if profile is None:
            profile = build_profile(
                user_films,
                self.films,
                weighting_mode=weighting_mode,
            )

        # Cold-start fallback: use TF-IDF similarity when profile is too sparse
        if profile.n_rated == 0 and profile.n_liked == 0:
            self._ensure_tfidf()
            anchor_slugs = [uf['slug'] for uf in user_films if uf.get('watched') or uf.get('watchlisted')]
            tfidf_scores = self._tfidf.score_to_centroid(anchor_slugs, candidates, top_k=n) if self._tfidf else []
            results = []
            for slug, score in tfidf_scores:
                film = self.films.get(slug)
                if not film:
                    continue
                results.append(Recommendation(
                    slug=slug,
                    title=film.get('title', slug),
                    year=film.get('year'),
                    score=score,
                    reasons=["Metadata similarity (cold-start)"]
                ))
            return results

        negatives = self._compute_negative_profile(user_films, self.films)

        scored_candidates = []
        for slug in candidates:
            if slug not in self.films:
                continue
            
            film = self.films[slug]
            score, reasons, warnings = self._score_film(film, profile)

            if score > 0:
                scored_candidates.append((slug, score, reasons, warnings))

        scored_candidates = self._apply_negative_filter(scored_candidates, negatives)
        # Sort by score
        scored_candidates.sort(key=lambda x: -x[1])

        results = []
        for slug, score, reasons, warnings in scored_candidates[:n]:
            film = self.films[slug]
            results.append(Recommendation(
                slug=slug,
                title=film.get('title', slug),
                year=film.get('year'),
                score=score,
                reasons=reasons[:3],
                warnings=warnings[:2]
            ))
        
        return results

    def find_gaps(
        self,
        user_films: list[dict],
        min_director_score: float = 2.0,
        limit_per_director: int = 3,
        min_year: int | None = None,
        max_year: int | None = None,
        weighting_mode: str = "absolute",
    ) -> dict[str, list[Recommendation]]:
        """Find unseen films from directors the user loves."""
        profile = build_profile(user_films, self.films, weighting_mode=weighting_mode)
        seen = {f['slug'] for f in user_films}

        # Identify high affinity directors
        favorite_directors = [d for d, s in profile.directors.items() if s >= min_director_score]

        if not favorite_directors:
            return {}

        gaps = {}

        # Process each director (no threading - CPU-bound work where GIL prevents speedup)
        for director in favorite_directors:
            # Find all films by this director
            director_films = []
            for slug, film in self.films.items():
                if slug in seen:
                    continue

                # Apply year filters
                year = film.get('year')
                if min_year and year and year < min_year:
                    continue
                if max_year and year and year > max_year:
                    continue

                film_directors = self._get_list(film, 'directors')
                if director in film_directors:
                    director_films.append(film)

            if not director_films:
                continue

            # Rank by community rating/popularity (using simple heuristic)
            # We want "essential" films, so rating count and avg rating matter
            ranked_films = []
            for film in director_films:
                # Score purely on "essentialness"
                score = 0
                if film.get('avg_rating'):
                    score += film['avg_rating']
                if film.get('rating_count'):
                    score += min(film['rating_count'] / 10000, 2.0)  # Cap popularity bonus

                ranked_films.append((film, score))

            ranked_films.sort(key=lambda x: -x[1])

            recs = []
            for film, score in ranked_films[:limit_per_director]:
                recs.append(Recommendation(
                    slug=film['slug'],
                    title=film.get('title', film['slug']),
                    year=film.get('year'),
                    score=score,
                    reasons=[f"Essential {director}"]
                ))

            if recs:
                gaps[director] = recs

        return gaps
    
    def _score_film(self, film: dict, profile: UserProfile) -> tuple[float, list[str], list[str]]:
        """
        Score a film against user profile using configuration-driven attribute scoring.
        Returns (score, list of reasons, list of warnings).
        """
        return self.scoring_engine.score(self, film, profile)
    
    def similar_to(self, slug: str, n: int = 10) -> list[Recommendation]:
        """Find films similar to a specific film (item-based)."""
        if slug not in self.films:
            return []

        target = self.films[slug]
        target_genres = set(self._get_list(target, 'genres'))
        target_directors = set(self._get_list(target, 'directors'))
        target_cast = set(self._get_list(target, 'cast')[:5])
        target_themes = set(self._get_list(target, 'themes'))
        target_countries = set(self._get_list(target, 'countries'))
        target_writers = set(self._get_list(target, 'writers'))

        target_year = target.get('year')
        target_decade = (
            (target_year // 10) * 10
            if isinstance(target_year, int) and target_year >= 1888
            else None
        )
        
        candidates = []
        for other_slug, film in self.films.items():
            if other_slug == slug:
                continue
            
            score = 0
            reasons = []
            
            # Genre overlap
            film_genres = set(self._get_list(film, 'genres'))
            genre_overlap = target_genres & film_genres
            score += len(genre_overlap) * 1.0
            
            # Same director
            film_directors = set(self._get_list(film, 'directors'))
            dir_overlap = target_directors & film_directors
            if dir_overlap:
                score += SIMILAR_DIRECTOR_BONUS
                reasons.append(f"Same director: {list(dir_overlap)[0]}")

            # Cast overlap
            film_cast = set(self._get_list(film, 'cast')[:5])
            cast_overlap = target_cast & film_cast
            score += len(cast_overlap) * SIMILAR_CAST_SCORE
            if cast_overlap:
                reasons.append(f"Shared cast: {list(cast_overlap)[0]}")

            # Theme overlap
            film_themes = set(self._get_list(film, 'themes'))
            theme_overlap = target_themes & film_themes
            score += len(theme_overlap) * 0.3

            # Country overlap
            film_countries = set(self._get_list(film, 'countries'))
            if target_countries & film_countries:
                score += 0.5

            # Writer overlap
            film_writers = set(self._get_list(film, 'writers'))
            writer_overlap = target_writers & film_writers
            if writer_overlap:
                score += 3.0
                reasons.append(f"Same writer: {list(writer_overlap)[0]}")

            # Same decade
            film_year = film.get('year')
            film_decade = (
                (film_year // 10) * 10
                if isinstance(film_year, int) and film_year >= 1888
                else None
            )

            if target_decade is not None and film_decade == target_decade:
                score += SIMILAR_DECADE_SCORE
            
            if score > 0:
                candidates.append((other_slug, score, reasons))
        
        candidates.sort(key=lambda x: -x[1])
        recs = [
            Recommendation(
                slug=s, 
                title=self.films[s].get('title', s),
                year=self.films[s].get('year'),
                score=sc,
                reasons=r[:2]
            )
            for s, sc, r in candidates[:n]
        ]

        # Cold-start / sparse fallback using TF-IDF embeddings
        if len(recs) < n:
            self._ensure_tfidf()
            if self._tfidf:
                tfidf_ranked = self._tfidf.rank_against(slug, list(self.films.keys()), top_k=n)
                for other_slug, score in tfidf_ranked:
                    if other_slug == slug or any(r.slug == other_slug for r in recs):
                        continue
                    film = self.films[other_slug]
                    recs.append(Recommendation(
                        slug=other_slug,
                        title=film.get('title', other_slug),
                        year=film.get('year'),
                        score=score,
                        reasons=["TF-IDF metadata similarity"]
                    ))
                    if len(recs) >= n:
                        break

        return recs[:n]
    
    def _diversify(self, candidates: list[tuple[str, float, list[str], list[str]]], n: int, max_per_director: int = 2) -> list[Recommendation]:
        """Select top n while limiting per-director concentration."""
        from collections import defaultdict

        results = []
        director_counts = defaultdict(int)

        for slug, score, reasons, warnings in candidates:
            film = self.films.get(slug)
            if not film:
                continue

            directors = self._get_list(film, 'directors')

            # Check if any director has hit the limit
            if any(director_counts[d] >= max_per_director for d in directors):
                continue

            # Add to results
            title = film.get('title', slug)
            year = film.get('year')
            results.append(Recommendation(
                slug=slug,
                title=title,
                year=year,
                score=score,
                reasons=reasons[:3],
                warnings=warnings[:2]
            ))
            
            # Update director counts
            for d in directors:
                director_counts[d] += 1
            
            if len(results) >= n:
                break

        # Warn if diversity constraints prevented reaching requested count
        if len(results) < n:
            warning_msg = (
                f"Diversity mode returned only {len(results)}/{n} results. "
                f"Director constraint (max {max_per_director} per director) limited options. "
                f"Consider increasing --max-per-director or disabling --diversity for more results."
            )
            logger.warning(warning_msg)
            logger.warning(f"\n⚠️  {warning_msg}")

        return results

    def explain_recommendation_detailed(
        self,
        film: dict,
        profile: UserProfile,
        user_films: list[dict]
    ) -> dict:
        """
        Generate detailed, human-readable explanation for why a film was recommended.
        """
        explanation = {
            "summary": "",
            "positive_factors": [],
            "negative_factors": [],
            "similar_films_you_liked": [],
            "confidence": 0.0,
            "discovery_potential": "",
        }

        score, reasons, warnings = self._score_film(film, profile)

        # Find similar films the user has liked
        film_directors = set(self._get_list(film, 'directors'))
        film_genres = set(self._get_list(film, 'genres'))

        similar_liked = []
        for uf in user_films:
            if (uf.get('rating') and uf['rating'] >= 4.0) or uf.get('liked'):
                other_film = self.films.get(uf['slug'])
                if not other_film:
                    continue

                other_directors = set(self._get_list(other_film, 'directors'))
                other_genres = set(self._get_list(other_film, 'genres'))

                overlap_reasons = []
                if film_directors & other_directors:
                    overlap_reasons.append(f"same director: {list(film_directors & other_directors)[0]}")
                if len(film_genres & other_genres) >= 2:
                    overlap_reasons.append("similar genres")

                if overlap_reasons:
                    similar_liked.append({
                        "title": other_film.get('title', uf['slug']),
                        "your_rating": uf.get('rating'),
                        "connection": ", ".join(overlap_reasons)
                    })

        explanation["similar_films_you_liked"] = similar_liked[:3]

        # Compute confidence based on how much data supports the recommendation
        supporting_observations = 0
        for d in film_directors:
            supporting_observations += profile.director_counts.get(d, 0)
        for g in film_genres:
            supporting_observations += profile.genre_counts.get(g, 0)

        explanation["confidence"] = min(1.0, supporting_observations / 20) if supporting_observations else 0.0

        # Discovery potential: is this outside their usual zone?
        genre_familiarity = sum(profile.genre_counts.get(g, 0) for g in film_genres)
        if genre_familiarity < 5:
            explanation["discovery_potential"] = "This explores genres you haven't watched much"
        elif genre_familiarity > 30:
            explanation["discovery_potential"] = "This is squarely in your comfort zone"
        else:
            explanation["discovery_potential"] = "A nice balance of familiar and new"

        # Build summary
        if similar_liked:
            explanation["summary"] = f"Based on your love of {similar_liked[0]['title']}"
        elif reasons:
            explanation["summary"] = reasons[0]
        else:
            explanation["summary"] = "Matches your overall taste profile"

        explanation["positive_factors"] = reasons
        explanation["negative_factors"] = warnings
        explanation["score"] = score

        return explanation

    def compute_recommendation_diversity(
        self,
        recommendations: list[Recommendation]
    ) -> dict:
        """
        Compute diversity metrics for a recommendation set.
        Helps ensure recommendations aren't too narrow.
        """
        if not recommendations:
            return {"diversity_score": 0.0}

        all_genres: list[str] = []
        all_directors: list[str] = []
        all_countries: list[str] = []
        all_decades: list[int] = []

        for rec in recommendations:
            film = self.films.get(rec.slug)
            if not film:
                continue

            all_genres.extend(self._get_list(film, 'genres'))
            all_directors.extend(self._get_list(film, 'directors'))
            all_countries.extend(self._get_list(film, 'countries')[:1])  # Primary only

            year = film.get('year')
            if year:
                all_decades.append((year // 10) * 10)

        def entropy(items: list) -> float:
            """Shannon entropy as diversity measure."""
            from collections import Counter
            if not items:
                return 0.0

            counts = Counter(items)
            total = len(items)
            probs = [c / total for c in counts.values()]
            return -sum(p * math.log2(p) for p in probs if p > 0)

        # Normalize entropies to 0-1 scale
        n = len(recommendations)
        max_entropy = math.log2(n) if n > 1 else 1.0

        genre_diversity = entropy(all_genres) / max_entropy if max_entropy > 0 else 0
        director_diversity = entropy(all_directors) / max_entropy if max_entropy > 0 else 0
        country_diversity = entropy(all_countries) / max_entropy if max_entropy > 0 else 0
        decade_diversity = entropy(all_decades) / max_entropy if max_entropy > 0 else 0

        # Weighted overall score
        overall = (
            genre_diversity * 0.3 +
            director_diversity * 0.3 +
            country_diversity * 0.2 +
            decade_diversity * 0.2
        )

        return {
            "diversity_score": round(overall, 3),
            "genre_diversity": round(genre_diversity, 3),
            "director_diversity": round(director_diversity, 3),
            "country_diversity": round(country_diversity, 3),
            "decade_diversity": round(decade_diversity, 3),
            "unique_genres": len(set(all_genres)),
            "unique_directors": len(set(all_directors)),
            "unique_countries": len(set(all_countries)),
            "decade_range": (min(all_decades), max(all_decades)) if all_decades else None,
        }


class CollaborativeRecommender:
    """
    Collaborative filtering recommender.
    Finds users with similar taste and recommends films they liked.

    Uses sparse matrices for efficient similarity computation on large datasets.
    """

    def __init__(self, all_user_films: dict[str, list[dict]], film_metadata: dict[str, dict] | None = None):
        """
        Args:
            all_user_films: Dict mapping username -> list of user_films dicts
            film_metadata: Optional dict mapping slug -> film metadata dict for filtering and display
        """
        self.all_user_films = all_user_films
        self.films = film_metadata or {}

        # Precompute user-item matrix for efficient similarity computation
        self._user_matrix = None
        self._user_index = None
        self._film_index = None
        self._normalized_matrix = None  # Cached normalized matrix for fast similarity
        self._overlap_matrix = None      # Cached binary overlap matrix
        self._item_similarities: dict[str, list[tuple[str, float]]] = {}
        self._item_sim_top_k = 50

        self._build_sparse_matrix()
        self._precompute_similarity_components()
        self._fingerprint = self._compute_fingerprint()
        self._maybe_load_item_similarity_cache()

    def _build_sparse_matrix(self):
        """
        Build sparse user-item rating matrix for efficient similarity computation.

        Creates a CSR matrix where rows are users and columns are films.
        Also builds index mappings for fast lookups.
        """
        from scipy.sparse import csr_matrix
        import numpy as np

        # Build user and film indexes
        usernames = list(self.all_user_films.keys())
        self._user_index = {username: idx for idx, username in enumerate(usernames)}

        # Collect all unique films
        all_films_set = set()
        for films in self.all_user_films.values():
            for film in films:
                all_films_set.add(film['slug'])

        all_films_list = list(all_films_set)
        self._film_index = {slug: idx for idx, slug in enumerate(all_films_list)}

        # Build sparse matrix (users × films)
        n_users = len(usernames)
        n_films = len(all_films_list)

        # Use COO format for building, then convert to CSR
        row_indices = []
        col_indices = []
        ratings = []

        for username, user_films in self.all_user_films.items():
            user_idx = self._user_index[username]
            for film in user_films:
                rating = film.get('rating')
                if rating:  # Only include rated films
                    film_idx = self._film_index.get(film['slug'])
                    if film_idx is not None:
                        row_indices.append(user_idx)
                        col_indices.append(film_idx)
                        ratings.append(rating)

        # Create sparse matrix
        if row_indices:
            self._user_matrix = csr_matrix(
                (ratings, (row_indices, col_indices)),
                shape=(n_users, n_films),
                dtype=np.float32
            )
        else:
            # Empty matrix if no ratings
            self._user_matrix = csr_matrix((n_users, n_films), dtype=np.float32)

        logger.debug(f"Built sparse user-item matrix: {n_users} users × {n_films} films, {len(ratings)} ratings")

    def _precompute_similarity_components(self):
        """
        Precompute normalized matrix and overlap matrix for fast similarity computation.

        This method performs the expensive mean-centering and normalization operations once,
        dramatically speeding up similarity computations in _find_neighbors.
        """
        import numpy as np
        from scipy.sparse import diags

        if self._user_matrix is None or self._user_matrix.nnz == 0:
            logger.debug("No ratings to precompute similarity components")
            return

        n_users = self._user_matrix.shape[0]

        # Compute row means efficiently: sum / count
        row_sums = np.array(self._user_matrix.sum(axis=1)).flatten()
        row_nnz = np.array(self._user_matrix.getnnz(axis=1), dtype=np.float64)
        row_nnz[row_nnz == 0] = 1  # Avoid division by zero
        row_means = row_sums / row_nnz

        # Mean-center the matrix: subtract row mean from each non-zero entry
        centered = self._user_matrix.copy().tocsr()
        for i in range(n_users):
            start, end = centered.indptr[i], centered.indptr[i + 1]
            centered.data[start:end] -= row_means[i]

        # Normalize to unit length (for cosine similarity)
        row_norms = np.sqrt(np.array(centered.power(2).sum(axis=1)).flatten())
        row_norms[row_norms == 0] = 1  # Avoid division by zero
        inv_norms = diags(1.0 / row_norms)
        self._normalized_matrix = inv_norms @ centered

        # Binary overlap matrix (which films each user has rated)
        self._overlap_matrix = (self._user_matrix > 0).astype(np.float32)

        logger.debug(f"Precomputed similarity components for {n_users} users")

    def _compute_fingerprint(self) -> dict:
        n_users = len(self.all_user_films)
        n_ratings = int(self._user_matrix.nnz) if self._user_matrix is not None else 0
        n_items = len(self._film_index) if self._film_index else 0
        return {
            "n_users": n_users,
            "n_items": n_items,
            "n_ratings": n_ratings,
        }

    def _maybe_load_item_similarity_cache(self):
        if not ITEM_SIM_CACHE_PATH.exists():
            return
        try:
            payload = json.loads(ITEM_SIM_CACHE_PATH.read_text())
            if payload.get("fingerprint") != self._fingerprint:
                return
            if payload.get("top_k") != self._item_sim_top_k:
                return
            data = payload.get("items", {})
            self._item_similarities = {
                slug: [(entry["slug"], entry["score"]) for entry in entries]
                for slug, entries in data.items()
            }
            logger.info(f"Loaded item-item similarity cache ({len(self._item_similarities)} items)")
        except Exception as e:
            logger.warning(f"Failed to load item similarity cache: {e}")

    def _save_item_similarity_cache(self):
        try:
            ITEM_SIM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "fingerprint": self._fingerprint,
                "top_k": self._item_sim_top_k,
                "items": {
                    slug: [{"slug": s, "score": float(sc)} for s, sc in sims]
                    for slug, sims in self._item_similarities.items()
                },
            }
            ITEM_SIM_CACHE_PATH.write_text(json.dumps(payload))
            logger.info(f"Saved item similarity cache to {ITEM_SIM_CACHE_PATH}")
        except Exception as e:
            logger.warning(f"Could not save item similarity cache: {e}")

    def _compute_item_similarities(self):
        """
        Compute item similarity using rating patterns, not just co-occurrence.

        Films are similar if users who rate one highly also rate the other highly.
        """
        import numpy as np
        from collections import defaultdict

        if not self.all_user_films:
            return

        logger.info("Computing item-item similarity cache...")

        # Build item -> [(user, rating)] mapping
        item_ratings: dict[str, list[tuple[int, float]]] = defaultdict(list)

        for username, films in self.all_user_films.items():
            user_idx = self._user_index[username]
            for f in films:
                rating = f.get('rating')
                if rating:
                    item_ratings[f['slug']].append((user_idx, rating))

        # Filter to reduce compute: drop very sparse items and cap total
        item_ratings = {
            slug: ratings for slug, ratings in item_ratings.items()
            if len(ratings) >= ITEM_SIM_MIN_RATINGS
        }
        if not item_ratings:
            logger.info("Item similarity: no items meet minimum rating count")
            return

        if len(item_ratings) > ITEM_SIM_MAX_ITEMS:
            sorted_items = sorted(item_ratings.items(), key=lambda x: -len(x[1]))
            dropped = len(sorted_items) - ITEM_SIM_MAX_ITEMS
            item_ratings = dict(sorted_items[:ITEM_SIM_MAX_ITEMS])
            logger.info(
                f"Item similarity: trimming items from {len(sorted_items)} to "
                f"{ITEM_SIM_MAX_ITEMS} (dropped {dropped}) to cap computation"
            )

        # Compute adjusted cosine similarity between items (ratings centered by user mean)
        user_means = {}
        for username, films in self.all_user_films.items():
            ratings = [f['rating'] for f in films if f.get('rating')]
            if ratings:
                user_means[self._user_index[username]] = sum(ratings) / len(ratings)

        film_slugs = list(item_ratings.keys())
        sim_map: dict[str, list[tuple[str, float]]] = defaultdict(list)

        for i, slug_a in enumerate(film_slugs):
            ratings_a = item_ratings[slug_a]
            users_a = {u for u, _ in ratings_a}

            for slug_b in film_slugs[i + 1:]:
                ratings_b = item_ratings[slug_b]
                users_b = {u for u, _ in ratings_b}

                common_users = users_a & users_b
                # Allow similarity computation for small communities; need at least 2 overlaps
                if len(common_users) < 2:
                    continue

                ratings_a_dict = dict(ratings_a)
                ratings_b_dict = dict(ratings_b)

                dot_product = 0.0
                norm_a = 0.0
                norm_b = 0.0

                for user in common_users:
                    mean = user_means.get(user, 3.0)
                    adj_a = ratings_a_dict[user] - mean
                    adj_b = ratings_b_dict[user] - mean

                    dot_product += adj_a * adj_b
                    norm_a += adj_a ** 2
                    norm_b += adj_b ** 2

                if norm_a > 0 and norm_b > 0:
                    sim = dot_product / (np.sqrt(norm_a) * np.sqrt(norm_b))
                    if sim > 0.3:
                        sim_map[slug_a].append((slug_b, sim))
                        sim_map[slug_b].append((slug_a, sim))

        for slug, sims in sim_map.items():
            sims.sort(key=lambda x: -x[1])
            self._item_similarities[slug] = sims[: self._item_sim_top_k]

        self._save_item_similarity_cache()

    def _item_recommend_from_anchors(self, anchors: list[str], seen: set[str], top_k: int = 100) -> list[tuple[str, float, list[str]]]:
        if not anchors:
            return []
        if not self._item_similarities:
            self._compute_item_similarities()
        scores = {}
        reasons = {}
        for anchor in anchors:
            neighbors = self._item_similarities.get(anchor, [])
            for slug, sim in neighbors:
                if slug in seen:
                    continue
                scores[slug] = scores.get(slug, 0.0) + sim
                reasons.setdefault(slug, []).append(f"Similar to {anchor}")
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [(slug, score, reasons.get(slug, [])[:2]) for slug, score in ranked[:top_k]]

    def recommend(
        self,
        username: str,
        n: int = 20,
        min_neighbors: int = 3,
        min_year: int | None = None,
        max_year: int | None = None,
        genres: list[str] | None = None,
        exclude_genres: list[str] | None = None,
    ) -> list[Recommendation]:
        """Generate collaborative recommendations."""
        
        if username not in self.all_user_films:
            return []
        
        target_films = self.all_user_films[username]
        
        # Find neighbors (users with similar taste)
        neighbor_k = max(10, min_neighbors * 2)
        influencers, _ = self._find_neighbors_asymmetric(username, k=neighbor_k)
        base_neighbors = self._find_neighbors(username, target_films, k=neighbor_k)

        neighbor_scores: dict[str, float] = {}
        for user, sim in influencers:
            neighbor_scores[user] = sim
        for user, sim in base_neighbors:
            if user not in neighbor_scores or sim > neighbor_scores[user]:
                neighbor_scores[user] = sim

        neighbors = sorted(neighbor_scores.items(), key=lambda x: -x[1])[:neighbor_k]
        
        if len(neighbors) < min_neighbors:
            logger.warning(f"Warning: Only found {len(neighbors)} neighbors (min: {min_neighbors})")
        
        # Get films seen by target
        seen = {f['slug'] for f in target_films}
        
        # Score unseen films based on neighbor ratings
        film_scores = {}
        film_reasons = {}
        
        for neighbor_user, similarity in neighbors:
            neighbor_films = self.all_user_films[neighbor_user]
            
            for interaction in neighbor_films:
                slug = interaction['slug']
                if slug in seen:
                    continue
                
                # Apply filters if metadata available
                if self.films and slug in self.films:
                    film_meta = self.films[slug]
                    year = film_meta.get('year')
                    if min_year and year and year < min_year:
                        continue
                    if max_year and year and year > max_year:
                        continue

                    # Apply genre filters (genres are stored lowercase)
                    film_genres = load_json(film_meta.get('genres'))
                    if genres:
                        genres_lower = [g.lower() for g in genres]
                        if not any(g in film_genres for g in genres_lower):
                            continue
                    if exclude_genres:
                        exclude_genres_lower = [g.lower() for g in exclude_genres]
                        if any(g in film_genres for g in exclude_genres_lower):
                            continue
                
                rating = interaction.get('rating')  # Use original variable
                liked = interaction.get('liked', False)
                watched = interaction.get('watched', False)
                watchlisted = interaction.get('watchlisted', False)
                
                # Score based on rating or implicit feedback
                if rating is not None:
                    score = (rating - 2.5) * similarity  # normalize around mid-point
                else:
                    implicit = 0.0
                    if liked:
                        implicit += COLLAB_IMPLICIT_WEIGHTS["liked"]
                    if watched:
                        implicit += COLLAB_IMPLICIT_WEIGHTS["watched"]
                    if watchlisted:
                        implicit += COLLAB_IMPLICIT_WEIGHTS["watchlisted"]
                    if implicit == 0:
                        implicit = 0.05
                    score = implicit * similarity

                # Popularity debias: down-weight very popular films
                if self.films and slug in self.films:
                    rating_count = self.films[slug].get('rating_count') or 0
                else:
                    rating_count = 0
                if rating_count:
                    penalty = COLLAB_POPULARITY_DEBIAS * (math.log1p(rating_count) / math.log1p(100_000))
                    score *= max(0.1, 1 - penalty)
                
                if slug not in film_scores:
                    film_scores[slug] = 0
                    film_reasons[slug] = []
                
                film_scores[slug] += score
                
                # Track who recommended it
                if score > 0.5 and len(film_reasons[slug]) < 3:
                    film_reasons[slug].append(f"Liked by {neighbor_user}")

        # Item-based fallback for sparse neighborhoods
        if len(neighbors) < min_neighbors:
            anchor_slugs = [
                f['slug'] for f in target_films
                if (f.get('rating') and f['rating'] >= 3.5) or f.get('liked')
            ]
            item_based = self._item_recommend_from_anchors(anchor_slugs, seen, top_k=n * 2)
            for slug, score, reasons in item_based:
                film_scores[slug] = film_scores.get(slug, 0.0) + score
                film_reasons.setdefault(slug, []).extend(reasons)
        
        # Sort by score
        ranked = sorted(film_scores.items(), key=lambda x: -x[1])
        
        # Build results with film metadata if available
        results = []
        for slug, score in ranked[:n]:
            if self.films and slug in self.films:
                film = self.films[slug]
                title = film.get('title', slug)
                year = film.get('year')
            else:
                title = slug
                year = None
            
            results.append(Recommendation(
                slug=slug,
                title=title,
                year=year,
                score=score,
                reasons=film_reasons.get(slug, [])[:3]
            ))
        
        return results
    
    def _find_neighbors(self, username: str, target_films: list[dict], k: int = 10) -> list[tuple[str, float]]:
        """
        Find k most similar users using precomputed matrices.

        Uses adjusted cosine similarity (mean-centered ratings) which approximates
        Pearson correlation but is much faster for sparse matrices thanks to precomputation.

        Args:
            username: Target username
            target_films: Target user's film interactions (unused, kept for API compatibility)
            k: Number of neighbors to return

        Returns:
            List of (username, similarity_score) tuples, sorted by score descending
        """
        import numpy as np

        if username not in self._user_index or self._normalized_matrix is None:
            return []

        target_idx = self._user_index[username]
        n_users = self._normalized_matrix.shape[0]

        # Similarity = normalized_matrix @ target_row.T (single sparse matrix-vector multiply)
        target_row = self._normalized_matrix[target_idx]
        similarities = np.asarray(self._normalized_matrix @ target_row.T).ravel()

        # Overlap counts for confidence weighting
        target_binary = self._overlap_matrix[target_idx]
        # Force dense array to avoid sparse truthiness/boolean issues downstream
        overlap_vec = self._overlap_matrix @ target_binary.T
        overlaps = np.asarray(overlap_vec.toarray()).ravel()

        # Filter and weight (be lenient on tiny datasets)
        min_overlap = 2 if n_users < 20 or self._overlap_matrix.shape[1] < 50 else 5
        valid = (overlaps >= min_overlap) & (np.arange(n_users) != target_idx)
        confidence = np.minimum(overlaps / 20.0, 1.0)  # Full confidence at 20+ common films
        shrinkage = overlaps / (overlaps + COLLAB_SHRINKAGE)

        # Apply confidence weighting and filter invalid
        weighted = np.where(valid, similarities * confidence * shrinkage, -np.inf)

        # Get top-k using partial sort (more efficient than full sort)
        if k >= len(weighted):
            top_k = np.argsort(weighted)[::-1]
        else:
            top_k = np.argpartition(weighted, -k)[-k:]
            top_k = top_k[np.argsort(weighted[top_k])[::-1]]

        # Filter to positive similarities only
        top_k = [i for i in top_k if weighted[i] > 0]

        # Build result list
        usernames = list(self._user_index.keys())
        return [(usernames[i], weighted[i]) for i in top_k]

    def _find_neighbors_asymmetric(
        self,
        username: str,
        k: int = 10
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """
        Find neighbors with asymmetric similarity.

        Returns:
            influencers: Users whose taste predicts yours (more experienced, you tend to agree)
            followers: Users who tend to agree with your ratings (less experienced, follow your taste)
        """
        if username not in self._user_index:
            return [], []

        target_films = self.all_user_films[username]
        target_ratings = {f['slug']: f.get('rating') for f in target_films if f.get('rating') is not None}

        if not target_ratings:
            return [], []

        influencer_scores: dict[str, float] = {}
        follower_scores: dict[str, float] = {}

        for other_username, other_films in self.all_user_films.items():
            if other_username == username:
                continue

            other_ratings = {f['slug']: f.get('rating') for f in other_films if f.get('rating') is not None}
            common = set(target_ratings.keys()) & set(other_ratings.keys())

            # Permit very small overlaps for sparse datasets
            if len(common) < 1:
                continue

            # Compute agreement score
            agreements = []
            for slug in common:
                diff = abs(target_ratings[slug] - other_ratings[slug])
                agreements.append(1.0 - (diff / 4.5))  # Normalize to 0-1

            agreement_score = sum(agreements) / len(agreements)
            overlap_factor = len(common) / 20  # Scale by overlap size

            # Experience ratio determines influencer vs follower
            target_experience = len(target_ratings)
            other_experience = len(other_ratings)
            experience_ratio = other_experience / max(target_experience, 1)

            if experience_ratio > 1.0:
                # They have more experience -> potential influencer
                influencer_weight = min(experience_ratio, 2.0)
                influencer_scores[other_username] = agreement_score * influencer_weight * overlap_factor
            else:
                # They have less experience -> potential follower
                follower_weight = min(1.0 / max(experience_ratio, 0.1), 2.0)
                follower_scores[other_username] = agreement_score * follower_weight * overlap_factor

        sorted_influencers = sorted(influencer_scores.items(), key=lambda x: -x[1])[:k]
        sorted_followers = sorted(follower_scores.items(), key=lambda x: -x[1])[:k]

        return sorted_influencers, sorted_followers

@dataclass
class ExplainedRecommendation(Recommendation):
    """Extended recommendation with interpretable explanations."""
    contribution_breakdown: dict[str, float] = field(default_factory=dict)
    counterfactuals: list[str] = field(default_factory=list)
    confidence: float = 0.0

    @staticmethod
    def explain_recommendation(
        recommender: "MetadataRecommender",
        film: dict,
        profile: UserProfile,
        seen_films: set[str] | None = None,
    ) -> "ExplainedRecommendation":
        """Generate detailed explanation for a single film recommendation."""

        _ = seen_films  # reserved for future counterfactual comparisons
        score, reasons, warnings = recommender._score_film(film, profile)

        # Decompose score by attribute type
        contributions = {}
        for config in ATTRIBUTE_CONFIGS:
            attr_score, _, _ = recommender._score_attribute(film, profile, config)
            if abs(attr_score) > 0.01:
                contributions[config.name] = round(attr_score, 2)

        # Generate counterfactuals
        counterfactuals = []

        # "If you hadn't liked X director..."
        film_directors = load_json(film.get('directors'))
        for director in film_directors:
            if director in profile.directors and profile.directors[director] > 1.0:
                learned_weight = (
                    recommender.feature_weights.factor("director", director)
                    if getattr(recommender, "feature_weights", None)
                    else 1.0
                )
                hypothetical_score = score - (
                    profile.directors[director] * WEIGHTS['director'] * learned_weight
                )
                if hypothetical_score < score * 0.5:
                    counterfactuals.append(
                        f"Without your {director} affinity, score would drop to {hypothetical_score:.1f}"
                    )

        # Confidence based on profile observation counts
        relevant_counts = []
        for director in film_directors:
            if director in profile.director_counts:
                relevant_counts.append(profile.director_counts[director])

        confidence = min(1.0, sum(relevant_counts) / 10) if relevant_counts else 0.3

        return ExplainedRecommendation(
            slug=film['slug'],
            title=film.get('title', film['slug']),
            year=film.get('year'),
            score=score,
            reasons=reasons[:3],
            warnings=warnings[:2],
            contribution_breakdown=contributions,
            counterfactuals=counterfactuals[:2],
            confidence=confidence
        )

