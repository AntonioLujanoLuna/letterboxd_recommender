"""Metadata-based recommender implementation."""

import logging
from typing import TYPE_CHECKING
from pathlib import Path

from ..database import load_json
from ..profile import UserProfile, build_profile
from ..feature_weights import FeatureWeights, load_feature_weights
from ..config import (
    WEIGHTS,
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
    NEGATIVE_PENALTY_MULTIPLIER,
    NEGATIVE_PENALTY_MULTIPLIERS,
    CONFIDENCE_MIN_SAMPLES,
    SIMILAR_DIRECTOR_BONUS,
    SIMILAR_CAST_SCORE,
    SIMILAR_DECADE_SCORE,
)
from .recommendation import Recommendation
from .scoring import AttributeConfig, ATTRIBUTE_CONFIGS, DEFAULT_SCORING_RULES, _confidence_weight
from .engine import ScoringEngine
from .vectorization import TfidfEmbedder

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

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
            from ..database import load_idf
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

        for idx, value in enumerate(film_values):
            if value not in profile_scores:
                continue
            value_score = profile_scores[value]
            count = profile_counts.get(value, 1)
            learned_weight = (
                self.feature_weights.factor(config.name, value)
                if self.feature_weights
                else 1.0
            )
            item_weight = (
                config.secondary_weight if config.secondary_weight is not None and idx > 0 else 1.0
            )
            adjusted_score = value_score * learned_weight * item_weight

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
                warning_list.append("Director you've disliked")

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

            serendipity_scored.append((slug, score, reasons + ["Discovery pick"], warnings, novelty))

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
            logger.warning(f"\n  {warning_msg}")

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
        import math

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
        import math

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
