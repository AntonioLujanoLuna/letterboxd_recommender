import logging
from dataclasses import dataclass
from typing import Optional
from .profile import UserProfile, build_profile
from .database import load_json
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
    NEGATIVE_THRESHOLD_DIRECTOR,
    NEGATIVE_THRESHOLD_GENRE,
    NEGATIVE_THRESHOLD_ACTOR,
    NEGATIVE_THRESHOLD_WRITER,
    NEGATIVE_THRESHOLD_CINE,
    NEGATIVE_THRESHOLD_COMPOSER,
    CONFIDENCE_MIN_SAMPLES,
    USE_IDF_WEIGHTING,
    IDF_DISTINCTIVE_THRESHOLD,
)

logger = logging.getLogger(__name__)


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

    def __init__(self, all_films: list[dict], use_idf: bool = USE_IDF_WEIGHTING):
        self.films = {f['slug']: f for f in all_films}
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
        # Load film attribute values
        film_values = load_json(film.get(config.film_field, []))

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

            # Apply confidence weighting
            confidence = _confidence_weight(count, CONFIDENCE_MIN_SAMPLES.get(config.name, 5))

            # Apply IDF weighting if enabled and configured
            idf_weight = 1.0
            if self.use_idf and config.idf_type and config.idf_type in self.idf:
                idf_weight = self.idf[config.idf_type].get(value, 1.0)

            # Handle negative scores with amplified penalty
            if value_score < 0:
                total_score += value_score * NEGATIVE_PENALTY_MULTIPLIER * confidence * idf_weight
                if config.negative_threshold != 0.0 and value_score < config.negative_threshold:
                    warnings.append(config.warning_template.format(value))
            else:
                total_score += value_score * confidence * idf_weight

                # Track positive matches for reasons
                if value_score > config.match_threshold:
                    # Check if distinctive (high IDF)
                    if self.use_idf and config.idf_type and idf_weight > IDF_DISTINCTIVE_THRESHOLD:
                        distinctive_items.append(value)
                    else:
                        matched_items.append(value)

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
    ) -> list[Recommendation]:
        """Generate recommendations."""

        # Build user profile
        profile = build_profile(user_films, self.films, user_lists=user_lists, username=username)
        
        # Get seen films
        seen = {f['slug'] for f in user_films}
        
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
            
            film_genres = load_json(film.get('genres'))
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

        # Sort by score
        candidates.sort(key=lambda x: -x[1])

        # Apply diversity if requested
        if diversity:
            return self._diversify(candidates, n, max_per_director)

        # Build results (standard mode)
        results = []
        for slug, score, reasons, warnings in candidates[:n]:
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
            profile = build_profile(user_films, self.films)

        scored_candidates = []
        for slug in candidates:
            if slug not in self.films:
                continue
            
            film = self.films[slug]
            score, reasons, warnings = self._score_film(film, profile)

            if score > 0:
                scored_candidates.append((slug, score, reasons, warnings))

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
        max_year: int | None = None
    ) -> dict[str, list[Recommendation]]:
        """Find unseen films from directors the user loves."""
        profile = build_profile(user_films, self.films)
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

                film_directors = load_json(film.get('directors'))
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
        score = 0.0
        reasons = []
        warnings = []

        # Score all configured attributes using generic method
        for config in ATTRIBUTE_CONFIGS:
            attr_score, attr_reasons, attr_warnings = self._score_attribute(film, profile, config)
            score += attr_score
            reasons.extend(attr_reasons)
            warnings.extend(attr_warnings)

        # Special case: Directors with low confidence get annotated reasons
        film_directors = load_json(film.get('directors'))
        for d in film_directors:
            if d in profile.directors and profile.directors[d] > MATCH_THRESHOLD_DIRECTOR:
                count = profile.director_counts.get(d, 1)
                confidence = _confidence_weight(count, CONFIDENCE_MIN_SAMPLES['director'])
                if confidence < 0.7:
                    # Replace generic director reason with annotated version
                    for i, reason in enumerate(reasons):
                        if reason.startswith(f"Director: {d}"):
                            reasons[i] = f"Director: {d} (based on {count} film{'s' if count > 1 else ''})"
                            break

        # Genre pair matching (co-occurrence preferences)
        film_genres = load_json(film.get('genres'))
        pair_score = 0.0
        matched_pairs = []
        for i, g1 in enumerate(film_genres):
            for g2 in film_genres[i+1:]:
                pair = "|".join(sorted([g1, g2]))
                if pair in profile.genre_pairs:
                    pair_value = profile.genre_pairs[pair]
                    # Apply negative penalty multiplier for negative pair preferences
                    if pair_value < 0:
                        pair_score += pair_value * NEGATIVE_PENALTY_MULTIPLIER
                        if pair_value < -0.5:  # Threshold for reporting negative pairs
                            warnings.append(f"⚠️ Genre combo: {g1}+{g2} (disliked)")
                    else:
                        pair_score += pair_value
                        if pair_value > 0.5:  # Threshold for reporting positive pairs
                            matched_pairs.append(f"{g1}+{g2}")

        score += pair_score * WEIGHTS.get('genre_pair', 0.6)
        if matched_pairs:
            reasons.append(f"Genre combo: {matched_pairs[0]}")

        # Decade match
        year = film.get('year')
        if year:
            decade = (year // 10) * 10
            if decade in profile.decades:
                score += profile.decades[decade] * WEIGHTS['decade']

        # Country match (with primary/secondary weighting)
        film_countries = load_json(film.get('countries', []))
        for i, country in enumerate(film_countries):
            if country in profile.countries:
                country_score = profile.countries[country]
                count = profile.country_counts.get(country, 1)
                confidence = _confidence_weight(count, CONFIDENCE_MIN_SAMPLES['country'])

                # Primary country gets full weight, secondary reduced
                country_weight = WEIGHTS['country'] if i == 0 else WEIGHTS['country'] * self.COUNTRY_SECONDARY_WEIGHT
                score += country_score * country_weight * confidence
                if i == 0 and country_score > 0.5:
                    reasons.append(f"Country: {country}")

        # Community rating bonus
        # Favor films rated similarly to user's liked films
        avg = film.get('avg_rating')
        if avg and profile.avg_liked_rating:
            # Bonus for films near user's sweet spot
            rating_diff = abs(avg - profile.avg_liked_rating)
            if rating_diff < RATING_DIFF_HIGH:
                score += 1.0 * WEIGHTS['community_rating']
                reasons.append(f"Highly rated ({avg:.1f}★)")
            elif rating_diff < RATING_DIFF_MED:
                score += 0.5 * WEIGHTS['community_rating']

        # Slight popularity boost (avoid total obscurity)
        count = film.get('rating_count') or 0
        if count > POPULARITY_HIGH_THRESHOLD:
            score += 0.3 * WEIGHTS['popularity']
        elif count > POPULARITY_MED_THRESHOLD:
            score += 0.1 * WEIGHTS['popularity']

        return score, reasons, warnings
    
    def similar_to(self, slug: str, n: int = 10) -> list[Recommendation]:
        """Find films similar to a specific film (item-based)."""
        if slug not in self.films:
            return []

        target = self.films[slug]
        target_genres = set(load_json(target.get('genres')))
        target_directors = set(load_json(target.get('directors')))
        target_cast = set(load_json(target.get('cast', []))[:5])
        target_themes = set(load_json(target.get('themes', [])))
        target_countries = set(load_json(target.get('countries', [])))
        target_writers = set(load_json(target.get('writers', [])))

        target_year = target.get('year')
        target_decade = (target_year // 10) * 10 if isinstance(target_year, int) and target_year > 0 else None
        
        candidates = []
        for other_slug, film in self.films.items():
            if other_slug == slug:
                continue
            
            score = 0
            reasons = []
            
            # Genre overlap
            film_genres = set(load_json(film.get('genres')))
            genre_overlap = target_genres & film_genres
            score += len(genre_overlap) * 1.0
            
            # Same director
            film_directors = set(load_json(film.get('directors')))
            dir_overlap = target_directors & film_directors
            if dir_overlap:
                score += SIMILAR_DIRECTOR_BONUS
                reasons.append(f"Same director: {list(dir_overlap)[0]}")

            # Cast overlap
            film_cast = set(load_json(film.get('cast', []))[:5])
            cast_overlap = target_cast & film_cast
            score += len(cast_overlap) * SIMILAR_CAST_SCORE
            if cast_overlap:
                reasons.append(f"Shared cast: {list(cast_overlap)[0]}")

            # Theme overlap
            film_themes = set(load_json(film.get('themes', [])))
            theme_overlap = target_themes & film_themes
            score += len(theme_overlap) * 0.3

            # Country overlap
            film_countries = set(load_json(film.get('countries', [])))
            if target_countries & film_countries:
                score += 0.5

            # Writer overlap
            film_writers = set(load_json(film.get('writers', [])))
            writer_overlap = target_writers & film_writers
            if writer_overlap:
                score += 3.0
                reasons.append(f"Same writer: {list(writer_overlap)[0]}")

            # Same decade
            film_year = film.get('year')
            film_decade = (film_year // 10) * 10 if isinstance(film_year, int) and film_year > 0 else None

            if target_decade is not None and film_decade == target_decade:
                score += SIMILAR_DECADE_SCORE
            
            if score > 0:
                candidates.append((other_slug, score, reasons))
        
        candidates.sort(key=lambda x: -x[1])
        
        return [
            Recommendation(
                slug=s, 
                title=self.films[s].get('title', s),
                year=self.films[s].get('year'),
                score=sc,
                reasons=r[:2]
            )
            for s, sc, r in candidates[:n]
        ]
    
    def _diversify(self, candidates: list[tuple[str, float, list[str], list[str]]], n: int, max_per_director: int = 2) -> list[Recommendation]:
        """Select top n while limiting per-director concentration."""
        from collections import defaultdict

        results = []
        director_counts = defaultdict(int)

        for slug, score, reasons, warnings in candidates:
            film = self.films.get(slug)
            if not film:
                continue

            directors = load_json(film.get('directors'))

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

        self._build_sparse_matrix()
        self._precompute_similarity_components()

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
        neighbors = self._find_neighbors(username, target_films)
        
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
                
                # Score based on rating or like
                if rating and rating >= 3.5:
                    score = (rating - 2.5) * similarity  # normalize around mid-point
                elif liked:
                    score = 1.0 * similarity
                else:
                    score = 0.1 * similarity  # just watched
                
                if slug not in film_scores:
                    film_scores[slug] = 0
                    film_reasons[slug] = []
                
                film_scores[slug] += score
                
                # Track who recommended it
                if score > 0.5 and len(film_reasons[slug]) < 3:
                    film_reasons[slug].append(f"Liked by {neighbor_user}")
        
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
        similarities = np.array(self._normalized_matrix @ target_row.T).flatten()

        # Overlap counts for confidence weighting
        target_binary = self._overlap_matrix[target_idx]
        overlaps = np.array(self._overlap_matrix @ target_binary.T).flatten()

        # Filter and weight
        min_overlap = 5
        valid = (overlaps >= min_overlap) & (np.arange(n_users) != target_idx)
        confidence = np.minimum(overlaps / 20.0, 1.0)  # Full confidence at 20+ common films

        # Apply confidence weighting and filter invalid
        weighted = np.where(valid, similarities * confidence, -np.inf)

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

