import logging
from dataclasses import dataclass
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
)

logger = logging.getLogger(__name__)

@dataclass
class Recommendation:
    slug: str
    title: str
    year: int | None
    score: float
    reasons: list[str]

class MetadataRecommender:
    """
    Score films by metadata match to user profile.
    No embeddings—just weighted feature matching.
    """

    COUNTRY_SECONDARY_WEIGHT = 0.3

    def __init__(self, all_films: list[dict]):
        self.films = {f['slug']: f for f in all_films}
    
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
            if genres:
                if not any(g.lower() in [fg.lower() for fg in film_genres] for g in genres):
                    continue
            if exclude_genres:
                if any(g.lower() in [fg.lower() for fg in film_genres] for g in exclude_genres):
                    continue
            
            if min_rating and film.get('avg_rating') and film['avg_rating'] < min_rating:
                continue
            
            # Score the film
            score, reasons = self._score_film(film, profile)
            
            if score > 0:
                candidates.append((slug, score, reasons))
        
        # Sort by score
        candidates.sort(key=lambda x: -x[1])
        
        # Apply diversity if requested
        if diversity:
            return self._diversify(candidates, n, max_per_director)
        
        # Build results (standard mode)
        results = []
        for slug, score, reasons in candidates[:n]:
            film = self.films[slug]
            results.append(Recommendation(
                slug=slug,
                title=film.get('title', slug),
                year=film.get('year'),
                score=score,
                reasons=reasons[:3]  # top 3 reasons
            ))
        
        return results
    
    def recommend_from_candidates(
        self,
        user_films: list[dict],
        candidates: list[str],
        n: int = 20,
    ) -> list[Recommendation]:
        """Score and rank a specific list of films (e.g. watchlist)."""
        # Build user profile
        profile = build_profile(user_films, self.films)
        
        scored_candidates = []
        for slug in candidates:
            if slug not in self.films:
                continue
            
            film = self.films[slug]
            score, reasons = self._score_film(film, profile)
            
            if score > 0:
                scored_candidates.append((slug, score, reasons))
        
        # Sort by score
        scored_candidates.sort(key=lambda x: -x[1])
        
        results = []
        for slug, score, reasons in scored_candidates[:n]:
            film = self.films[slug]
            results.append(Recommendation(
                slug=slug,
                title=film.get('title', slug),
                year=film.get('year'),
                score=score,
                reasons=reasons[:3]
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
    
    def _score_film(self, film: dict, profile: UserProfile) -> tuple[float, list[str]]:
        """
        Score a film against user profile.
        Returns (score, list of reasons).
        """
        score = 0.0
        reasons = []
        
        # Genre match
        film_genres = load_json(film.get('genres'))
        genre_score = 0
        matched_genres = []
        for g in film_genres:
            if g in profile.genres:
                genre_score += profile.genres[g]
                if profile.genres[g] > MATCH_THRESHOLD_GENRE:
                    matched_genres.append(g)
        score += genre_score * WEIGHTS['genre']
        if matched_genres:
            reasons.append(f"Genre: {', '.join(matched_genres[:2])}")

        # Director match (strong signal)
        film_directors = load_json(film.get('directors'))
        for d in film_directors:
            if d in profile.directors:
                dir_score = profile.directors[d]
                score += dir_score * WEIGHTS['director']
                if dir_score > MATCH_THRESHOLD_DIRECTOR:
                    reasons.append(f"Director: {d}")

        # Actor match
        film_cast = load_json(film.get('cast', []))[:5]
        matched_actors = []
        for a in film_cast:
            if a in profile.actors and profile.actors[a] > MATCH_THRESHOLD_ACTOR:
                score += profile.actors[a] * WEIGHTS['actor']
                matched_actors.append(a)
        if matched_actors:
            reasons.append(f"Cast: {', '.join(matched_actors[:2])}")

        # Theme match
        film_themes = load_json(film.get('themes', []))
        for t in film_themes:
            if t in profile.themes:
                score += profile.themes[t] * WEIGHTS['theme']

        # Decade match
        year = film.get('year')
        if year:
            decade = (year // 10) * 10
            if decade in profile.decades:
                score += profile.decades[decade] * WEIGHTS['decade']

        # Phase 1: Country match
        film_countries = load_json(film.get('countries', []))
        for i, country in enumerate(film_countries):
            if country in profile.countries:
                # Primary country gets full weight, secondary reduced
                country_weight = WEIGHTS['country'] if i == 0 else WEIGHTS['country'] * self.COUNTRY_SECONDARY_WEIGHT
                score += profile.countries[country] * country_weight
                if i == 0 and profile.countries[country] > 0.5:
                    reasons.append(f"Country: {country}")

        # Phase 1: Language match
        film_languages = load_json(film.get('languages', []))
        matched_languages = []
        for lang in film_languages:
            if lang in profile.languages and profile.languages[lang] > MATCH_THRESHOLD_LANGUAGE:
                score += profile.languages[lang] * WEIGHTS['language']
                matched_languages.append(lang)
        if matched_languages:
            reasons.append(f"Language: {matched_languages[0]}")

        # Phase 1: Writer match
        film_writers = load_json(film.get('writers', []))
        for w in film_writers:
            if w in profile.writers:
                writer_score = profile.writers[w]
                score += writer_score * WEIGHTS['writer']
                if writer_score > MATCH_THRESHOLD_WRITER:
                    reasons.append(f"Writer: {w}")

        # Phase 1: Cinematographer match
        film_cinematographers = load_json(film.get('cinematographers', []))
        for c in film_cinematographers:
            if c in profile.cinematographers:
                cine_score = profile.cinematographers[c]
                score += cine_score * WEIGHTS['cinematographer']
                if cine_score > MATCH_THRESHOLD_CINE:
                    reasons.append(f"Cinematography: {c}")

        # Phase 1: Composer match
        film_composers = load_json(film.get('composers', []))
        for comp in film_composers:
            if comp in profile.composers:
                comp_score = profile.composers[comp]
                score += comp_score * WEIGHTS['composer']
                if comp_score > MATCH_THRESHOLD_COMPOSER:
                    reasons.append(f"Composer: {comp}")

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
        
        return score, reasons
    
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
    
    def _diversify(self, candidates: list[tuple[str, float, list[str]]], n: int, max_per_director: int = 2) -> list[Recommendation]:
        """Select top n while limiting per-director concentration."""
        from collections import defaultdict
        
        results = []
        director_counts = defaultdict(int)
        
        for slug, score, reasons in candidates:
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
                reasons=reasons[:3]
            ))
            
            # Update director counts
            for d in directors:
                director_counts[d] += 1
            
            if len(results) >= n:
                break
        
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
        self._build_sparse_matrix()

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
            print(f"Warning: Only found {len(neighbors)} neighbors (min: {min_neighbors})")
        
        # Get films seen by target
        seen = {f['slug'] for f in target_films}
        
        # Score unseen films based on neighbor ratings
        film_scores = {}
        film_reasons = {}
        
        for neighbor_user, similarity in neighbors:
            neighbor_films = self.all_user_films[neighbor_user]
            
            for film in neighbor_films:
                slug = film['slug']
                if slug in seen:
                    continue
                
                # Apply filters if metadata available
                if self.films and slug in self.films:
                    film = self.films[slug]
                    year = film.get('year')
                    if min_year and year and year < min_year:
                        continue
                    if max_year and year and year > max_year:
                        continue

                    # Apply genre filters
                    film_genres = load_json(film.get('genres'))
                    if genres:
                        if not any(g.lower() in [fg.lower() for fg in film_genres] for g in genres):
                            continue
                    if exclude_genres:
                        if any(g.lower() in [fg.lower() for fg in film_genres] for g in exclude_genres):
                            continue
                
                rating = film.get('rating')
                liked = film.get('liked', False)
                
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
    
    def _find_neighbors(self, username: str, target_films: list[dict], k: int = 10, max_users_to_compare: int = 10000) -> list[tuple[str, float]]:
        """
        Find k most similar users based on rating overlap using vectorized sparse matrix operations.
        Uses mean-centered ratings to account for different rating scales (Pearson correlation).
        Returns list of (username, similarity_score) tuples.

        Args:
            username: Target username
            target_films: Target user's film interactions
            k: Number of neighbors to return
            max_users_to_compare: Maximum number of users to compare (for large datasets) - ignored when using sparse matrix
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        if username not in self._user_index or self._user_matrix is None:
            return []

        target_idx = self._user_index[username]

        # Get target user's ratings as sparse row
        target_row = self._user_matrix[target_idx]

        # Check if target has any ratings
        target_nonzero = target_row.nonzero()[1]
        if len(target_nonzero) == 0:
            return []

        # Mean-center all users' ratings for Pearson correlation
        # Convert to dense for efficient row-wise operations (still faster than looping)
        all_ratings = self._user_matrix.toarray()

        # Compute masks and means for all users at once
        masks = all_ratings > 0
        row_means = np.sum(all_ratings * masks, axis=1) / np.maximum(masks.sum(axis=1), 1)

        # Mean-center all ratings
        centered_ratings = all_ratings.copy()
        for i in range(len(centered_ratings)):
            centered_ratings[i, masks[i]] -= row_means[i]

        # Compute number of common rated films between target and all other users
        target_mask = masks[target_idx]
        common_counts = (masks & target_mask).sum(axis=1)

        # Filter users with at least 5 common films
        valid_users_mask = (common_counts >= 5) & (np.arange(len(all_ratings)) != target_idx)

        if not valid_users_mask.any():
            return []

        # Use cosine similarity on centered ratings (equivalent to Pearson correlation)
        # Only compute for valid users
        target_centered = centered_ratings[target_idx:target_idx+1]  # Keep 2D shape
        valid_centered = centered_ratings[valid_users_mask]

        # Cosine similarity on mean-centered ratings = Pearson correlation
        similarities = cosine_similarity(target_centered, valid_centered).flatten()

        # Apply significance weighting - more common films = more reliable
        valid_common_counts = common_counts[valid_users_mask]
        confidence = np.minimum(valid_common_counts / 20.0, 1.0)  # Full confidence at 20+ common films
        weighted_similarities = similarities * confidence

        # Build result list with usernames
        usernames = list(self._user_index.keys())
        valid_usernames = [usernames[i] for i in np.where(valid_users_mask)[0]]

        # Combine usernames and scores, sort by score
        results = list(zip(valid_usernames, weighted_similarities))
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:k]

