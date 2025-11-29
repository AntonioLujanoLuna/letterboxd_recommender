from dataclasses import dataclass
from .profile import UserProfile, build_profile
from .database import load_json

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
    
    # Weights for different signal types
    WEIGHTS = {
        'genre': 1.0,
        'director': 3.0,      # strong signal
        'actor': 0.5,
        'theme': 0.4,
        'decade': 0.3,
        'community_rating': 0.8,
        'popularity': 0.2,
    }
    
    def __init__(self, all_films: list[dict]):
        self.films = {f['slug']: f for f in all_films}
        self.films_dict = self.films  # Alias for consistency
    
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
    ) -> list[Recommendation]:
        """Generate recommendations."""
        
        # Build user profile
        profile = build_profile(user_films, self.films)
        
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
                if profile.genres[g] > 0.5:
                    matched_genres.append(g)
        score += genre_score * self.WEIGHTS['genre']
        if matched_genres:
            reasons.append(f"Genre: {', '.join(matched_genres[:2])}")
        
        # Director match (strong signal)
        film_directors = load_json(film.get('directors'))
        for d in film_directors:
            if d in profile.directors:
                dir_score = profile.directors[d]
                score += dir_score * self.WEIGHTS['director']
                if dir_score > 1.0:
                    reasons.append(f"Director: {d}")
        
        # Actor match
        film_cast = load_json(film.get('cast', []))[:5]
        matched_actors = []
        for a in film_cast:
            if a in profile.actors and profile.actors[a] > 0.5:
                score += profile.actors[a] * self.WEIGHTS['actor']
                matched_actors.append(a)
        if matched_actors:
            reasons.append(f"Cast: {', '.join(matched_actors[:2])}")
        
        # Theme match
        film_themes = load_json(film.get('themes', []))
        for t in film_themes:
            if t in profile.themes:
                score += profile.themes[t] * self.WEIGHTS['theme']
        
        # Decade match
        year = film.get('year')
        if year:
            decade = (year // 10) * 10
            if decade in profile.decades:
                score += profile.decades[decade] * self.WEIGHTS['decade']
        
        # Community rating bonus
        # Favor films rated similarly to user's liked films
        avg = film.get('avg_rating')
        if avg and profile.avg_liked_rating:
            # Bonus for films near user's sweet spot
            rating_diff = abs(avg - profile.avg_liked_rating)
            if rating_diff < 0.3:
                score += 1.0 * self.WEIGHTS['community_rating']
                reasons.append(f"Highly rated ({avg:.1f}★)")
            elif rating_diff < 0.5:
                score += 0.5 * self.WEIGHTS['community_rating']
        
        # Slight popularity boost (avoid total obscurity)
        count = film.get('rating_count') or 0
        if count > 10000:
            score += 0.3 * self.WEIGHTS['popularity']
        elif count > 1000:
            score += 0.1 * self.WEIGHTS['popularity']
        
        return score, reasons
    
    def similar_to(self, slug: str, n: int = 10) -> list[Recommendation]:
        """Find films similar to a specific film (item-based)."""
        if slug not in self.films:
            return []
        
        target = self.films[slug]
        target_genres = set(load_json(target.get('genres')))
        target_directors = set(load_json(target.get('directors')))
        target_cast = set(load_json(target.get('cast', []))[:5])
        target_decade = (target.get('year') // 10) * 10 if target.get('year') else None
        
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
                score += 5.0
                reasons.append(f"Same director: {list(dir_overlap)[0]}")
            
            # Cast overlap
            film_cast = set(load_json(film.get('cast', []))[:5])
            cast_overlap = target_cast & film_cast
            score += len(cast_overlap) * 0.5
            if cast_overlap:
                reasons.append(f"Shared cast: {list(cast_overlap)[0]}")
            
            # Same decade
            film_decade = (film.get('year') // 10) * 10 if film.get('year') else None
            if target_decade and film_decade == target_decade:
                score += 0.5
            
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
            film = self.films_dict.get(slug)
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
    """
    
    def __init__(self, all_user_films: dict[str, list[dict]], film_metadata: dict[str, dict] | None = None):
        """
        Args:
            all_user_films: Dict mapping username -> list of user_films dicts
            film_metadata: Optional dict mapping slug -> film metadata dict for filtering and display
        """
        self.all_user_films = all_user_films
        self.films = film_metadata or {}
    
    def recommend(
        self,
        username: str,
        n: int = 20,
        min_neighbors: int = 3,
        min_year: int | None = None,
        max_year: int | None = None,
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
                
                # Apply year filters if metadata available
                if self.films and slug in self.films:
                    year = self.films[slug].get('year')
                    if min_year and year and year < min_year:
                        continue
                    if max_year and year and year > max_year:
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
    
    def _find_neighbors(self, username: str, target_films: list[dict], k: int = 10) -> list[tuple[str, float]]:
        """
        Find k most similar users based on rating overlap.
        Uses mean-centered ratings to account for different rating scales.
        Returns list of (username, similarity_score) tuples.
        """
        similarities = []
        
        # Build target user's rating dict and mean
        target_ratings = {}
        for film in target_films:
            rating = film.get('rating')
            if rating:
                target_ratings[film['slug']] = rating
        
        if not target_ratings:
            return []
        
        target_mean = sum(target_ratings.values()) / len(target_ratings)
        
        # Compare with all other users
        for other_user, other_films in self.all_user_films.items():
            if other_user == username:
                continue
            
            # Build other user's rating dict and mean
            other_ratings = {}
            for film in other_films:
                rating = film.get('rating')
                if rating:
                    other_ratings[film['slug']] = rating
            
            if not other_ratings:
                continue
            
            other_mean = sum(other_ratings.values()) / len(other_ratings)
            
            # Find common films
            common = set(target_ratings.keys()) & set(other_ratings.keys())
            
            if len(common) < 5:  # Need at least 5 common films
                continue
            
            # Compute similarity with mean-centered ratings (Pearson-like)
            numerator = sum((target_ratings[s] - target_mean) * (other_ratings[s] - other_mean) for s in common)
            target_variance = sum((target_ratings[s] - target_mean) ** 2 for s in common) ** 0.5
            other_variance = sum((other_ratings[s] - other_mean) ** 2 for s in common) ** 0.5
            
            if target_variance == 0 or other_variance == 0:
                continue
            
            similarity = numerator / (target_variance * other_variance)
            
            # Apply significance weighting - more common films = more reliable
            confidence = min(len(common) / 20, 1.0)  # Full confidence at 20+ common films
            weighted_similarity = similarity * confidence
            
            similarities.append((other_user, weighted_similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: -x[1])
        return similarities[:k]

