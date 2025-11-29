from collections import defaultdict
from dataclasses import dataclass, field
from .database import load_json

@dataclass
class UserProfile:
    """Aggregated preferences extracted from user's film history."""
    
    # Attribute -> score (positive = likes, negative = dislikes)
    genres: dict[str, float] = field(default_factory=dict)
    directors: dict[str, float] = field(default_factory=dict)
    actors: dict[str, float] = field(default_factory=dict)
    themes: dict[str, float] = field(default_factory=dict)
    
    # Decade preferences
    decades: dict[int, float] = field(default_factory=dict)
    
    # Stats
    avg_liked_rating: float = 0.0  # avg Letterboxd rating of films user liked
    n_films: int = 0
    n_rated: int = 0
    has_explicit_feedback: bool = False

def build_profile(user_films: list[dict], film_metadata: dict[str, dict]) -> UserProfile:
    """
    Build preference profile from user's film interactions.
    
    Weighting strategy:
    - Rating 4.5-5.0: +2.0 (loved it)
    - Rating 3.5-4.0: +1.0 (liked it)
    - Rating 3.0:     +0.3 (neutral-positive)
    - Rating 2.0-2.5: -0.5 (disliked)
    - Rating 0.5-1.5: -1.5 (hated)
    - Liked (heart):  +1.5
    - Watched only:   +0.4 (mild positive)
    - Watchlisted:    +0.2 (interest signal)
    """
    profile = UserProfile()
    
    genre_scores = defaultdict(float)
    director_scores = defaultdict(float)
    actor_scores = defaultdict(float)
    theme_scores = defaultdict(float)
    decade_scores = defaultdict(float)
    
    genre_counts = defaultdict(int)
    director_counts = defaultdict(int)
    actor_counts = defaultdict(int)
    theme_counts = defaultdict(int)
    decade_counts = defaultdict(int)
    
    liked_ratings = []
    
    for uf in user_films:
        slug = uf['slug']
        meta = film_metadata.get(slug)
        if not meta:
            continue
        
        # Determine weight for this film
        weight = _compute_weight(uf)
        if weight == 0:
            continue
        
        # Extract attributes
        genres = load_json(meta.get('genres'))
        directors = load_json(meta.get('directors'))
        cast = load_json(meta.get('cast', []))[:5]  # top 5 actors
        themes = load_json(meta.get('themes', []))[:10]
        year = meta.get('year')
        
        # Accumulate scores
        for g in genres:
            genre_scores[g] += weight
            genre_counts[g] += 1
        
        for d in directors:
            director_scores[d] += weight
            director_counts[d] += 1
        
        for a in cast:
            # Diminishing weight for lower-billed actors
            actor_scores[a] += weight * 0.7
            actor_counts[a] += 1
        
        for t in themes:
            theme_scores[t] += weight * 0.5
            theme_counts[t] += 1
        
        if year:
            decade = (year // 10) * 10
            decade_scores[decade] += weight
            decade_counts[decade] += 1
        
        # Track Letterboxd ratings of liked films
        if weight > 0.5 and meta.get('avg_rating'):
            liked_ratings.append(meta['avg_rating'])
    
    # Normalize by count (so prolific directors don't dominate)
    # But keep raw score influence—someone with 10 Spielberg films rated 5★ loves Spielberg
    profile.genres = {k: v / (genre_counts[k] ** 0.5) for k, v in genre_scores.items()}
    profile.directors = {k: v for k, v in director_scores.items()}  # don't normalize directors
    profile.actors = {k: v / (actor_counts[k] ** 0.3) for k, v in actor_scores.items()}
    profile.themes = {k: v / (theme_counts[k] ** 0.5) for k, v in theme_scores.items()}
    profile.decades = {k: v / (decade_counts[k] ** 0.5) for k, v in decade_scores.items()}
    
    # Stats
    profile.n_films = len(user_films)
    profile.n_rated = sum(1 for f in user_films if f.get('rating'))
    profile.has_explicit_feedback = profile.n_rated >= 10
    profile.avg_liked_rating = sum(liked_ratings) / len(liked_ratings) if liked_ratings else 3.5
    
    return profile

def _compute_weight(uf: dict) -> float:
    """Compute preference weight for a single film interaction."""
    rating = uf.get('rating')
    liked = uf.get('liked', False)
    watched = uf.get('watched', False)
    watchlisted = uf.get('watchlisted', False)
    
    # Explicit rating takes precedence
    if rating is not None:
        if rating >= 4.5:
            return 2.0
        elif rating >= 3.5:
            return 1.0
        elif rating >= 3.0:
            return 0.3
        elif rating >= 2.0:
            return -0.5
        else:
            return -1.5
    
    # Liked without rating
    if liked:
        return 1.5
    
    # Just watched
    if watched:
        return 0.4
    
    # Just watchlisted
    if watchlisted:
        return 0.2
    
    return 0
