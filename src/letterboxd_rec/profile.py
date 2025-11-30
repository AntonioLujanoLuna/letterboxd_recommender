from collections import defaultdict
from dataclasses import dataclass, field
from .database import load_json

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
    # Phase 1 enhancements
    countries: dict[str, float] = field(default_factory=dict)
    languages: dict[str, float] = field(default_factory=dict)
    writers: dict[str, float] = field(default_factory=dict)
    cinematographers: dict[str, float] = field(default_factory=dict)
    composers: dict[str, float] = field(default_factory=dict)


def build_profile(
    user_films: list[dict], 
    film_metadata: dict[str, dict],
    user_lists: list[dict] | None = None
) -> UserProfile:
    """
    Build preference profile from user's film interactions and lists.
    
    Weighting strategy:
    - Rating 4.5-5.0: +2.0 (loved it)
    - Rating 3.5-4.0: +1.0 (liked it)
    - Rating 3.0:     +0.3 (neutral-positive)
    - Rating 2.0-2.5: -0.5 (disliked)
    - Rating 0.5-1.5: -1.5 (hated)
    - Liked (heart):  +1.5
    - Watched only:   +0.4 (mild positive)
    - Watchlisted:    +0.2 (interest signal)
    
    List multipliers (Phase 2.1):
    - Favorites:      3.0x (strongest signal)
    - Ranked top 10:  2.0x
    - Ranked 11-30:   1.5x
    - Ranked 31+:     1.2x
    - Curated list:   1.3x
    """
    profile = UserProfile()
    
    # Phase 2.1: Build list weight lookup
    list_weights = {}
    if user_lists:
        for entry in user_lists:
            slug = entry['film_slug']
            
            # Determine weight multiplier based on list type
            if entry.get('is_favorites'):
                multiplier = 3.0  # Triple weight for favorites
            elif entry.get('is_ranked') and entry.get('position'):
                # Position-based weight for ranked lists
                position = entry['position']
                if position <= 10:
                    multiplier = 2.0  # Top 10
                elif position <= 30:
                    multiplier = 1.5  # Top 30
                else:
                    multiplier = 1.2  # Lower ranked
            else:
                multiplier = 1.3  # Curated list presence
            
            # Keep highest multiplier if film is in multiple lists
            if slug not in list_weights or multiplier > list_weights[slug]:
                list_weights[slug] = multiplier
    
    genre_scores = defaultdict(float)
    director_scores = defaultdict(float)
    actor_scores = defaultdict(float)
    theme_scores = defaultdict(float)
    decade_scores = defaultdict(float)
    country_scores = defaultdict(float)
    language_scores = defaultdict(float)
    writer_scores = defaultdict(float)
    cinematographer_scores = defaultdict(float)
    composer_scores = defaultdict(float)
    
    genre_counts = defaultdict(int)
    director_counts = defaultdict(int)
    actor_counts = defaultdict(int)
    theme_counts = defaultdict(int)
    decade_counts = defaultdict(int)
    country_counts = defaultdict(int)
    language_counts = defaultdict(int)
    writer_counts = defaultdict(int)
    cinematographer_counts = defaultdict(int)
    composer_counts = defaultdict(int)
    
    rated_films = []
    
    for uf in user_films:
        slug = uf['slug']
        meta = film_metadata.get(slug)
        if not meta:
            continue
        
        # Determine weight for this film
        weight = _compute_weight(uf)
        
        # Phase 2.1: Apply list multiplier if film is in any lists
        if slug in list_weights:
            weight *= list_weights[slug]
        
        if weight == 0:
            continue
        
        # Extract attributes
        genres = load_json(meta.get('genres'))
        directors = load_json(meta.get('directors'))
        cast = load_json(meta.get('cast', []))[:5]
        themes = load_json(meta.get('themes', []))[:10]
        year = meta.get('year')
        countries = load_json(meta.get('countries', []))
        languages = load_json(meta.get('languages', []))
        writers = load_json(meta.get('writers', []))
        cinematographers = load_json(meta.get('cinematographers', []))
        composers = load_json(meta.get('composers', []))
        
        # Accumulate scores
        for g in genres:
            genre_scores[g] += weight
            genre_counts[g] += 1
        
        for d in directors:
            director_scores[d] += weight
            director_counts[d] += 1
        
        for a in cast:
            actor_scores[a] += weight * 0.7
            actor_counts[a] += 1
        
        for t in themes:
            theme_scores[t] += weight * 0.5
            theme_counts[t] += 1
        
        if year:
            decade = (year // 10) * 10
            decade_scores[decade] += weight
            decade_counts[decade] += 1
        
        # Phase 1: New fields
        for i, country in enumerate(countries):
            # Primary country gets full weight, secondary get reduced
            country_weight = weight if i == 0 else weight * 0.3
            country_scores[country] += country_weight
            country_counts[country] += 1
        
        for lang in languages:
            language_scores[lang] += weight
            language_counts[lang] += 1
        
        for w in writers:
            writer_scores[w] += weight
            writer_counts[w] += 1
        
        for c in cinematographers:
            cinematographer_scores[c] += weight
            cinematographer_counts[c] += 1
        
        for comp in composers:
            composer_scores[comp] += weight
            composer_counts[comp] += 1
        
        # Track rated films for average
        if uf.get('rating'):
            rated_films.append(uf['rating'])
    
    # Normalize scores
    profile.genres = {k: v / (genre_counts[k] ** 0.5) for k, v in genre_scores.items()}
    profile.directors = {k: v for k, v in director_scores.items()}
    profile.actors = {k: v / (actor_counts[k] ** 0.3) for k, v in actor_scores.items()}
    profile.themes = {k: v / (theme_counts[k] ** 0.5) for k, v in theme_scores.items()}
    profile.decades = {k: v / (decade_counts[k] ** 0.5) for k, v in decade_scores.items()}
    profile.countries = {k: v / (country_counts[k] ** 0.5) for k, v in country_scores.items()}
    profile.languages = {k: v / (language_counts[k] ** 0.5) for k, v in language_scores.items()}
    profile.writers = {k: v for k, v in writer_scores.items()}
    profile.cinematographers = {k: v for k, v in cinematographer_scores.items()}
    profile.composers = {k: v for k, v in composer_scores.items()}
    
    # Aggregate counts
    profile.n_films = len(user_films)
    profile.n_rated = len(rated_films)
    profile.n_liked = sum(1 for f in user_films if f.get('liked'))
    profile.avg_liked_rating = sum(rated_films) / len(rated_films) if rated_films else None
    
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
            base_weight = 2.0
        elif rating >= 3.5:
            base_weight = 1.0
        elif rating >= 3.0:
            base_weight = 0.3
        elif rating >= 2.0:
            base_weight = -0.5
        else:
            base_weight = -1.5
    # Liked without rating
    elif liked:
        base_weight = 1.5
    # Just watched
    elif watched:
        base_weight = 0.4
    # Just watchlisted
    elif watchlisted:
        base_weight = 0.2
    else:
        base_weight = 0.0
    
    # Phase 1: Apply temporal decay
    if 'scraped_at' in uf and uf['scraped_at']:
        try:
            from datetime import datetime
            scraped = datetime.fromisoformat(uf['scraped_at'])
            days_ago = (datetime.now() - scraped).days
            decay = 0.95 ** (days_ago / 365)  # 5% decay per year
            base_weight *= max(decay, 0.5)  # floor at 50%
        except (ValueError, TypeError):
            pass  # If parsing fails, use base weight
    
    return base_weight
