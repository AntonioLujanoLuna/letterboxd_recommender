"""Rank fusion for combining recommendation lists."""


RRF_C = 60

# Short histories: popularity beats every personalized model below ~20 films (recall@20 0.10 vs
# 0.06-0.10 at 3-20 films, 200-250 held-out users, two splits), and mixing the two by rank beat both
# at every size from 3 to 40 films. The best popularity share was 0.75 up to 20 films and fell
# linearly to 0 by 50, where EASE alone is clearly ahead (0.130 vs 0.104).
SPARSE_PLATEAU_FILMS = 20
SPARSE_FULL_HISTORY_FILMS = 50
SPARSE_MAX_POPULARITY_WEIGHT = 0.75


def sparse_popularity_weight(n_films: int) -> float:
    """Share of the final ranking to give to popular films for a user with `n_films` in their history."""
    if n_films >= SPARSE_FULL_HISTORY_FILMS:
        return 0.0
    if n_films <= SPARSE_PLATEAU_FILMS:
        return SPARSE_MAX_POPULARITY_WEIGHT
    span = SPARSE_FULL_HISTORY_FILMS - SPARSE_PLATEAU_FILMS
    return SPARSE_MAX_POPULARITY_WEIGHT * (SPARSE_FULL_HISTORY_FILMS - n_films) / span


def rrf_fuse(ranked_lists: list[tuple[list[str], float]], c: int = RRF_C) -> list[tuple[str, float]]:
    """Weighted reciprocal-rank fusion: score(slug) = sum(weight / (c + rank)) over the lists.

    `ranked_lists` is [(slugs best-first, weight), ...]. Rank-based, so models whose raw scores
    live on different scales (EASE vs MovieLens EASE) combine without any normalization.
    Returns (slug, fused score) best-first.
    """
    scores: dict[str, float] = {}
    for slugs, weight in ranked_lists:
        if weight <= 0:
            continue
        for rank, slug in enumerate(slugs):
            scores[slug] = scores.get(slug, 0.0) + weight / (c + rank + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
