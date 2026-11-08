"""Rank fusion for combining recommendation lists."""


RRF_C = 60


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
