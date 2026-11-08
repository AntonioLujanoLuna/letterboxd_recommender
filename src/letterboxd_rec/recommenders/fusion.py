"""Score fusion utilities for hybrid recommendations."""

from .recommendation import Recommendation


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


def hybrid_weights(
    n_users: int,
    rated_count: int,
    meta_weight: float | None = None,
    collab_weight: float | None = None,
) -> tuple[float, float]:
    """
    Resolve (metadata, collaborative) weights for the hybrid strategy.

    With no overrides, lean on metadata when there are few users or the target user
    has few ratings. With one override, derive the other as its complement.
    """
    if meta_weight is None and collab_weight is None:
        if n_users < 10 or rated_count < 20:
            return 0.7, 0.3
        return 0.6, 0.4
    if meta_weight is None:
        meta_weight = max(0.0, 1.0 - collab_weight)
    if collab_weight is None:
        collab_weight = max(0.0, 1.0 - meta_weight)
    return meta_weight, collab_weight


def _fuse_normalized(
    meta_results: list[Recommendation],
    collab_results: list[Recommendation],
    weight_meta: float = 0.6,
    weight_collab: float = 0.4,
) -> list[tuple[str, float, list[str]]]:
    """
    Score-based fusion with min-max normalization.

    Preserves relative score differences within each strategy.
    """

    def normalize_scores(recs: list[Recommendation]) -> dict[str, float]:
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
