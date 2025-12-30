"""Score fusion utilities for hybrid recommendations."""

from .recommendation import Recommendation


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
