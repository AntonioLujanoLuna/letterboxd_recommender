"""
Offline evaluation of recommendation strategies.

Protocol (per-user hold-out):
  1. Pick users with at least `min_positives` positive interactions (rated >= 3.5★ or liked).
  2. For each, hide a random `holdout_frac` of those positives. The hidden rows are removed
     entirely from the training data, so no strategy can see them.
  3. Fit each strategy on the training data and ask it for top-k films per user.
  4. Score with recall@k, NDCG@k and hit-rate@k against the hidden films, plus catalog
     coverage (share of the catalog that appears in anyone's top-k).

Every strategy excludes everything left in the user's training history (including
watchlist entries), as it would in production. The "popularity" baseline (most-liked
films the user hasn't seen) is the bar the other strategies need to clear.

There are no watch dates in the database, so the hold-out is random rather than "most
recent". That rewards recovering taste in general, not predicting the next watch.
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable

logger = logging.getLogger(__name__)

POSITIVE_RATING_THRESHOLD = 3.5

# The graph strategy builds from the live database, so held-out films would leak into it.
EVALUABLE_STRATEGIES = ("popularity", "svd", "collaborative", "metadata", "hybrid", "ease", "content")

# A recommender function: (username, training films for that user, k) -> ranked slugs
RecommendFn = Callable[[str, list[dict], int], list[str]]


def is_positive(film: dict) -> bool:
    rating = film.get("rating")
    return (rating is not None and rating >= POSITIVE_RATING_THRESHOLD) or bool(film.get("liked"))


@dataclass
class HoldoutSplit:
    train: dict[str, list[dict]]
    test: dict[str, set[str]]


def make_holdout_split(
    all_user_films: dict[str, list[dict]],
    holdout_frac: float = 0.2,
    min_positives: int = 10,
    max_users: int | None = 200,
    users: Iterable[str] | None = None,
    catalog: set[str] | None = None,
    seed: int = 42,
) -> HoldoutSplit:
    """
    Hide a random share of each evaluated user's positives.

    Args:
        users: Evaluate exactly these users (skips the min_positives/max_users sampling).
        catalog: If given, only films in it can be held out, so every strategy (including
            metadata-only ones) can in principle recommend every hidden film.
    """
    if not 0 < holdout_frac < 1:
        raise ValueError("holdout_frac must be between 0 and 1")
    rng = random.Random(seed)

    def eligible_positives(films: list[dict]) -> list[str]:
        return sorted(
            f["slug"] for f in films
            if is_positive(f) and (catalog is None or f["slug"] in catalog)
        )

    if users is not None:
        candidates = [u for u in users if u in all_user_films]
    else:
        candidates = sorted(
            u for u, films in all_user_films.items()
            if len(eligible_positives(films)) >= min_positives
        )
        if max_users and len(candidates) > max_users:
            candidates = sorted(rng.sample(candidates, max_users))

    train = dict(all_user_films)
    test: dict[str, set[str]] = {}
    for username in candidates:
        films = all_user_films[username]
        positives = eligible_positives(films)
        if len(positives) < 2:  # need something left to learn from
            continue
        n_hold = max(1, round(len(positives) * holdout_frac))
        held = set(rng.sample(positives, n_hold))
        train[username] = [f for f in films if f["slug"] not in held]
        test[username] = held
    return HoldoutSplit(train=train, test=test)


# --- Metrics ---------------------------------------------------------------

def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for slug in recommended[:k] if slug in relevant)
    return hits / min(k, len(relevant))


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, slug in enumerate(recommended[:k]) if slug in relevant)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / idcg


def hit_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(slug in relevant for slug in recommended[:k]) else 0.0


# --- Strategy adapters -----------------------------------------------------
# Each builder fits once on the training split and returns a per-user RecommendFn.

def _build_popularity(train: dict[str, list[dict]], films: dict[str, dict]) -> RecommendFn:
    counts = Counter(f["slug"] for user_films in train.values() for f in user_films if is_positive(f))
    ranked = [slug for slug, _ in counts.most_common()]

    def recommend(username: str, user_films: list[dict], k: int) -> list[str]:
        seen = {f["slug"] for f in user_films}
        out = []
        for slug in ranked:
            if slug not in seen:
                out.append(slug)
                if len(out) >= k:
                    break
        return out

    return recommend


def _build_svd(train: dict[str, list[dict]], films: dict[str, dict]) -> RecommendFn:
    from .matrix_factorization import SVDRecommender

    n_factors = max(1, min(50, len(train) - 1))
    svd = SVDRecommender(n_factors=n_factors).fit(train)

    def recommend(username: str, user_films: list[dict], k: int) -> list[str]:
        seen = {f["slug"] for f in user_films}
        # Over-fetch, then keep films with metadata, as the CLI does
        recs = svd.recommend(username, seen, n=k * 3)
        return [slug for slug, _ in recs if not films or slug in films][:k]

    return recommend


def _build_collaborative(train: dict[str, list[dict]], films: dict[str, dict]) -> RecommendFn:
    from .recommender import CollaborativeRecommender

    collab = CollaborativeRecommender(train, films, use_cache=False)

    def recommend(username: str, user_films: list[dict], k: int) -> list[str]:
        return [r.slug for r in collab.recommend(username, n=k)]

    return recommend


def _build_metadata(train: dict[str, list[dict]], films: dict[str, dict]) -> RecommendFn:
    from .recommender import MetadataRecommender

    meta = MetadataRecommender([dict(f) for f in films.values()])

    def recommend(username: str, user_films: list[dict], k: int) -> list[str]:
        # username=None and user_lists=[]: the profile cache and the user's lists in the DB
        # were built from the full history, so they would leak held-out films.
        recs = meta.recommend(user_films, n=k, username=None, user_lists=[])
        return [r.slug for r in recs]

    return recommend


def _build_hybrid(train: dict[str, list[dict]], films: dict[str, dict]) -> RecommendFn:
    from .recommender import (
        CollaborativeRecommender,
        MetadataRecommender,
        _fuse_normalized,
        hybrid_weights,
    )

    meta = MetadataRecommender([dict(f) for f in films.values()])
    collab = CollaborativeRecommender(train, films, use_cache=False)

    def recommend(username: str, user_films: list[dict], k: int) -> list[str]:
        meta_recs = meta.recommend(user_films, n=k * 2, username=None, user_lists=[])
        collab_recs = collab.recommend(username, n=k * 2)
        rated = sum(1 for f in user_films if f.get("rating"))
        w_meta, w_collab = hybrid_weights(len(train), rated)
        fused = _fuse_normalized(meta_recs, collab_recs, weight_meta=w_meta, weight_collab=w_collab)
        return [slug for slug, _, _ in fused if not films or slug in films][:k]

    return recommend


def _build_ease(train: dict[str, list[dict]], films: dict[str, dict]) -> RecommendFn:
    from .ease import EASERecommender

    model = EASERecommender().fit(train)

    def recommend(username: str, user_films: list[dict], k: int) -> list[str]:
        recs = model.recommend(user_films, n=k * 3)
        return [slug for slug, _ in recs if not films or slug in films][:k]

    return recommend


def _build_content(train: dict[str, list[dict]], films: dict[str, dict]) -> RecommendFn:
    from .content import ContentRecommender

    model = ContentRecommender(films)

    def recommend(username: str, user_films: list[dict], k: int) -> list[str]:
        return [slug for slug, _ in model.recommend(user_films, n=k)]

    return recommend


def _build_blend(train: dict[str, list[dict]], films: dict[str, dict]) -> RecommendFn:
    """Local EASE fused with the pre-trained MovieLens EASE (reciprocal-rank fusion)."""
    from .config import BLEND_EASE_WEIGHT
    from .recommenders.fusion import rrf_fuse

    local = _build_ease(train, films)
    pretrained = _build_movielens(train, films)

    def recommend(username: str, user_films: list[dict], k: int) -> list[str]:
        depth = k * 3
        fused = rrf_fuse([(local(username, user_films, depth), BLEND_EASE_WEIGHT),
                          (pretrained(username, user_films, depth), 1.0 - BLEND_EASE_WEIGHT)])
        return [slug for slug, _ in fused][:k]

    return recommend


def _build_movielens(train: dict[str, list[dict]], films: dict[str, dict]) -> RecommendFn:
    """Pre-trained MovieLens EASE: nothing is fitted on the split, so nothing can leak."""
    from .movielens import FilmMatcher, MovieLensModel

    model = MovieLensModel.load()
    matcher = FilmMatcher(model)
    col_to_slug = matcher.letterboxd_slugs(films)

    def recommend(username: str, user_films: list[dict], k: int) -> list[str]:
        weighted, seen, _ = matcher.map_history(user_films, films)
        train_slugs = {f["slug"] for f in user_films}
        out = []
        for col, _ in model.recommend_columns(weighted, seen, n=k * 5):
            slug = col_to_slug.get(col)
            if slug and slug not in train_slugs:
                out.append(slug)
                if len(out) >= k:
                    break
        return out

    return recommend


STRATEGY_BUILDERS: dict[str, Callable[[dict, dict], RecommendFn]] = {
    "popularity": _build_popularity,
    "svd": _build_svd,
    "collaborative": _build_collaborative,
    "metadata": _build_metadata,
    "hybrid": _build_hybrid,
    "ease": _build_ease,
    "content": _build_content,
    "movielens": _build_movielens,
    "blend": _build_blend,
}


# --- Runner ----------------------------------------------------------------

@dataclass
class StrategyResult:
    strategy: str
    k: int
    n_users: int = 0
    recall: float = 0.0
    ndcg: float = 0.0
    hit_rate: float = 0.0
    coverage: float = 0.0
    fit_seconds: float = 0.0
    recommend_seconds: float = 0.0
    error: str | None = None
    n_failed_users: int = 0          # users whose recommend call raised (scored as empty)
    first_failure: str | None = None
    per_user: dict[str, dict[str, float]] = field(default_factory=dict, repr=False)

    def as_dict(self, include_per_user: bool = False) -> dict:
        d = {
            "strategy": self.strategy,
            "k": self.k,
            "n_users": self.n_users,
            f"recall@{self.k}": round(self.recall, 4),
            f"ndcg@{self.k}": round(self.ndcg, 4),
            f"hit@{self.k}": round(self.hit_rate, 4),
            "coverage": round(self.coverage, 4),
            "fit_seconds": round(self.fit_seconds, 2),
            "recommend_seconds": round(self.recommend_seconds, 2),
        }
        if self.error:
            d["error"] = self.error
        if self.n_failed_users:
            d["n_failed_users"] = self.n_failed_users
            d["first_failure"] = self.first_failure
        if include_per_user:
            d["per_user"] = self.per_user
        return d


def evaluate_strategy(
    name: str,
    split: HoldoutSplit,
    films: dict[str, dict],
    k: int = 20,
    builder: Callable[[dict, dict], RecommendFn] | None = None,
    progress: bool = False,
) -> StrategyResult:
    result = StrategyResult(strategy=name, k=k)
    builder = builder or STRATEGY_BUILDERS[name]

    start = time.perf_counter()
    try:
        recommend = builder(split.train, films)
    except Exception as exc:  # one broken strategy shouldn't sink the whole report
        logger.warning("Strategy %s failed to fit: %s", name, exc)
        result.error = f"fit failed: {exc}"
        return result
    result.fit_seconds = time.perf_counter() - start

    users = sorted(split.test)
    iterator = users
    if progress:
        from tqdm import tqdm
        iterator = tqdm(users, desc=name, unit="user", leave=False)

    recommended_anywhere: set[str] = set()
    start = time.perf_counter()
    for username in iterator:
        relevant = split.test[username]
        try:
            recs = recommend(username, split.train[username], k)
        except Exception as exc:
            # Scored as an empty list, but counted and surfaced: a strategy that crashes
            # for every user must not look like a strategy that is merely bad.
            if not result.n_failed_users:
                result.first_failure = f"{username}: {type(exc).__name__}: {exc}"
                logger.warning("Strategy %s failed for %s: %s", name, username, exc)
            result.n_failed_users += 1
            recs = []
        recs = list(dict.fromkeys(recs))[:k]
        recommended_anywhere.update(recs)
        result.per_user[username] = {
            "recall": recall_at_k(recs, relevant, k),
            "ndcg": ndcg_at_k(recs, relevant, k),
            "hit": hit_at_k(recs, relevant, k),
        }
    result.recommend_seconds = time.perf_counter() - start

    result.n_users = len(users)
    if users:
        result.recall = sum(m["recall"] for m in result.per_user.values()) / len(users)
        result.ndcg = sum(m["ndcg"] for m in result.per_user.values()) / len(users)
        result.hit_rate = sum(m["hit"] for m in result.per_user.values()) / len(users)
    catalog_size = len(films) or len({f["slug"] for fs in split.train.values() for f in fs})
    result.coverage = len(recommended_anywhere) / catalog_size if catalog_size else 0.0
    return result


def run_evaluation(
    all_user_films: dict[str, list[dict]],
    films: dict[str, dict],
    strategies: Iterable[str] = EVALUABLE_STRATEGIES,
    k: int = 20,
    holdout_frac: float = 0.2,
    min_positives: int = 10,
    max_users: int | None = 200,
    users: Iterable[str] | None = None,
    seed: int = 42,
    progress: bool = False,
) -> tuple[HoldoutSplit, list[StrategyResult]]:
    strategies = list(strategies)
    unknown = [s for s in strategies if s not in STRATEGY_BUILDERS]
    if unknown:
        raise ValueError(
            f"Unknown or non-evaluable strategies: {', '.join(unknown)} "
            f"(choose from {', '.join(STRATEGY_BUILDERS)})"
        )

    split = make_holdout_split(
        all_user_films,
        holdout_frac=holdout_frac,
        min_positives=min_positives,
        max_users=max_users,
        users=users,
        catalog=set(films) if films else None,
        seed=seed,
    )
    results = [
        evaluate_strategy(name, split, films, k=k, progress=progress)
        for name in strategies
    ]
    return split, results


def format_results_table(results: list[StrategyResult], baseline: str = "popularity") -> str:
    """Plain-text table; NDCG lift is relative to the baseline when it was evaluated."""
    if not results:
        return "(no results)"
    k = results[0].k
    base = next((r for r in results if r.strategy == baseline and not r.error), None)
    header = (
        f"{'strategy':<14}{'users':>7}{f'recall@{k}':>11}{f'ndcg@{k}':>10}"
        f"{f'hit@{k}':>8}{'coverage':>10}{'vs pop':>9}{'fit s':>8}{'rec s':>8}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        if r.error:
            lines.append(f"{r.strategy:<14}  {r.error}")
            continue
        lift = ""
        if base and r is not base and base.ndcg > 0:
            lift = f"{(r.ndcg / base.ndcg - 1) * 100:+.0f}%"
        lines.append(
            f"{r.strategy:<14}{r.n_users:>7}{r.recall:>11.4f}{r.ndcg:>10.4f}"
            f"{r.hit_rate:>8.3f}{r.coverage:>10.3f}{lift:>9}{r.fit_seconds:>8.1f}{r.recommend_seconds:>8.1f}"
        )
    for r in results:
        if r.n_failed_users:
            lines.append(
                f"WARNING: {r.strategy} raised for {r.n_failed_users}/{r.n_users} users "
                f"(scored as empty). First: {r.first_failure}"
            )
    return "\n".join(lines)
