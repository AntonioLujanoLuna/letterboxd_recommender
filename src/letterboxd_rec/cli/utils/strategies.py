"""Recommendation strategy runners for CLI commands."""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

from ...database import load_json, load_user_lists
from ...recommender import (
    MetadataRecommender,
    CollaborativeRecommender,
    Recommendation,
    _fuse_normalized,
)
from ...matrix_factorization import SVDRecommender
from ...profile import build_profile
from ...graph_config import GraphConfig
from ...graph_recommender import GraphRecommender
from ...config import SVD_CACHE_PATH

logger = logging.getLogger(__name__)


def _run_metadata_strategy(
    user_films: list[dict],
    all_films: dict[str, dict],
    args: argparse.Namespace,
    username: str,
    user_lists: list[dict] | None,
    all_user_films: dict[str, list[dict]] | None = None,
) -> list[Recommendation]:
    """Metadata-based recommendation strategy."""
    diversity = getattr(args, 'diversity', False)
    max_per_director = getattr(args, 'max_per_director', 2)
    use_temporal_decay = not getattr(args, 'no_temporal_decay', False)
    lists = user_lists if user_lists is not None else load_user_lists(username)

    profile = build_profile(
        user_films,
        all_films,
        user_lists=lists,
        username=username,
        use_temporal_decay=use_temporal_decay,
        weighting_mode=args.weighting_mode,
    )

    recommender = MetadataRecommender(list(all_films.values()))
    return recommender.recommend(
        user_films,
        n=args.limit,
        min_year=args.min_year,
        max_year=args.max_year,
        genres=args.genres,
        exclude_genres=args.exclude_genres,
        min_rating=args.min_rating,
        diversity=diversity,
        max_per_director=max_per_director,
        username=username,
        user_lists=lists,
        profile=profile,
        weighting_mode=args.weighting_mode,
    )


def _run_collaborative_strategy(
    user_films: list[dict],
    all_films: dict[str, dict],
    args: argparse.Namespace,
    username: str,
    user_lists: list[dict] | None,
    all_user_films: dict[str, list[dict]] | None,
) -> list[Recommendation]:
    """Collaborative filtering strategy."""
    if not all_user_films:
        logger.error("Collaborative strategy requires all user data.")
        return []

    recommender = CollaborativeRecommender(all_user_films, all_films)
    return recommender.recommend(
        username,
        n=args.limit,
        min_year=args.min_year,
        max_year=args.max_year,
        genres=args.genres,
        exclude_genres=args.exclude_genres
    )


def _run_hybrid_strategy(
    user_films: list[dict],
    all_films: dict[str, dict],
    args: argparse.Namespace,
    username: str,
    user_lists: list[dict] | None,
    all_user_films: dict[str, list[dict]] | None,
) -> list[Recommendation]:
    """Hybrid metadata + collaborative strategy."""
    if not all_user_films:
        logger.error("Hybrid strategy requires all user data.")
        return []

    meta_rec = MetadataRecommender(list(all_films.values()))
    collab_rec = CollaborativeRecommender(all_user_films, all_films)

    rated_count = sum(1 for f in user_films if f.get('rating'))
    meta_weight = args.hybrid_meta_weight
    collab_weight = args.hybrid_collab_weight

    if meta_weight is None and collab_weight is None:
        # Heuristic: lean on metadata when user data or neighbors are sparse
        if len(all_user_films) < 10 or rated_count < 20:
            meta_weight = 0.7
            collab_weight = 0.3
        else:
            meta_weight = 0.6
            collab_weight = 0.4
    else:
        # If only one is provided, derive the other to preserve user intent
        if meta_weight is None:
            meta_weight = max(0.0, 1.0 - collab_weight) if collab_weight is not None else 0.6
        if collab_weight is None:
            collab_weight = max(0.0, 1.0 - meta_weight) if meta_weight is not None else 0.4

    meta_recs = meta_rec.recommend(
        user_films,
        n=args.limit * 2,
        min_year=args.min_year,
        max_year=args.max_year,
        genres=args.genres,
        exclude_genres=args.exclude_genres,
        min_rating=args.min_rating,
        username=username,
        weighting_mode=args.weighting_mode,
    )
    collab_recs = collab_rec.recommend(
        username,
        n=args.limit * 2,
        min_year=args.min_year,
        max_year=args.max_year,
        genres=args.genres,
        exclude_genres=args.exclude_genres
    )

    ranked = _fuse_normalized(meta_recs, collab_recs, weight_meta=meta_weight, weight_collab=collab_weight)

    max_per_director = getattr(args, 'max_per_director', 2)
    apply_diversity = bool(getattr(args, 'hybrid_diversity', False))
    director_counts = defaultdict(int)

    recs: list[Recommendation] = []
    for slug, score, reasons in ranked:
        film = all_films.get(slug)
        if not film:
            continue

        directors = load_json(film.get('directors'))
        if apply_diversity and max_per_director and directors:
            if any(director_counts[d] >= max_per_director for d in directors):
                continue

        recs.append(Recommendation(
            slug=slug,
            title=film.get('title', slug),
            year=film.get('year'),
            score=score,
            reasons=list(dict.fromkeys(reasons))[:3]
        ))

        for d in directors:
            director_counts[d] += 1

        if len(recs) >= args.limit:
            break

    return recs


def _run_graph_strategy(
    user_films: list[dict],
    all_films: dict[str, dict],
    args: argparse.Namespace,
    username: str,
    user_lists: list[dict] | None,
    all_user_films: dict[str, list[dict]] | None = None,
) -> list[Recommendation]:
    """Graph-based recommendation strategy."""
    config = GraphConfig(
        alpha=getattr(args, "graph_alpha", GraphConfig().alpha),
        relation_preset=getattr(args, "relation_preset", None),
        relation_weights_path=getattr(args, "relation_weights_path", None),
        use_idf=not getattr(args, "no_graph_idf", False),
    )
    overrides = {}
    if getattr(args, "like_weight", None) is not None:
        overrides["liked"] = args.like_weight
    if getattr(args, "watch_weight", None) is not None:
        overrides["watched"] = args.watch_weight
    if getattr(args, "watchlist_weight", None) is not None:
        overrides["watchlisted"] = args.watchlist_weight
    if overrides:
        config.restart_weights = {**config.restart_weights, **overrides}
    if getattr(args, "graph_cache", None):
        config.cache_path = Path(args.graph_cache)
    if getattr(args, "graph_idf_floor", None) is not None:
        config.idf_floor = args.graph_idf_floor
    if getattr(args, "graph_idf_ceiling", None) is not None:
        config.idf_ceiling = args.graph_idf_ceiling

    graph_rec = GraphRecommender(config=config, rebuild=getattr(args, "rebuild_graph", False))
    return graph_rec.recommend(
        username,
        n=args.limit,
        min_year=args.min_year,
        max_year=args.max_year,
        genres=args.genres,
        exclude_genres=args.exclude_genres,
        min_rating=args.min_rating,
        explain=getattr(args, "explain", False),
    )


def _run_svd_strategy(
    user_films: list[dict],
    all_films: dict[str, dict],
    args: argparse.Namespace,
    username: str,
    user_lists: list[dict] | None,
    all_user_films: dict[str, list[dict]] | None,
) -> list[Recommendation]:
    """Matrix factorization (SVD) strategy."""
    if not all_user_films:
        logger.error("SVD strategy requires all user data.")
        return []

    if username not in all_user_films:
        logger.error(f"No data for '{username}'.")
        return []

    n_users = len(all_user_films)
    n_ratings = sum(
        1 for films in all_user_films.values()
        for f in films if f.get('rating')
    )

    if n_users < 10 or n_ratings < 100:
        logger.warning(
            f"SVD works best with more data (have {n_users} users, {n_ratings} ratings). "
            "Consider using 'metadata' strategy or running 'discover' to add more users."
        )

    cache_path = SVD_CACHE_PATH
    n_factors = min(50, n_users - 1)
    use_implicit = True
    implicit_weight = 0.3
    fingerprint = SVDRecommender.compute_fingerprint(
        all_user_films,
        hyperparams={
            "n_factors": n_factors,
            "use_implicit": use_implicit,
            "implicit_weight": implicit_weight,
        },
    )
    svd = SVDRecommender.load(cache_path, expected_fingerprint=fingerprint)

    if svd:
        logger.info(f"Using cached SVD model ({n_users} users, {n_ratings} ratings).")
    else:
        logger.info(f"Fitting SVD model on {n_users} users...")
        svd = SVDRecommender(n_factors=n_factors, use_implicit=use_implicit, implicit_weight=implicit_weight)
        try:
            svd.fit(all_user_films)
        except ValueError as e:
            logger.error(f"Unable to fit SVD model: {e}")
            logger.error("Try the 'metadata' strategy or add more rated films.")
            return []

        cache_metadata = {
            "fingerprint": fingerprint,
            "n_users": n_users,
            "n_items": len(svd.item_index) if svd.item_index else 0,
            "n_ratings": n_ratings,
            "hyperparams": {
                "n_factors": n_factors,
                "use_implicit": use_implicit,
                "implicit_weight": implicit_weight,
            },
        }
        svd.save(cache_path, metadata=cache_metadata)

    seen_slugs = {f['slug'] for f in user_films}
    svd_recs = svd.recommend(username, seen_slugs, n=args.limit * 3)

    recs: list[Recommendation] = []
    for slug, predicted_rating in svd_recs:
        if slug not in all_films:
            continue

        film = all_films[slug]

        year = film.get('year')
        if args.min_year and year and year < args.min_year:
            continue
        if args.max_year and year and year > args.max_year:
            continue

        film_genres = load_json(film.get('genres', []))
        if args.genres:
            genres_lower = [g.lower() for g in args.genres]
            if not any(g in film_genres for g in genres_lower):
                continue
        if args.exclude_genres:
            exclude_lower = [g.lower() for g in args.exclude_genres]
            if any(g in film_genres for g in exclude_lower):
                continue

        if args.min_rating and film.get('avg_rating') and film['avg_rating'] < args.min_rating:
            continue

        recs.append(Recommendation(
            slug=slug,
            title=film.get('title', slug),
            year=year,
            score=predicted_rating,
            reasons=[f"Predicted rating: {predicted_rating:.1f}★"]
        ))

        if len(recs) >= args.limit:
            break

    return recs
