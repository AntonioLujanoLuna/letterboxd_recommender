"""Recommendation-related CLI commands."""

import argparse
import logging

from ...database import init_db, get_db, load_json
from ...recommender import MetadataRecommender, CollaborativeRecommender
from ...profile import build_profile
from ...group_recommender import recommend_for_group
from ..utils.helpers import _validate_username, _parse_weights
from ..utils.loaders import (
    _load_all_user_films,
    _load_recommendation_data,
    _warn_missing_metadata,
)
from ..utils.strategies import (
    _run_metadata_strategy,
    _run_collaborative_strategy,
    _run_hybrid_strategy,
    _run_graph_strategy,
    _run_svd_strategy,
)
from ..utils.output import _output_recommendations

logger = logging.getLogger(__name__)


def cmd_recommend(args: argparse.Namespace) -> None:
    """Generate recommendations."""
    username = _validate_username(args.username)
    strategy = getattr(args, 'strategy', 'metadata')

    with get_db() as conn:
        user_films, all_films, all_user_films, user_lists = _load_recommendation_data(
            conn, username, strategy, args
        )

    if not user_films:
        logger.error(f"No data for '{username}'. Run: python main.py scrape {username}")
        return

    _warn_missing_metadata(user_films, all_films, username)

    strategy_handlers = {
        'metadata': _run_metadata_strategy,
        'collaborative': _run_collaborative_strategy,
        'hybrid': _run_hybrid_strategy,
        'svd': _run_svd_strategy,
        'graph': _run_graph_strategy,
    }

    handler = strategy_handlers[strategy]
    recs = handler(user_films, all_films, args, username, user_lists, all_user_films)

    # Build recommender and profile for explanations if needed
    recommender = None
    profile = None
    if getattr(args, 'explain', False) or getattr(args, 'diversity_report', False):
        recommender = MetadataRecommender(list(all_films.values()))
        profile = build_profile(
            user_films,
            all_films,
            user_lists=user_lists,
            username=username,
            weighting_mode=args.weighting_mode,
        )

    _output_recommendations(
        recs, all_films, args, username, strategy,
        recommender=recommender,
        profile=profile,
        user_films=user_films,
    )


def cmd_similar_users(args: argparse.Namespace) -> None:
    """Find users with similar taste (influencers and followers)."""
    init_db()
    username = _validate_username(args.username)

    with get_db(read_only=True) as conn:
        all_user_films = _load_all_user_films(conn)
        all_films = {r['slug']: dict(r) for r in conn.execute("SELECT * FROM films")}

    if username not in all_user_films:
        logger.error(f"No data for '{username}'. Run: python main.py scrape {username}")
        return

    recommender = CollaborativeRecommender(all_user_films, all_films)
    influencers, followers = recommender._find_neighbors_asymmetric(username, k=args.limit)

    if influencers:
        logger.info(f"\nInfluencers (users whose taste predicts yours):")
        logger.info("-" * 50)
        for user, score in influencers:
            film_count = len(all_user_films.get(user, []))
            logger.info(f"  {user}: {score:.2f} similarity ({film_count} films)")
    else:
        logger.info("\nNo influencers found with sufficient overlap.")

    if followers:
        logger.info(f"\nFollowers (users who share your taste):")
        logger.info("-" * 50)
        for user, score in followers:
            film_count = len(all_user_films.get(user, []))
            logger.info(f"  {user}: {score:.2f} similarity ({film_count} films)")
    else:
        logger.info("\nNo followers found with sufficient overlap.")

    # Optionally show taste compatibility summary
    if args.verbose and (influencers or followers):
        logger.info(f"\nTaste Network Summary for {username}:")
        logger.info(f"  Total users in database: {len(all_user_films)}")
        logger.info(f"  Influencers found: {len(influencers)}")
        logger.info(f"  Followers found: {len(followers)}")

        if influencers:
            top_influencer = influencers[0][0]
            logger.info(f"  Closest influencer: {top_influencer}")
            logger.info(f"  Tip: Check what {top_influencer} has rated highly that you haven't seen!")


def cmd_jam(args: argparse.Namespace) -> None:
    """Generate recommendations for a group watch session."""
    usernames = [_validate_username(u) for u in args.usernames]

    if len(usernames) < 2:
        logger.error("Need at least 2 usernames for group recommendations")
        return

    weights = _parse_weights(getattr(args, "weights", None))

    logger.info(f"\n Film Jam: Finding movies for {', '.join(usernames)}\n")

    try:
        recs, group_info = recommend_for_group(
            usernames,
            n=args.limit,
            strategy=args.strategy,
            min_year=args.min_year,
            max_year=args.max_year,
            genres=args.genres,
            exclude_divisive=not args.include_divisive,
            weights=weights,
            triage_watchlist=getattr(args, "triage_watchlist", False),
        )
    except ValueError as exc:  # e.g., not enough valid users
        logger.error(str(exc))
        return

    if args.format == "json":
        import json

        output = [
            {
                "title": rec.title,
                "year": rec.year,
                "slug": rec.slug,
                "group_score": round(rec.group_score, 2),
                "user_scores": {k: round(v, 2) for k, v in rec.user_scores.items()},
                "agreement": round(rec.agreement_score, 2),
                "unanimous": rec.is_unanimous,
                "reasons": rec.consensus_reasons,
                "url": f"https://letterboxd.com/film/{rec.slug}/",
            }
            for rec in recs
        ]
        print(json.dumps({"group_info": group_info, "recommendations": output}, indent=2))
        return

    logger.info("Group Dynamics:")
    logger.info(
        f"  Compatibility: {group_info['overall_compatibility']} - "
        f"{group_info['compatibility_label']}"
    )

    if group_info.get("consensus_genres"):
        logger.info(f"  You all like: {', '.join(group_info['consensus_genres'])}")

    if group_info.get("divisive_genres"):
        logger.info(f"  Mixed feelings on: {', '.join(group_info['divisive_genres'])}")

    if group_info.get("best_pair"):
        logger.info(f"  Best match: {group_info['best_pair']}")

    if group_info.get("challenging_pair"):
        logger.info(f"  Most different: {group_info['challenging_pair']}")

    logger.info(f"  {group_info['recommendation']}")

    if group_info.get("shared_watchlist_count", 0) > 0:
        logger.info(
            f"  {group_info['shared_watchlist_count']} films on multiple watchlists"
        )

    logger.info(f"\n{'=' * 60}")
    if getattr(args, "triage_watchlist", False):
        logger.info(f"Top {len(recs)} Shared Watchlist Picks:")
    else:
        logger.info(f"Top {len(recs)} Films for Your Group:")
    logger.info(f"{'=' * 60}\n")

    for i, rec in enumerate(recs, 1):
        unanimous = "+" if rec.is_unanimous else ""
        logger.info(f"{i}. {rec.title} ({rec.year}) {unanimous}")
        logger.info(
            f"   Group Score: {rec.group_score:.1f} | Agreement: {rec.agreement_score:.0%}"
        )

        score_parts = [f"{user}: {score:.1f}" for user, score in rec.user_scores.items()]
        logger.info(f"   Individual: {' | '.join(score_parts)}")

        if rec.consensus_reasons:
            logger.info(f"   Why: {', '.join(rec.consensus_reasons)}")

        for warning in rec.warnings:
            logger.info(f"   {warning}")

        logger.info("")
