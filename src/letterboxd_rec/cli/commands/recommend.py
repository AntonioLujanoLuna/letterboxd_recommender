"""Recommendation-related CLI commands."""

import argparse
import json
import logging
from pathlib import Path

from ...database import init_db, get_db, load_json
from ...config import DB_PATH
from ...recommender import MetadataRecommender, CollaborativeRecommender
from ...profile import build_profile
from ...group_recommender import recommend_for_group
from ..utils.helpers import _validate_username, _parse_weights, film_url
from ..utils.loaders import (
    _load_all_user_films,
    _load_recommendation_data,
    _load_films_with_filters,
    _warn_missing_metadata,
)
from ..utils.strategies import (
    _run_metadata_strategy,
    _run_collaborative_strategy,
    _run_graph_strategy,
    _run_svd_strategy,
    _run_ease_strategy,
    _run_content_strategy,
    _run_movielens_strategy,
    _run_blend_strategy,
    apply_sparse_popularity,
    filters_active,
)
from ...recommenders.fusion import sparse_popularity_weight
from ..utils.output import _output_recommendations

logger = logging.getLogger(__name__)


# EASE beat every other strategy in offline evaluation (see README), but it learns from other users:
# with fewer than this many users with data it has nothing to generalize from.
MIN_USERS_FOR_EASE = 20


def _default_strategy() -> str:
    """'blend' when the MovieLens model is built, 'ease' when the database holds enough users to
    train it, else the single-user metadata strategy."""
    if not DB_PATH.exists():
        return 'metadata'
    try:
        with get_db(read_only=True) as conn:
            n_users = conn.execute("SELECT COUNT(DISTINCT username) FROM user_films").fetchone()[0]
    except Exception:  # noqa: BLE001  (no schema yet, locked db...): the safe default still works
        return 'metadata'
    if n_users < MIN_USERS_FOR_EASE:
        return 'metadata'
    from ...movielens import ML_MODEL_PATH
    return 'blend' if ML_MODEL_PATH.exists() else 'ease'


FULL_FILTER_STRATEGIES = frozenset({'ease', 'movielens', 'blend', 'svd'})
SPARSE_FALLBACK_STRATEGIES = frozenset({'ease', 'blend'})  # popularity mixed in for short histories


def _uses_extended_filters(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, 'min_runtime', None) or getattr(args, 'max_runtime', None)
        or getattr(args, 'exclude_shorts', False) or getattr(args, 'exclude_animation', False)
        or getattr(args, 'countries', None) or getattr(args, 'languages', None)
        or getattr(args, 'actors', None) or getattr(args, 'directors', None)
    )


def apply_decade(args: argparse.Namespace) -> bool:
    """Turn --decade (1990 or 1990s) into min/max year, intersected with any explicit years.

    Returns False (after logging why) if the value isn't a decade.
    """
    decade = getattr(args, 'decade', None)
    if not decade:
        return True
    text = str(decade).strip().lower().rstrip('s')
    if not (text.isdigit() and len(text) == 4 and int(text) % 10 == 0):
        logger.error(f"--decade must be a year ending in 0, like 1990 or 1990s (got {decade!r})")
        return False
    start = int(text)
    args.min_year = max(args.min_year or start, start)
    args.max_year = min(args.max_year or start + 9, start + 9)
    return True


def cmd_recommend(args: argparse.Namespace) -> None:
    """Generate recommendations."""
    username = _validate_username(args.username)
    novelty = getattr(args, 'novelty', 0.0) or 0.0
    if not 0.0 <= novelty <= 1.0:
        logger.error(f"--novelty must be between 0 and 1 (got {novelty})")
        return
    if not apply_decade(args):
        return
    strategy = getattr(args, 'strategy', None)
    if strategy is None:
        strategy = _default_strategy()
        args.strategy = strategy
        logger.info(f"Using the '{strategy}' strategy (override with --strategy)")

    if strategy not in FULL_FILTER_STRATEGIES and _uses_extended_filters(args):
        logger.warning(
            f"Runtime, short, animation, country and language filters only apply to the "
            f"{', '.join(sorted(FULL_FILTER_STRATEGIES))} strategies; they are ignored by '{strategy}'."
        )

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
        'svd': _run_svd_strategy,
        'graph': _run_graph_strategy,
        'ease': _run_ease_strategy,
        'content': _run_content_strategy,
        'movielens': _run_movielens_strategy,
        'blend': _run_blend_strategy,
    }

    handler = strategy_handlers[strategy]
    sparse = strategy in SPARSE_FALLBACK_STRATEGIES and sparse_popularity_weight(len(user_films)) > 0
    if sparse:
        run_args = argparse.Namespace(**vars(args))
        run_args.limit = max(args.limit * 3, 60)   # deep list to fuse with popularity
    else:
        run_args = args
    recs = handler(user_films, all_films, run_args, username, user_lists, all_user_films)
    if sparse:
        recs = apply_sparse_popularity(recs, user_films, all_films, all_user_films, args)

    if len(recs) < args.limit and filters_active(args):
        logger.warning(f"Only {len(recs)} of the {args.limit} requested films match your filters "
                       f"(among the films in the database the model can score); loosen a filter or crawl more films.")

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
                "url": film_url(rec.slug),
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
def cmd_evaluate(args: argparse.Namespace) -> None:
    """Score strategies offline by hiding some of each user's liked films."""
    from ...evaluation import EVALUABLE_STRATEGIES, format_results_table, run_evaluation

    strategies = args.strategies or list(EVALUABLE_STRATEGIES)
    if not args.strategies:
        from ...movielens import ML_MODEL_PATH
        if ML_MODEL_PATH.exists():
            strategies.extend(["movielens", "blend"])
    if "popularity" not in strategies:
        strategies = ["popularity", *strategies]  # always report the baseline

    no_data_msg = (
        f"No scraped data in {DB_PATH.resolve()}. Run 'python main.py scrape YOUR_USERNAME' "
        "and then 'python main.py discover' to add other users before evaluating "
        "(set LETTERBOXD_DB if your database lives elsewhere)."
    )
    if not DB_PATH.exists():  # don't let sqlite create an empty file
        logger.error(no_data_msg)
        return

    with get_db(read_only=True) as conn:
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='user_films'"
        ).fetchone()
        all_user_films = _load_all_user_films(conn) if has_table else {}
        films = _load_films_with_filters(conn) if has_table else {}

    if not all_user_films:
        logger.error(no_data_msg)
        return
    if len(all_user_films) < 2:
        logger.error(
            "Only one user in the database. Evaluation needs other users "
            "(run 'python main.py discover') so the collaborative strategies have data."
        )
        return

    users = [_validate_username(u) for u in args.users] if args.users else None
    split, results = run_evaluation(
        all_user_films,
        films,
        strategies=strategies,
        k=args.k,
        holdout_frac=args.holdout,
        min_positives=args.min_positives,
        max_users=args.max_users,
        users=users,
        seed=args.seed,
        progress=not args.json,
    )

    if not split.test:
        logger.error(
            "No users qualified for evaluation (need %d+ films rated >= 3.5 or liked). "
            "Lower --min-positives or add users.", args.min_positives,
        )
        return

    if args.json:
        payload = {
            "k": args.k,
            "holdout_frac": args.holdout,
            "seed": args.seed,
            "n_users": len(split.test),
            "n_held_out": sum(len(v) for v in split.test.values()),
            "results": [r.as_dict(include_per_user=args.per_user) for r in results],
        }
        print(json.dumps(payload, indent=2))
        return

    n_held = sum(len(v) for v in split.test.values())
    logger.info(
        f"\nEvaluated {len(split.test)} users, {n_held} hidden films "
        f"({args.holdout:.0%} of each user's positives, seed {args.seed})\n"
    )
    logger.info(format_results_table(results))
    logger.info(
        "\nrecall/ndcg/hit: higher is better. 'vs pop' is the NDCG change against the "
        "popularity baseline.\nThe graph strategy is not evaluated: it reads the live "
        "database, so hidden films would leak in."
    )


def cmd_movielens_build(args: argparse.Namespace) -> None:
    """Download (optionally) MovieLens, fit EASE, report held-out quality, save the model."""
    from ...movielens import ML_DIR, ML_DOWNLOAD_URL, build_model

    ml_dir = Path(args.dir) if args.dir else ML_DIR
    if not (ml_dir / "ratings.csv").exists() and not (ml_dir / "ratings.npz").exists():
        if not args.download:
            logger.error(f"MovieLens not found in {ml_dir}. Re-run with --download, or download "
                         f"{ML_DOWNLOAD_URL} and unzip it into {ml_dir.parent}.")
            return
        import httpx
        import zipfile
        ml_dir.parent.mkdir(parents=True, exist_ok=True)
        zip_path = ml_dir.parent / "ml-32m.zip"
        logger.info(f"Downloading {ML_DOWNLOAD_URL} (~240 MB)...")
        with httpx.stream("GET", ML_DOWNLOAD_URL, follow_redirects=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(zip_path, "wb") as fh:
                for chunk in resp.iter_bytes(1 << 20):
                    fh.write(chunk)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(ml_dir.parent)

    model, report = build_model(
        ml_dir=ml_dir, max_items=args.max_items, lambdas=tuple(args.lambdas),
        eval_users=args.eval_users, k=args.k, seed=args.seed,
    )
    k = report.k
    logger.info(f"\nMovieLens: {report.n_ratings:,} ratings, {report.n_users:,} users, "
                f"{report.n_items:,} films kept. Held-out test: {report.n_eval_users:,} users, "
                f"20% of their liked films hidden.\n")
    base = report.results[0][f"ndcg@{k}"]
    logger.info(f"{'model':<22}{f'recall@{k}':>11}{f'ndcg@{k}':>10}{'vs pop':>9}{'fit s':>8}")
    for r in report.results:
        name = r["model"] if r["lambda"] is None else f"ease (lambda={r['lambda']:g})"
        lift = "" if r["lambda"] is None or not base else f"{(r[f'ndcg@{k}'] / base - 1) * 100:+.0f}%"
        best = "  <- saved" if r["lambda"] == report.best_lambda else ""
        logger.info(f"{name:<22}{r[f'recall@{k}']:>11.4f}{r[f'ndcg@{k}']:>10.4f}{lift:>9}{r['fit_seconds']:>8.0f}{best}")
    logger.info("\nNext: python main.py recommend YOUR_USERNAME --strategy movielens")


