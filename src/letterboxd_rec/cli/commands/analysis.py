"""Analysis-related CLI commands (profile, similar, triage, gaps, etc.)."""

import argparse
import logging

from ...database import (
    init_db,
    get_db,
    load_json,
    load_films_by_attribute,
    compute_and_store_idf,
)
from ...recommender import MetadataRecommender
from ...profile import build_profile
from ...matrix_factorization import SVDRecommender
from ..utils.helpers import _validate_username, _require_slug
from ..utils.loaders import _load_all_user_films

logger = logging.getLogger(__name__)


def cmd_explore(args: argparse.Namespace) -> None:
    """Explore films by attribute (director, genre, actor)."""
    init_db()

    results = load_films_by_attribute(
        args.attribute_type,
        [args.value],
        limit_per_value=args.limit
    )

    films = results.get(args.value, [])
    if not films:
        logger.info(f"No films found for {args.attribute_type}: {args.value}")
        return

    logger.info(f"\nTop {len(films)} films for {args.attribute_type} '{args.value}':")
    logger.info("-" * 50)

    for i, film in enumerate(films, 1):
        title = film.get('title', film.get('slug', 'Unknown'))
        year = film.get('year', '?')
        rating = film.get('avg_rating') or 0
        rating_count = film.get('rating_count') or 0

        rating_str = f"{rating:.1f}*" if rating else "N/A"
        count_str = f"({rating_count:,} ratings)" if rating_count else ""

        logger.info(f"  {i:2}. {title} ({year}) - {rating_str} {count_str}")


def cmd_profile(args: argparse.Namespace) -> None:
    """Show user's preference profile."""
    # Validate username
    username = _validate_username(args.username)

    with get_db() as conn:
        user_films = [dict(r) for r in conn.execute("""
            SELECT film_slug as slug, rating, watched, watchlisted, liked
            FROM user_films WHERE username = ?
        """, (username,))]
        all_films = {r['slug']: dict(r) for r in conn.execute("SELECT * FROM films")}

    if not user_films:
        logger.error(f"No data for '{username}'. Run: python main.py scrape {username}")
        return

    profile = build_profile(user_films, all_films, username=username)

    logger.info(f"\nProfile for {username}")
    logger.info(f"  Films: {profile.n_films} ({profile.n_rated} rated, {profile.n_liked} liked)")
    if profile.avg_liked_rating:
        logger.info(f"  Average rating: {profile.avg_liked_rating:.2f}*")

    if profile.genres:
        logger.info("\nTop genres:")
        for g, score in sorted(profile.genres.items(), key=lambda x: -x[1])[:10]:
            logger.info(f"  {g}: {score:+.2f}")

    if profile.directors:
        logger.info("\nTop directors:")
        for d, score in sorted(profile.directors.items(), key=lambda x: -x[1])[:10]:
            logger.info(f"  {d}: {score:+.2f}")

    if profile.actors:
        logger.info("\nTop actors:")
        for a, score in sorted(profile.actors.items(), key=lambda x: -x[1])[:10]:
            logger.info(f"  {a}: {score:+.2f}")

    if profile.decades:
        logger.info("\nDecade preferences:")
        for dec in sorted(profile.decades.keys()):
            score = profile.decades[dec]
            bar_length = int(max(0, score * 2))
            bar = "#" * bar_length
            logger.info(f"  {dec}s: {bar} ({score:+.1f})")


def cmd_similar(args: argparse.Namespace) -> None:
    """Find films similar to a specific film."""
    # Validate slug
    slug = _require_slug(args.slug)

    with get_db() as conn:
        all_films = [dict(r) for r in conn.execute("SELECT * FROM films")]

    if not all_films:
        logger.error("No films in database. Run scrape first.")
        return

    recommender = MetadataRecommender(all_films)
    recs = recommender.similar_to(slug, n=args.limit)

    if not recs:
        logger.error(f"No film found with slug '{args.slug}'")
        return

    logger.info(f"\nFilms similar to {args.slug}:")
    for i, r in enumerate(recs, 1):
        logger.info(f"{i}. {r.title} ({r.year}) - Score: {r.score:.1f}")
        logger.info(f"   Why: {', '.join(r.reasons)}")


def cmd_triage(args: argparse.Namespace) -> None:
    """Rank user's watchlist by predicted enjoyment."""
    # Validate username
    username = _validate_username(args.username)

    with get_db() as conn:
        user_films = [dict(r) for r in conn.execute("""
            SELECT film_slug as slug, rating, watched, watchlisted, liked
            FROM user_films WHERE username = ?
        """, (username,))]

        watchlist = [r['film_slug'] for r in conn.execute("""
            SELECT film_slug FROM user_films
            WHERE username = ? AND watchlisted = 1
        """, (username,))]

        all_films = {r['slug']: dict(r) for r in conn.execute("SELECT * FROM films")}

    if not user_films:
        logger.error(f"No data for '{username}'. Run: python main.py scrape {username}")
        return

    if not watchlist:
        logger.error(f"No watchlist data for '{username}'.")
        return

    recommender = MetadataRecommender(list(all_films.values()))
    recs = recommender.recommend_from_candidates(user_films, watchlist, n=args.limit)

    logger.info(f"\nWatchlist Triage for {username} (Top {len(recs)}):")
    for i, r in enumerate(recs, 1):
        logger.info(f"{i}. {r.title} ({r.year}) - Score: {r.score:.1f}")
        logger.info(f"   Why: {', '.join(r.reasons)}")


def cmd_svd_info(args: argparse.Namespace) -> None:
    """Show SVD model diagnostics."""
    with get_db() as conn:
        all_user_films = _load_all_user_films(conn)

    if len(all_user_films) < 5:
        logger.error("Need at least 5 users for SVD. Run 'discover' first.")
        return

    svd = SVDRecommender(n_factors=min(50, len(all_user_films) - 1))
    try:
        svd.fit(all_user_films)
    except ValueError as e:
        logger.error(f"Unable to fit SVD model: {e}")
        return

    logger.info(f"\nSVD Model Info:")
    logger.info(f"  Users: {len(svd.user_index)}")
    logger.info(f"  Films: {len(svd.item_index)}")
    logger.info(f"  Latent factors: {svd.n_factors}")
    logger.info(f"  Global mean rating: {svd.global_mean:.2f}*")

    # Show user with highest/lowest bias
    if svd.user_biases is not None:
        usernames = list(svd.user_index.keys())
        harshest_idx = svd.user_biases.argmin()
        generous_idx = svd.user_biases.argmax()
        logger.info(f"\n  Most generous rater: {usernames[generous_idx]} (+{svd.user_biases[generous_idx]:.2f})")
        logger.info(f"  Harshest rater: {usernames[harshest_idx]} ({svd.user_biases[harshest_idx]:.2f})")


def cmd_gaps(args: argparse.Namespace) -> None:
    """Find gaps in filmography of favorite directors."""
    # Validate username
    username = _validate_username(args.username)

    with get_db() as conn:
        user_films = [dict(r) for r in conn.execute("""
            SELECT film_slug as slug, rating, watched, watchlisted, liked
            FROM user_films WHERE username = ?
        """, (username,))]

        all_films = {r['slug']: dict(r) for r in conn.execute("SELECT * FROM films")}

    if not user_films:
        logger.error(f"No data for '{username}'. Run: python main.py scrape {username}")
        return

    recommender = MetadataRecommender(list(all_films.values()))
    gaps = recommender.find_gaps(
        user_films,
        min_director_score=args.min_score,
        limit_per_director=args.limit,
        min_year=args.min_year,
        max_year=args.max_year
    )

    if not gaps:
        logger.error(f"No gaps found for {username}. Try lowering --min-score.")
        return

    logger.info(f"\nFilmography Gaps for {username}:")
    for director, recs in sorted(gaps.items(), key=lambda x: -len(x[1])):
        logger.info(f"\n{director}:")
        for r in recs:
            logger.info(f"  - {r.title} ({r.year}) [{r.score:.1f}]")


def cmd_rebuild_idf(args: argparse.Namespace) -> None:
    """Rebuild IDF (Inverse Document Frequency) scores for all attributes."""
    init_db()

    logger.info("Computing IDF scores for all attributes...")
    logger.info("This may take a while depending on database size...")

    results = compute_and_store_idf()

    if not results:
        logger.error("\nNo films found in database. IDF table is empty.")
        logger.error("Run 'python main.py scrape <username>' to add films first.")
        return

    logger.info("\nIDF computation complete!")
    logger.info("\nAttribute counts:")
    for attr_type, count in sorted(results.items()):
        logger.info(f"  {attr_type}: {count} unique values")

    total = sum(results.values())
    logger.info(f"\nTotal: {total} attribute values indexed")
    logger.info("\nIDF scores will now be used to prioritize rare/distinctive preferences in recommendations.")
