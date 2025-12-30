"""Output formatting utilities for CLI commands."""

import argparse
import json
import logging

from ...database import load_json
from ...recommender import Recommendation
from ...profile import UserProfile

logger = logging.getLogger(__name__)


def _output_recommendations(
    recs: list[Recommendation],
    all_films: dict[str, dict],
    args: argparse.Namespace,
    username: str,
    strategy: str,
    recommender=None,
    profile: 'UserProfile | None' = None,
    user_films: list[dict] | None = None,
) -> None:
    """Format and log recommendations in the requested format."""
    recs = recs or []
    output_format = getattr(args, 'format', 'text')
    explain_mode = getattr(args, 'explain', False)
    diversity_report = getattr(args, 'diversity_report', False)

    if output_format == 'json':
        output = []
        for r in recs:
            film = all_films.get(r.slug, {})
            rec_data = {
                "title": r.title,
                "year": r.year,
                "slug": r.slug,
                "score": round(r.score, 2),
                "reasons": r.reasons,
                "url": f"https://letterboxd.com/film/{r.slug}/",
                "directors": load_json(film.get('directors', [])),
                "genres": load_json(film.get('genres', [])),
                "cast": load_json(film.get('cast', []))[:5],
                "themes": load_json(film.get('themes', [])),
                "countries": load_json(film.get('countries', [])),
                "avg_rating": film.get('avg_rating'),
                "rating_count": film.get('rating_count')
            }

            # Add explanation if requested and available
            if explain_mode and recommender and profile and user_films:
                explanation = recommender.explain_recommendation_detailed(film, profile, user_films)
                rec_data["explanation"] = {
                    "summary": explanation["summary"],
                    "confidence": explanation["confidence"],
                    "discovery_potential": explanation["discovery_potential"],
                    "similar_films_you_liked": explanation["similar_films_you_liked"],
                    "contribution_breakdown": explanation.get("contribution_breakdown", {}),
                }

            output.append(rec_data)

        # Add diversity metrics if requested
        if diversity_report and recommender:
            metrics = recommender.compute_recommendation_diversity(recs)
            logger.info(json.dumps({"recommendations": output, "diversity": metrics}, indent=2))
        else:
            logger.info(json.dumps(output, indent=2))

    elif output_format == 'csv':
        logger.info("Title,Year,URL,Score,Reasons")
        for r in recs:
            reasons = "; ".join(r.reasons).replace('"', '""')
            logger.info(f'"{r.title}",{r.year},https://letterboxd.com/film/{r.slug}/,{r.score:.2f},"{reasons}"')

    elif output_format == 'markdown':
        logger.info(f"\n# Top {len(recs)} recommendations for {username} ({strategy})\n")
        for i, r in enumerate(recs, 1):
            logger.info(f"## {i}. [{r.title} ({r.year})](https://letterboxd.com/film/{r.slug}/)")
            logger.info(f"**Score**: {r.score:.1f}  ")
            logger.info(f"**Why**: {', '.join(r.reasons)}\n")

    else:  # text format
        logger.info(f"\nTop {len(recs)} recommendations for {username} ({strategy}):")

        for i, r in enumerate(recs, 1):
            logger.info(f"{i}. {r.title} ({r.year}) - Score: {r.score:.1f}")
            logger.info(f"   Why: {', '.join(r.reasons)}")

            # Detailed explanation if requested
            if explain_mode and recommender and profile and user_films:
                film = all_films.get(r.slug, {})
                explanation = recommender.explain_recommendation_detailed(film, profile, user_films)

                logger.info(f"   Confidence: {explanation['confidence']:.0%}")
                logger.info(f"   Discovery: {explanation['discovery_potential']}")

                if explanation['similar_films_you_liked']:
                    similar = explanation['similar_films_you_liked'][0]
                    rating_str = f" (you rated {similar['your_rating']}★)" if similar.get('your_rating') else ""
                    logger.info(f"   Because you liked: {similar['title']}{rating_str} - {similar['connection']}")

                if explanation.get('counterfactuals'):
                    logger.info(f"   Note: {explanation['counterfactuals'][0]}")

                logger.info("")  # Blank line between detailed entries

    # Diversity report at the end
    if diversity_report and recommender and output_format != 'json':
        metrics = recommender.compute_recommendation_diversity(recs)
        logger.info(f"\n{'=' * 50}")
        logger.info("Diversity Report:")
        logger.info(f"  Overall diversity: {metrics['diversity_score']:.0%}")
        logger.info(f"  Genres: {metrics['unique_genres']} unique ({metrics['genre_diversity']:.0%} entropy)")
        logger.info(f"  Directors: {metrics['unique_directors']} unique ({metrics['director_diversity']:.0%} entropy)")
        logger.info(f"  Countries: {metrics['unique_countries']} unique ({metrics['country_diversity']:.0%} entropy)")
        if metrics['decade_range']:
            logger.info(f"  Decades: {metrics['decade_range'][0]}s to {metrics['decade_range'][1]}s ({metrics['decade_diversity']:.0%} entropy)")
