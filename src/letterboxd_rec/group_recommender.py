"""
Group recommendation engine for watch parties.

Finds films that multiple users would enjoy watching together while balancing
individual preferences with group harmony.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from .database import get_db, load_json, load_user_lists
from .profile import UserProfile, build_profile
from .recommender import MetadataRecommender

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Strategy for combining individual scores into a group score."""

    LEAST_MISERY = "least_misery"  # No one should hate it
    MOST_PLEASURE = "most_pleasure"  # Someone should love it
    AVERAGE = "average"  # Democratic balance
    MULTIPLICATIVE = "multiplicative"  # Geometric mean (requires all positive)
    FAIRNESS = "fairness"  # Penalize variance
    APPROVAL = "approval"  # Count users above threshold


@dataclass
class GroupMember:
    """A member of a watch group with their preferences."""

    username: str
    profile: UserProfile
    films: list[dict]
    weight: float = 1.0  # For weighted voting (e.g., birthday person gets 2x)


@dataclass
class GroupRecommendation:
    """A film recommendation for a group."""

    slug: str
    title: str
    year: int | None

    # Scores
    group_score: float
    user_scores: dict[str, float]

    # Explanations
    consensus_reasons: list[str]
    user_reasons: dict[str, list[str]]
    warnings: list[str]

    # Metrics
    agreement_score: float
    min_user_score: float
    max_user_score: float

    @property
    def is_unanimous(self) -> bool:
        """True if all users have positive scores."""
        return all(score > 0 for score in self.user_scores.values())

    @property
    def controversial(self) -> bool:
        """True if scores vary widely."""
        return (self.max_user_score - self.min_user_score) > 3.0


@dataclass
class GroupProfile:
    """Aggregated preferences for a group of users."""

    members: list[GroupMember]

    # Consensus preferences (attributes everyone likes)
    consensus_genres: dict[str, float] = field(default_factory=dict)
    consensus_directors: dict[str, float] = field(default_factory=dict)
    consensus_decades: dict[int, float] = field(default_factory=dict)

    # Divisive attributes (some love, some hate)
    divisive_genres: set[str] = field(default_factory=set)
    divisive_directors: set[str] = field(default_factory=set)

    # Shared watchlist (films multiple people want to see)
    shared_watchlist: dict[str, int] = field(default_factory=dict)  # slug -> count

    # Compatibility metrics
    overall_compatibility: float = 0.0
    pairwise_compatibility: dict[tuple[str, str], float] = field(default_factory=dict)


class GroupRecommender:
    """
    Recommender for groups of users watching together.

    Handles the complex task of finding films that balance multiple users'
    preferences while avoiding divisive choices.
    """

    def __init__(
        self,
        all_films: dict[str, dict],
        strategy: AggregationStrategy = AggregationStrategy.FAIRNESS,
        approval_threshold: float = 2.0,
    ):
        self.all_films = all_films
        self.strategy = strategy
        self.approval_threshold = approval_threshold
        self._recommender = MetadataRecommender(list(all_films.values()))

    def create_group(
        self,
        usernames: list[str],
        weights: dict[str, float] | None = None,
    ) -> GroupProfile:
        """
        Create a group profile from multiple users.

        Args:
            usernames: List of Letterboxd usernames
            weights: Optional per-user weights (e.g., {"birthday_person": 2.0})

        Returns:
            GroupProfile with aggregated preferences and compatibility metrics
        """
        weights = weights or {}
        members: list[GroupMember] = []

        with get_db(read_only=True) as conn:
            for username in usernames:
                user_films = [
                    dict(row)
                    for row in conn.execute(
                        """
                        SELECT film_slug as slug, rating, watched, watchlisted, liked, scraped_at
                        FROM user_films
                        WHERE username = ?
                        """,
                        (username,),
                    )
                ]

                if not user_films:
                    logger.warning("No data for user '%s', skipping", username)
                    continue

                user_lists = load_user_lists(username)
                profile = build_profile(
                    user_films,
                    self.all_films,
                    user_lists=user_lists,
                    username=username,
                )

                members.append(
                    GroupMember(
                        username=username,
                        profile=profile,
                        films=user_films,
                        weight=weights.get(username, 1.0),
                    )
                )

        if len(members) < 2:
            raise ValueError(f"Need at least 2 valid users, got {len(members)}")

        return self._build_group_profile(members)

    def _build_group_profile(self, members: list[GroupMember]) -> GroupProfile:
        """Analyze group preferences and find consensus/conflicts."""
        group = GroupProfile(members=members)

        # Find consensus genres (everyone rates positively)
        all_genre_scores: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for member in members:
            for genre, score in member.profile.genres.items():
                all_genre_scores[genre].append(
                    (member.username, score * member.weight)
                )

        for genre, user_scores in all_genre_scores.items():
            scores = [s for _, s in user_scores]
            if len(scores) == len(members):  # Everyone has an opinion
                if all(s > 0 for s in scores):
                    group.consensus_genres[genre] = min(scores)  # Conservative
                elif any(s > 1 for s in scores) and any(s < -0.5 for s in scores):
                    group.divisive_genres.add(genre)

        # Same for directors
        all_director_scores: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for member in members:
            for director, score in member.profile.directors.items():
                all_director_scores[director].append(
                    (member.username, score * member.weight)
                )

        for director, user_scores in all_director_scores.items():
            scores = [s for _, s in user_scores]
            if len(scores) >= 2:  # At least 2 people know this director
                if all(s > 0 for s in scores):
                    group.consensus_directors[director] = min(scores)
                elif max(scores) > 2 and min(scores) < -1:
                    group.divisive_directors.add(director)

        # Find shared watchlist items
        watchlist_counts: dict[str, int] = defaultdict(int)
        for member in members:
            for film in member.films:
                if film.get("watchlisted"):
                    watchlist_counts[film["slug"]] += 1

        group.shared_watchlist = {
            slug: count for slug, count in watchlist_counts.items() if count >= 2
        }

        # Compute pairwise compatibility
        for i, m1 in enumerate(members):
            for m2 in members[i + 1 :]:
                compat = self._compute_compatibility(m1.profile, m2.profile)
                group.pairwise_compatibility[(m1.username, m2.username)] = compat

        if group.pairwise_compatibility:
            group.overall_compatibility = (
                sum(group.pairwise_compatibility.values())
                / len(group.pairwise_compatibility)
            )

        return group

    def _compute_compatibility(self, p1: UserProfile, p2: UserProfile) -> float:
        """
        Compute taste compatibility between two users.

        Returns value between 0 (incompatible) and 1 (perfect match).
        """
        scores: list[float] = []

        # Genre overlap
        common_genres = set(p1.genres.keys()) & set(p2.genres.keys())
        if common_genres:
            genre_agreements = []
            for genre in common_genres:
                genre_agreements.append(
                    1.0 if (p1.genres[genre] > 0) == (p2.genres[genre] > 0) else 0.0
                )
            scores.append(sum(genre_agreements) / len(genre_agreements))

        # Director overlap
        common_directors = set(p1.directors.keys()) & set(p2.directors.keys())
        if common_directors:
            director_agreements = []
            for director in common_directors:
                director_agreements.append(
                    1.0
                    if (p1.directors[director] > 0) == (p2.directors[director] > 0)
                    else 0.0
                )
            scores.append(sum(director_agreements) / len(director_agreements))

        # Decade preferences
        common_decades = set(p1.decades.keys()) & set(p2.decades.keys())
        if common_decades:
            decade_corr = sum(p1.decades[d] * p2.decades[d] for d in common_decades)
            norm1 = math.sqrt(sum(v**2 for v in p1.decades.values())) or 1
            norm2 = math.sqrt(sum(v**2 for v in p2.decades.values())) or 1
            scores.append((decade_corr / (norm1 * norm2) + 1) / 2)  # Scale to 0-1

        return sum(scores) / len(scores) if scores else 0.5

    def recommend(
        self,
        group: GroupProfile,
        n: int = 20,
        strategy: AggregationStrategy | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
        genres: list[str] | None = None,
        exclude_divisive: bool = True,
        prioritize_shared_watchlist: bool = True,
    ) -> list[GroupRecommendation]:
        """
        Generate recommendations for the group.

        Args:
            group: GroupProfile from create_group()
            n: Number of recommendations
            strategy: Override default aggregation strategy
            min_year: Minimum release year filter
            max_year: Maximum release year filter
            genres: Required genres (consensus filter)
            exclude_divisive: Automatically exclude divisive genres/directors
            prioritize_shared_watchlist: Boost films multiple people want to see
        """
        strategy = strategy or self.strategy

        seen: set[str] = set()
        for member in group.members:
            seen.update(f["slug"] for f in member.films if f.get("watched"))

        candidates: list[GroupRecommendation] = []

        for slug, film in self.all_films.items():
            if slug in seen:
                continue

            year = film.get("year")
            if min_year and year and year < min_year:
                continue
            if max_year and year and year > max_year:
                continue

            film_genres = set(load_json(film.get("genres", [])))
            film_directors = set(load_json(film.get("directors", [])))

            if genres:
                genres_lower = {g.lower() for g in genres}
                if not (film_genres & genres_lower):
                    continue

            if exclude_divisive:
                if film_genres & group.divisive_genres:
                    continue
                if film_directors & group.divisive_directors:
                    continue

            user_scores: dict[str, float] = {}
            user_reasons: dict[str, list[str]] = {}

            for member in group.members:
                score, reasons, _warnings = self._recommender._score_film(
                    film, member.profile
                )
                user_scores[member.username] = score * member.weight
                user_reasons[member.username] = reasons

            group_score = self._aggregate_scores(list(user_scores.values()), strategy)

            if prioritize_shared_watchlist and slug in group.shared_watchlist:
                watchlist_count = group.shared_watchlist[slug]
                boost = 1.0 + (0.2 * watchlist_count)  # +20% per person
                group_score *= boost

            consensus_reasons = self._find_consensus_reasons(
                user_reasons, group.members
            )

            warnings = [
                f"⚠️ {username} may not enjoy this"
                for username, score in user_scores.items()
                if score < 0
            ]

            scores_list = list(user_scores.values())
            mean_score = sum(scores_list) / len(scores_list)
            variance = sum((s - mean_score) ** 2 for s in scores_list) / len(
                scores_list
            )
            max_variance = (
                (max(scores_list) - min(scores_list)) ** 2 / 4
                if len(scores_list) > 1
                else 1
            )
            agreement = 1.0 - (variance / max_variance) if max_variance > 0 else 1.0

            candidates.append(
                GroupRecommendation(
                    slug=slug,
                    title=film.get("title", slug),
                    year=year,
                    group_score=group_score,
                    user_scores=user_scores,
                    consensus_reasons=consensus_reasons,
                    user_reasons=user_reasons,
                    warnings=warnings,
                    agreement_score=agreement,
                    min_user_score=min(scores_list),
                    max_user_score=max(scores_list),
                )
            )

        candidates.sort(key=lambda rec: -rec.group_score)
        return candidates[:n]

    def _aggregate_scores(
        self,
        scores: list[float],
        strategy: AggregationStrategy,
    ) -> float:
        """Combine individual scores into a group score."""
        if not scores:
            return 0.0

        if strategy == AggregationStrategy.LEAST_MISERY:
            return min(scores)
        if strategy == AggregationStrategy.MOST_PLEASURE:
            return max(scores)
        if strategy == AggregationStrategy.AVERAGE:
            return sum(scores) / len(scores)
        if strategy == AggregationStrategy.MULTIPLICATIVE:
            if all(score > 0 for score in scores):
                return math.prod(scores) ** (1 / len(scores))
            return 0.0
        if strategy == AggregationStrategy.FAIRNESS:
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
            return mean - (variance ** 0.5) * 0.5
        if strategy == AggregationStrategy.APPROVAL:
            approvals = sum(1 for score in scores if score >= self.approval_threshold)
            return approvals + (sum(scores) / len(scores) * 0.1)

        return sum(scores) / len(scores)

    def _find_consensus_reasons(
        self,
        user_reasons: dict[str, list[str]],
        members: list[GroupMember],
    ) -> list[str]:
        """Find reasons that apply to multiple users."""
        reason_counts: dict[str, int] = defaultdict(int)
        for reasons in user_reasons.values():
            for reason in reasons:
                normalized = reason.split(":")[0] if ":" in reason else reason
                reason_counts[normalized] += 1

        threshold = len(members) / 2
        consensus = [
            f"{reason} (all)"
            if count == len(members)
            else f"{reason} ({count}/{len(members)})"
            for reason, count in reason_counts.items()
            if count >= threshold
        ]
        return consensus[:3]

    def triage_watchlists(
        self,
        group: GroupProfile,
        n: int = 20,
    ) -> list[GroupRecommendation]:
        """
        Special mode: rank the intersection of group's watchlists.

        Finds films that multiple people already want to watch and ranks
        them by group fit.
        """
        watchlist_films = list(group.shared_watchlist.keys())
        if not watchlist_films:
            logger.info("No shared watchlist items found")
            return []

        logger.info("Found %s films on multiple watchlists", len(watchlist_films))
        return self.recommend(group, n=n, prioritize_shared_watchlist=True)

    def explain_group(self, group: GroupProfile) -> dict:
        """Generate a human-readable summary of group dynamics."""
        return {
            "member_count": len(group.members),
            "members": [m.username for m in group.members],
            "overall_compatibility": f"{group.overall_compatibility:.0%}",
            "compatibility_label": self._compatibility_label(
                group.overall_compatibility
            ),
            "consensus_genres": list(group.consensus_genres.keys())[:5],
            "divisive_genres": list(group.divisive_genres)[:3],
            "shared_watchlist_count": len(group.shared_watchlist),
            "best_pair": self._best_pair(group),
            "challenging_pair": self._challenging_pair(group),
            "recommendation": self._group_recommendation_hint(group),
        }

    def _compatibility_label(self, score: float) -> str:
        if score >= 0.8:
            return "Excellent match! 🎬"
        if score >= 0.6:
            return "Good compatibility"
        if score >= 0.4:
            return "Some differences to navigate"
        return "Diverse tastes - finding common ground..."

    def _best_pair(self, group: GroupProfile) -> str | None:
        if not group.pairwise_compatibility:
            return None
        best = max(group.pairwise_compatibility.items(), key=lambda item: item[1])
        return f"{best[0][0]} & {best[0][1]} ({best[1]:.0%})"

    def _challenging_pair(self, group: GroupProfile) -> str | None:
        if not group.pairwise_compatibility:
            return None
        worst = min(group.pairwise_compatibility.items(), key=lambda item: item[1])
        if worst[1] < 0.4:
            return f"{worst[0][0]} & {worst[0][1]} ({worst[1]:.0%})"
        return None

    def _group_recommendation_hint(self, group: GroupProfile) -> str:
        if group.overall_compatibility >= 0.7:
            return "You can be adventurous - the group has aligned tastes!"
        if group.divisive_genres:
            genres = ", ".join(list(group.divisive_genres)[:2])
            return f"Consider avoiding {genres} to keep everyone happy"
        if group.shared_watchlist:
            return "Start with your shared watchlist - you all want to see these!"
        return "Stick to crowd-pleasers or highly-rated films"


def recommend_for_group(
    usernames: list[str],
    n: int = 20,
    strategy: str = "fairness",
    weights: dict[str, float] | None = None,
    **filters,
) -> tuple[list[GroupRecommendation], dict]:
    """
    High-level convenience function to get group recommendations.

    Returns (recommendations, group_info) tuple.
    """
    with get_db(read_only=True) as conn:
        all_films = {row["slug"]: dict(row) for row in conn.execute("SELECT * FROM films")}

    strategy_enum = AggregationStrategy(strategy)
    recommender = GroupRecommender(all_films, strategy=strategy_enum)

    group = recommender.create_group(usernames, weights=weights)
    group_info = recommender.explain_group(group)
    recommendations = recommender.recommend(group, n=n, **filters)

    return recommendations, group_info

