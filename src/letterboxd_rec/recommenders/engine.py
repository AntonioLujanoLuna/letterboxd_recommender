"""Scoring engine for composable recommendation scoring."""

from typing import TYPE_CHECKING

from .scoring import AttributeConfig, RuleFunc

if TYPE_CHECKING:
    from ..profile import UserProfile


class ScoringEngine:
    """Composable scoring pipeline for metadata recommendations."""

    def __init__(self, attribute_configs: list[AttributeConfig], rules: list[RuleFunc] | None = None):
        self.attribute_configs = attribute_configs
        self.rules = rules or []

    def score(
        self,
        recommender,
        film: dict,
        profile: "UserProfile",
    ) -> tuple[float, list[str], list[str]]:
        score = 0.0
        reasons: list[str] = []
        warnings: list[str] = []

        for config in self.attribute_configs:
            attr_score, attr_reasons, attr_warnings = recommender._score_attribute(film, profile, config)
            score += attr_score
            reasons.extend(attr_reasons)
            warnings.extend(attr_warnings)

        for rule in self.rules:
            delta, extra_reasons, extra_warnings = rule(recommender, film, profile, reasons, warnings)
            score += delta
            if extra_reasons:
                reasons.extend(extra_reasons)
            if extra_warnings:
                warnings.extend(extra_warnings)

        return score, reasons, warnings
