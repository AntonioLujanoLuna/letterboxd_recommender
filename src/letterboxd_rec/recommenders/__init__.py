"""Recommenders package for film recommendations.

This package provides metadata-based and collaborative filtering recommenders.
"""

from .recommendation import Recommendation
from .scoring import (
    AttributeConfig,
    RuleFunc,
    ATTRIBUTE_CONFIGS,
    DEFAULT_SCORING_RULES,
    _confidence_weight,
    _director_confidence_rule,
    _genre_pair_rule,
    _decade_rule,
    _community_rating_rule,
    _popularity_rule,
    _momentum_rule,
)
from .engine import ScoringEngine
from .vectorization import TfidfEmbedder
from .fusion import _fuse_normalized
from .metadata import MetadataRecommender
from .collaborative import CollaborativeRecommender

__all__ = [
    # Core types
    "Recommendation",
    "AttributeConfig",
    "RuleFunc",
    # Scoring
    "ScoringEngine",
    "ATTRIBUTE_CONFIGS",
    "DEFAULT_SCORING_RULES",
    "_confidence_weight",
    "_director_confidence_rule",
    "_genre_pair_rule",
    "_decade_rule",
    "_community_rating_rule",
    "_popularity_rule",
    "_momentum_rule",
    # Vectorization
    "TfidfEmbedder",
    # Fusion
    "_fuse_normalized",
    # Recommenders
    "MetadataRecommender",
    "CollaborativeRecommender",
]
