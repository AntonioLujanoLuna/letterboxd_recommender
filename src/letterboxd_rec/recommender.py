"""Recommender module - thin wrapper for backward compatibility.

All functionality has been moved to the recommenders/ package.
Import from recommenders directly for new code.
"""

from .recommenders import (
    # Core types
    Recommendation,
    AttributeConfig,
    RuleFunc,
    # Scoring
    ScoringEngine,
    ATTRIBUTE_CONFIGS,
    DEFAULT_SCORING_RULES,
    _confidence_weight,
    _director_confidence_rule,
    _genre_pair_rule,
    _decade_rule,
    _community_rating_rule,
    _popularity_rule,
    _momentum_rule,
    # Vectorization
    TfidfEmbedder,
    # Fusion
    _fuse_normalized,
    # Recommenders
    MetadataRecommender,
    CollaborativeRecommender,
)

# Re-export from profile for backward compatibility
from .profile import UserProfile, build_profile

# Re-export from config for backward compatibility
from .config import NEGATIVE_PENALTY_MULTIPLIER, NEGATIVE_PENALTY_MULTIPLIERS, WEIGHTS

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
    # Profile (backward compatibility)
    "UserProfile",
    "build_profile",
    # Config (backward compatibility)
    "NEGATIVE_PENALTY_MULTIPLIER",
    "NEGATIVE_PENALTY_MULTIPLIERS",
    "WEIGHTS",
]
