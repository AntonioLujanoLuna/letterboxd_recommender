"""
Utilities for loading and applying learnable feature weights.

Weights are stored per attribute value (e.g., per genre/director/decade)
and loaded from a JSON file. When no weights are available, the system
falls back to neutral multipliers (1.0) to preserve existing behavior.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import FEATURE_WEIGHTS_PATH

logger = logging.getLogger(__name__)

# Bound weights to avoid extreme amplification
MIN_WEIGHT = 0.2
MAX_WEIGHT = 3.0
DEFAULT_WEIGHT = 1.0


def _clamp_weight(value: float) -> float:
    return max(MIN_WEIGHT, min(MAX_WEIGHT, value))


@dataclass
class FeatureWeights:
    """Container for per-feature multipliers."""

    genre: dict[str, float] = field(default_factory=dict)
    director: dict[str, float] = field(default_factory=dict)
    decade: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize keys and clamp values to keep behavior predictable."""
        def _normalize(table: dict[Any, Any], lower_keys: bool = True) -> dict[str, float]:
            normalized: dict[str, float] = {}
            for key, value in (table or {}).items():
                str_key = str(key)
                norm_key = str_key.lower() if lower_keys else str_key
                try:
                    normalized[norm_key] = _clamp_weight(float(value))
                except (TypeError, ValueError):
                    normalized[norm_key] = DEFAULT_WEIGHT
            return normalized

        self.genre = _normalize(self.genre, lower_keys=True)
        self.director = _normalize(self.director, lower_keys=True)
        # Decades are kept as string keys to match persisted JSON shape
        self.decade = _normalize(self.decade, lower_keys=False)

    def factor(self, attr: str, key: str | int | None) -> float:
        """
        Return multiplier for an attribute value.

        Falls back to neutral 1.0 if the attribute/value is not present.
        """
        if key is None:
            return DEFAULT_WEIGHT

        table = getattr(self, attr, None)
        if not isinstance(table, dict):
            return DEFAULT_WEIGHT

        lookup_keys = [key]
        if isinstance(key, str):
            lookup_keys.append(key.lower())
        else:
            lookup_keys.append(str(key))

        for candidate in lookup_keys:
            if candidate in table:
                try:
                    return _clamp_weight(float(table[candidate]))
                except (TypeError, ValueError):
                    return DEFAULT_WEIGHT

        return DEFAULT_WEIGHT

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "genre": self.genre,
            "director": self.director,
            "decade": self.decade,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureWeights":
        return cls(
            genre={str(k).lower(): float(v) for k, v in payload.get("genre", {}).items()},
            director={
                str(k).lower(): float(v) for k, v in payload.get("director", {}).items()
            },
            decade={str(k): float(v) for k, v in payload.get("decade", {}).items()},
            metadata=payload.get("metadata", {}),
        )


def load_feature_weights(path: str | Path | None = None) -> FeatureWeights | None:
    """Load weights from disk; return None if missing or invalid."""
    weight_path = Path(path) if path else FEATURE_WEIGHTS_PATH
    if not weight_path.exists():
        logger.debug("Feature weights file not found at %s; using defaults", weight_path)
        return None

    try:
        return FeatureWeights.from_dict(json.loads(weight_path.read_text()))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load feature weights from %s: %s", weight_path, exc)
        return None


def save_feature_weights(weights: FeatureWeights, path: str | Path | None = None) -> Path:
    """Persist weights to disk (used by trainer scripts)."""
    weight_path = Path(path) if path else FEATURE_WEIGHTS_PATH
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path.write_text(json.dumps(weights.to_dict(), indent=2))
    return weight_path

