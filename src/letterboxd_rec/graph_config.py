import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ITEM_SIM_CACHE_PATH


@dataclass
class GraphConfig:
    """
    Configuration for the heterogeneous graph recommender.

    The defaults mirror the values proposed in the PRD and are intentionally
    conservative so we can tune later without changing call sites.
    """

    # PageRank damping / restart probability
    alpha: float = 0.15
    max_iterations: int = 30
    tolerance: float = 1e-6

    # Relation weight loading / presets
    relation_weights_path: Path | None = None
    relation_preset: str | None = None

    # Global relation-type multipliers used during transition normalization
    relation_weights: dict[str, float] = field(
        default_factory=lambda: {
            # User ⇄ item interactions
            "liked": 1.0,
            "rated_high": 0.9,
            "rated_mid": 0.6,
            "rated_low": 0.0,
            "watched": 0.2,
            "watchlisted": 0.1,
            "user_bridge": 0.4,
            # Content / metadata relations
            "director": 1.2,
            "writer": 1.0,
            "cinematographer": 0.9,
            "composer": 0.8,
            "genre": 0.8,
            "theme": 0.6,
            "country": 0.7,
            "language": 0.5,
            "actor": 0.5,
            "decade": 0.3,
            "similar_to": 0.6,
            "follows": 0.4,
        }
    )

    # Restart distribution weights (per interaction type)
    restart_weights: dict[str, float] = field(
        default_factory=lambda: {
            "liked": 0.70,
            "rated_high": 0.60,
            "rated_mid": 0.20,
            "rated_low": 0.00,
            "watched": 0.02,
            "watchlisted": 0.05,
        }
    )

    # Personalization mixing: how much restart mass to keep on the user node
    restart_user_weight: float = 0.2

    # Optional seed personalization (cold-start); applied when provided explicitly
    seed_restart_weight: float = 0.8

    # User→item interaction edge weights (separate from restart priors)
    interaction_weights: dict[str, float] = field(
        default_factory=lambda: {
            "liked": 0.70,
            "rated_high": 0.60,
            "rated_mid": 0.20,
            "rated_low": 0.00,
            "watched": 0.02,
            "watchlisted": 0.05,
        }
    )

    # Semantic budgets for type-aware normalization
    edge_type_weights: dict[str, float] = field(
        default_factory=lambda: {
            "director": 3.0,
            "writer": 2.5,
            "cinematographer": 1.8,
            "composer": 1.5,
            "genre": 1.5,
            "theme": 1.2,
            "country": 1.2,
            "language": 0.8,
            "actor": 0.7,
            "decade": 0.4,
        }
    )

    # Whether to allocate probability budget per edge type (old behavior)
    # or use simple relation-weighted normalization (new default).
    use_type_budgets: bool = False

    # Temporal decay for user→film edges and restart mass
    use_temporal_decay: bool = True
    temporal_decay_half_life_days: int = 365 * 2
    temporal_decay_min_weight: float = 0.1

    # Item similarity edges
    use_item_similarity_edges: bool = True
    item_similarity_path: Path = ITEM_SIM_CACHE_PATH
    item_similarity_top_k: int = 30
    item_similarity_weight: float = 1.0

    # User social edges (follows). Safe no-op if table/data unavailable.
    use_follow_edges: bool = True
    follow_edge_weight: float = 0.3

    # IDF weighting toggle
    use_idf: bool = True
    idf_floor: float = 0.1
    idf_ceiling: float = 5.0

    # Cache settings
    cache_enabled: bool = True
    cache_path: Path = Path("data/graph_cache.npz")
    cache_version: str = "v1"

    # Explanation settings
    max_explanation_paths: int = 3
    max_path_length: int = 4

    # Guardrails for when to fall back to other strategies
    min_likes_for_graph: int = 3
    min_films_for_graph: int = 20

    def __post_init__(self) -> None:
        # Normalize and validate eagerly so mistakes fail fast.
        self.cache_path = Path(self.cache_path)
        if self.relation_weights_path:
            self.relation_weights_path = Path(self.relation_weights_path)
        if self.item_similarity_path:
            self.item_similarity_path = Path(self.item_similarity_path)
        if self.relation_preset:
            self._apply_relation_preset(self.relation_preset)
        self._maybe_load_relation_weights_file()
        self.validate()

    def validate(self) -> None:
        # Re-normalize cache path in case it was overridden after initialization.
        self.cache_path = Path(self.cache_path)
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.max_explanation_paths <= 0 or self.max_path_length <= 0:
            raise ValueError("path limits must be positive")
        if self.min_likes_for_graph < 0 or self.min_films_for_graph < 0:
            raise ValueError("minimum thresholds must be non-negative")
        if self.temporal_decay_half_life_days <= 0:
            raise ValueError("temporal_decay_half_life_days must be positive")
        if not (0 < self.temporal_decay_min_weight <= 1):
            raise ValueError("temporal_decay_min_weight must be in (0,1]")
        if self.item_similarity_top_k <= 0:
            raise ValueError("item_similarity_top_k must be positive")
        if self.item_similarity_weight < 0:
            raise ValueError("item_similarity_weight must be non-negative")
        if self.follow_edge_weight < 0:
            raise ValueError("follow_edge_weight must be non-negative")

        if not (0.0 <= self.restart_user_weight < 1.0):
            raise ValueError("restart_user_weight must be in [0, 1)")
        if not (0.0 <= self.seed_restart_weight <= 1.0):
            raise ValueError("seed_restart_weight must be in [0, 1]")

        self._validate_weights("relation_weights", self.relation_weights, require_positive_sum=True)
        self._validate_weights("restart_weights", self.restart_weights, require_positive_sum=True)
        self._validate_weights("interaction_weights", self.interaction_weights, require_positive_sum=True)
        self._validate_weights("edge_type_weights", self.edge_type_weights, require_positive_sum=False)

    def _validate_weights(self, name: str, weights: dict[str, float], require_positive_sum: bool) -> None:
        if not isinstance(weights, dict):
            raise ValueError(f"{name} must be a dict of weights")
        if any(v < 0 for v in weights.values()):
            raise ValueError(f"{name} must be non-negative")
        if require_positive_sum and sum(weights.values()) <= 0:
            raise ValueError(f"{name} must contain at least one positive weight")

    @property
    def cache_key(self) -> str:
        return f"{self.cache_version}-{self._fingerprint()}"

    @property
    def cache_file_path(self) -> Path:
        base = Path(self.cache_path)
        return base.with_name(f"{base.stem}-{self.cache_key}{base.suffix}")

    def _fingerprint(self) -> str:
        payload = {
            "alpha": self.alpha,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "relation_preset": self.relation_preset,
            "relation_weights": self.relation_weights,
            "restart_weights": self.restart_weights,
            "restart_user_weight": self.restart_user_weight,
            "seed_restart_weight": self.seed_restart_weight,
            "interaction_weights": self.interaction_weights,
            "edge_type_weights": self.edge_type_weights,
            "use_type_budgets": self.use_type_budgets,
            "use_temporal_decay": self.use_temporal_decay,
            "temporal_decay_half_life_days": self.temporal_decay_half_life_days,
            "temporal_decay_min_weight": self.temporal_decay_min_weight,
            "use_item_similarity_edges": self.use_item_similarity_edges,
            "item_similarity_top_k": self.item_similarity_top_k,
            "item_similarity_weight": self.item_similarity_weight,
            "use_follow_edges": self.use_follow_edges,
            "follow_edge_weight": self.follow_edge_weight,
            "use_idf": self.use_idf,
            "idf_floor": self.idf_floor,
            "idf_ceiling": self.idf_ceiling,
            "max_explanation_paths": self.max_explanation_paths,
            "max_path_length": self.max_path_length,
            "min_likes_for_graph": self.min_likes_for_graph,
            "min_films_for_graph": self.min_films_for_graph,
        }
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]

    # Presets / external weights ---------------------------------------
    def _apply_relation_preset(self, preset: str) -> None:
        presets: dict[str, dict[str, float]] = {
            "balanced": self.relation_weights,
            "genre_heavy": {**self.relation_weights, "genre": 1.4, "theme": 0.8, "actor": 0.4, "director": 1.0},
            "people_heavy": {**self.relation_weights, "director": 1.6, "writer": 1.3, "actor": 0.8, "composer": 1.0},
            "social": {**self.relation_weights, "follows": 1.2, "user_bridge": 0.6},
            "recent": {**self.relation_weights, "watched": 0.4, "watchlisted": 0.2},
        }
        if preset in presets:
            self.relation_weights = presets[preset]

    def _maybe_load_relation_weights_file(self) -> None:
        if not self.relation_weights_path:
            return
        try:
            if not Path(self.relation_weights_path).exists():
                return
            payload: Any = json.loads(Path(self.relation_weights_path).read_text())
            if isinstance(payload, dict):
                self.relation_weights = {str(k): float(v) for k, v in payload.items()}
        except Exception:
            # Best-effort load; fall back to defaults on error.
            return

