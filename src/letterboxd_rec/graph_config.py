from dataclasses import dataclass, field
from pathlib import Path


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

    # IDF weighting toggle
    use_idf: bool = True
    idf_floor: float = 0.1
    idf_ceiling: float = 5.0

    # Cache settings
    cache_enabled: bool = True
    cache_path: Path = Path("data/graph_cache.npz")

    # Explanation settings
    max_explanation_paths: int = 3
    max_path_length: int = 4

    # Guardrails for when to fall back to other strategies
    min_likes_for_graph: int = 3
    min_films_for_graph: int = 20

