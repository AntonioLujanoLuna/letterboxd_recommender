"""CLI utility modules."""

from .helpers import (
    send_notification,
    _require_slug,
    _validate_slug,
    _validate_username,
    _parse_weights,
)
from .loaders import (
    _load_all_user_films,
    _load_films_with_filters,
    _load_recommendation_data,
    _warn_missing_metadata,
)
from .scrapers import (
    _scrape_film_metadata,
    _persist_metadata_batch,
    _scrape_film_metadata_async,
    _scrape_users_parallel,
)
from .strategies import (
    _run_metadata_strategy,
    _run_collaborative_strategy,
    _run_hybrid_strategy,
    _run_graph_strategy,
    _run_svd_strategy,
)
from .output import _output_recommendations

__all__ = [
    'send_notification',
    '_require_slug',
    '_validate_slug',
    '_validate_username',
    '_parse_weights',
    '_load_all_user_films',
    '_load_films_with_filters',
    '_load_recommendation_data',
    '_warn_missing_metadata',
    '_scrape_film_metadata',
    '_persist_metadata_batch',
    '_scrape_film_metadata_async',
    '_scrape_users_parallel',
    '_run_metadata_strategy',
    '_run_collaborative_strategy',
    '_run_hybrid_strategy',
    '_run_graph_strategy',
    '_run_svd_strategy',
    '_output_recommendations',
]
