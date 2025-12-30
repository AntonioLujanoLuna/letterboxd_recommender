"""Database package - provides connection pooling, schema management, and data access."""

from .connection import (
    RetryConnection,
    ConnectionPool,
    parse_timestamp_naive,
    get_db,
    close_pool,
    cleanup_connection_pool,
    _execute_with_retry,
    _is_lock_error,
)
from .schema import (
    init_db,
    run_versioned_migrations,
    populate_normalized_tables_batch,
)
from .loaders import (
    load_json,
    load_user_lists,
    load_user_films_batch,
    load_films_by_attribute,
)
from .profile_cache import (
    load_cached_profile,
    save_user_profile,
    purge_stale_profile_caches,
)
from .discovery import (
    get_discovery_source,
    update_discovery_source,
    add_pending_users,
    get_pending_users,
    remove_pending_user,
    get_pending_queue_stats,
)
from .sessions import (
    create_scrape_session,
    update_session_progress,
    complete_session,
    get_session_history,
)
from .idf import (
    compute_and_store_idf,
    load_idf,
    update_idf_incremental,
)
from .social import (
    save_user_follows,
    save_user_followers,
    get_social_graph_stats,
)
from .maintenance import run_maintenance

__all__ = [
    # Connection
    'RetryConnection',
    'ConnectionPool',
    'parse_timestamp_naive',
    'get_db',
    'close_pool',
    'cleanup_connection_pool',
    '_execute_with_retry',
    '_is_lock_error',
    # Schema
    'init_db',
    'run_versioned_migrations',
    'populate_normalized_tables_batch',
    # Loaders
    'load_json',
    'load_user_lists',
    'load_user_films_batch',
    'load_films_by_attribute',
    # Profile cache
    'load_cached_profile',
    'save_user_profile',
    'purge_stale_profile_caches',
    # Discovery
    'get_discovery_source',
    'update_discovery_source',
    'add_pending_users',
    'get_pending_users',
    'remove_pending_user',
    'get_pending_queue_stats',
    # Sessions
    'create_scrape_session',
    'update_session_progress',
    'complete_session',
    'get_session_history',
    # IDF
    'compute_and_store_idf',
    'load_idf',
    'update_idf_incremental',
    # Social
    'save_user_follows',
    'save_user_followers',
    'get_social_graph_stats',
    # Maintenance
    'run_maintenance',
]
