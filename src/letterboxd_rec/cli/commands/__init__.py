"""CLI command modules."""

from .scrape import (
    cmd_scrape,
    cmd_scrape_social,
    cmd_backfill_social,
    cmd_scrape_daemon,
)
from .discover import (
    cmd_discover,
    cmd_discover_refill,
    cmd_discover_from_taste,
)
from .recommend import (
    cmd_recommend,
    cmd_similar_users,
    cmd_jam,
)
from .data import (
    cmd_stats,
    cmd_export,
    cmd_import,
    cmd_refresh_metadata,
)
from .queue import (
    cmd_queue_status,
    cmd_queue_add,
    cmd_queue_clear,
    cmd_prune_pending,
    cmd_session_history,
)
from .analysis import (
    cmd_explore,
    cmd_profile,
    cmd_similar,
    cmd_triage,
    cmd_svd_info,
    cmd_gaps,
    cmd_rebuild_idf,
)

__all__ = [
    # Scrape commands
    'cmd_scrape',
    'cmd_scrape_social',
    'cmd_backfill_social',
    'cmd_scrape_daemon',
    # Discover commands
    'cmd_discover',
    'cmd_discover_refill',
    'cmd_discover_from_taste',
    # Recommend commands
    'cmd_recommend',
    'cmd_similar_users',
    'cmd_jam',
    # Data commands
    'cmd_stats',
    'cmd_export',
    'cmd_import',
    'cmd_refresh_metadata',
    # Queue commands
    'cmd_queue_status',
    'cmd_queue_add',
    'cmd_queue_clear',
    'cmd_prune_pending',
    'cmd_session_history',
    # Analysis commands
    'cmd_explore',
    'cmd_profile',
    'cmd_similar',
    'cmd_triage',
    'cmd_svd_info',
    'cmd_gaps',
    'cmd_rebuild_idf',
]
