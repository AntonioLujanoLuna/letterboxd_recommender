"""Letterboxd Recommender CLI package."""

import argparse
import atexit
import logging

from ..database import (
    init_db,
    close_pool,
    get_db,
    get_pending_queue_stats,
    get_pending_users,
    add_pending_users,
    get_discovery_source,
    update_discovery_source,
)
from ..config import (
    DEFAULT_MAX_PER_BATCH,
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_MAX_CONCURRENT_USERS,
    DEFAULT_ASYNC_DELAY,
    DEFAULT_SCRAPER_DELAY,
    PENDING_STALE_DAYS,
    RUN_VACUUM_ANALYZE_DEFAULT,
    NOTIFICATION_WEBHOOK_URL,
    DISCOVERY_PRIORITY_MAP,
)
from ..scraper import LetterboxdScraper

from .commands import (
    cmd_scrape,
    cmd_scrape_social,
    cmd_backfill_social,
    cmd_scrape_daemon,
    cmd_discover,
    cmd_discover_refill,
    cmd_discover_from_taste,
    cmd_recommend,
    cmd_similar_users,
    cmd_jam,
    cmd_stats,
    cmd_export,
    cmd_import,
    cmd_refresh_metadata,
    cmd_queue_status,
    cmd_queue_add,
    cmd_queue_clear,
    cmd_prune_pending,
    cmd_session_history,
    cmd_explore,
    cmd_profile,
    cmd_similar,
    cmd_triage,
    cmd_svd_info,
    cmd_gaps,
    cmd_rebuild_idf,
)

# For backward compatibility - re-export utilities
from .utils import (
    send_notification,
    _require_slug,
    _validate_slug,
    _validate_username,
    _parse_weights,
    _load_all_user_films,
    _load_films_with_filters,
    _load_recommendation_data,
    _warn_missing_metadata,
    _scrape_film_metadata,
    _persist_metadata_batch,
    _scrape_film_metadata_async,
    _scrape_users_parallel,
    _run_metadata_strategy,
    _run_collaborative_strategy,
    _run_hybrid_strategy,
    _run_graph_strategy,
    _run_svd_strategy,
    _output_recommendations,
)

logger = logging.getLogger(__name__)

# Register cleanup on exit
atexit.register(close_pool)


def setup_jam_parser(subparsers):
    jam_parser = subparsers.add_parser(
        "jam",
        help="Generate recommendations for a group watch session",
        description="Find films that work for everyone in your watch party",
    )
    jam_parser.add_argument(
        "usernames",
        nargs="+",
        help="Letterboxd usernames (at least 2)",
    )
    jam_parser.add_argument(
        "--strategy",
        choices=[
            "least_misery",
            "most_pleasure",
            "average",
            "fairness",
            "approval",
            "multiplicative",
        ],
        default="fairness",
        help="How to combine preferences (default: fairness)",
    )
    jam_parser.add_argument("--limit", type=int, default=15, help="Number of recommendations")
    jam_parser.add_argument("--min-year", type=int, help="Minimum release year")
    jam_parser.add_argument("--max-year", type=int, help="Maximum release year")
    jam_parser.add_argument("--genres", nargs="+", help="Required genres")
    jam_parser.add_argument(
        "--include-divisive",
        action="store_true",
        help="Include films with divisive genres/directors",
    )
    jam_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    jam_parser.add_argument(
        "--weights",
        nargs="+",
        help="User weights as user:weight pairs (e.g., alex:2.0 for birthday person)",
    )
    jam_parser.add_argument(
        "--triage-watchlist",
        action="store_true",
        help="Rank films that appear on multiple watchlists (triage mode)",
    )
    jam_parser.set_defaults(func=cmd_jam)


def main():
    parser = argparse.ArgumentParser(description="Letterboxd Recommender")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Scrape user data")
    scrape_parser.add_argument("username", help="Letterboxd username")
    scrape_parser.add_argument("--refresh", type=int, metavar="DAYS",
                                help="Only scrape if last scraped more than N days ago")
    scrape_parser.add_argument("--include-lists", action="store_true", default=True,
                                help="Include user lists (favorites, ranked lists)")
    scrape_parser.add_argument("--no-include-lists", dest="include_lists", action="store_false",
                                help="Skip scraping user lists")
    scrape_parser.add_argument("--max-lists", type=int, default=50,
                                help="Maximum number of lists to scrape per user (default: 50)")
    scrape_parser.add_argument("--incremental", action="store_true",
                                help="Stop pagination when hitting already-scraped films (faster for updates)")
    scrape_parser.add_argument("--include-social", action="store_true",
                                help="Also scrape following/followers")
    scrape_parser.add_argument("--social-limit", type=int, default=200,
                                help="Max following/followers to scrape (default: 200)")
    scrape_parser.set_defaults(func=cmd_scrape)

    # Discover command
    discover_parser = subparsers.add_parser("discover", help="Discover and scrape other users")
    discover_parser.add_argument("source",
                                  nargs='?',
                                  choices=['following', 'followers', 'popular', 'film', 'film_reviews'],
                                  help="Source for user discovery (optional with --continue)")
    discover_parser.add_argument("--username", help="Username (for following/followers)")
    discover_parser.add_argument("--film-slug", help="Film slug (for film/film_reviews source, e.g., 'perfect-blue')")
    discover_parser.add_argument("--limit", type=int, default=50, help="Number of users to scrape")
    discover_parser.add_argument("--dry-run", action="store_true",
                                  help="Show which users would be scraped without actually scraping")
    discover_parser.add_argument("--continue", dest="continue_mode", action="store_true",
                                  help="Continue scraping from pending user queue without discovering new users")
    discover_parser.add_argument("--source-refresh-days", type=int, default=7,
                                  help="Days before re-crawling a source from page 1 (default: 7)")
    discover_parser.add_argument("--min-films", type=int, default=50,
                                  help="Minimum film count for activity pre-filtering (default: 50)")
    discover_parser.add_argument("--maintenance", action="store_true", default=False,
                                  help="Run VACUUM/ANALYZE after scraping batch")
    discover_parser.add_argument("--parallel-users", type=int, default=1,
                                  help="Scrape discovered users in parallel (async) with this many users at once")
    discover_parser.add_argument("--max-concurrent-requests", type=int, default=DEFAULT_MAX_CONCURRENT,
                                  help="Max concurrent HTTP requests when using --parallel-users")
    discover_parser.add_argument("--async-delay", type=float, default=DEFAULT_ASYNC_DELAY,
                                  help="Base async delay between requests when using --parallel-users")
    discover_parser.add_argument(
        "--discover-delay",
        type=float,
        default=DEFAULT_SCRAPER_DELAY,
        help="Delay between discovery HTTP requests (seconds); lower with care to go faster",
    )
    discover_parser.add_argument(
        "--discover-overfetch",
        type=int,
        default=3,
        help="Multiplier for discovery pagination to offset duplicates (default: 3)",
    )
    discover_parser.add_argument(
        "--skip-activity-checks",
        action="store_true",
        help="Skip per-user activity checks (much faster; may enqueue low-activity profiles)",
    )
    discover_parser.add_argument("--batch", type=int, default=DEFAULT_MAX_PER_BATCH,
                                  help="Batch size for film metadata fetches (parallel or serial)")
    discover_parser.add_argument("--queue-only", action="store_true",
                                 help="Only add discovered users to queue without scraping")
    discover_parser.set_defaults(func=cmd_discover)

    # Scrape social graph command
    social_parser = subparsers.add_parser("scrape-social", help="Scrape user's social graph (following/followers)")
    social_parser.add_argument("username", help="Letterboxd username")
    social_parser.add_argument("--following", action="store_true", help="Scrape who they follow")
    social_parser.add_argument("--followers", action="store_true", help="Scrape who follows them")
    social_parser.add_argument("--both", action="store_true", default=True, help="Scrape both (default)")
    social_parser.add_argument("--limit", type=int, default=500, help="Max relationships to scrape per direction")
    social_parser.set_defaults(func=cmd_scrape_social)

    # Backfill social graph command
    backfill_social_parser = subparsers.add_parser("backfill-social", help="Backfill social graph for existing users")
    backfill_social_parser.add_argument("--limit", type=int, default=100, help="Max users to process")
    backfill_social_parser.add_argument("--social-limit", type=int, default=200, help="Max following/followers per user")
    backfill_social_parser.add_argument("--dry-run", action="store_true", help="Show which users would be processed")
    backfill_social_parser.set_defaults(func=cmd_backfill_social)

    # Scrape daemon command
    daemon_parser = subparsers.add_parser("scrape-daemon", help="Continuously scrape from pending queue")
    daemon_parser.add_argument("--target", type=int, help="Stop after scraping N users")
    daemon_parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    daemon_parser.add_argument("--user-delay", type=float, default=0, help="Extra delay between users (seconds)")
    daemon_parser.add_argument("--wait-for-queue", action="store_true", help="Wait for new queue entries instead of exiting when empty")
    daemon_parser.add_argument("--wait-seconds", type=int, default=60, help="Wait duration when queue is empty and --wait-for-queue is set")
    daemon_parser.add_argument("--batch", type=int, default=DEFAULT_MAX_PER_BATCH, help="Batch size for metadata fetches")
    daemon_parser.add_argument("--remove-on-error", action="store_true", help="Remove user from queue on scrape error")
    daemon_parser.add_argument("--max-concurrent-users", type=int, default=DEFAULT_MAX_CONCURRENT_USERS,
                               help="Max users to scrape in parallel (async daemon)")
    daemon_parser.add_argument("--max-concurrent-requests", type=int, default=DEFAULT_MAX_CONCURRENT,
                               help="Max concurrent HTTP requests across tasks (async)")
    daemon_parser.add_argument("--async-delay", type=float, default=DEFAULT_ASYNC_DELAY,
                               help="Base async delay between requests (seconds)")
    daemon_parser.add_argument("--include-social", action="store_true", help="Also scrape following/followers for each user")
    daemon_parser.add_argument("--social-limit", type=int, default=200, help="Max following/followers to scrape per user")
    daemon_parser.set_defaults(func=cmd_scrape_daemon)

    # Recommend command
    rec_parser = subparsers.add_parser("recommend", help="Generate recommendations")
    rec_parser.add_argument("username", help="Letterboxd username")
    rec_parser.add_argument("--strategy", choices=['metadata', 'collaborative', 'hybrid', 'svd', 'graph'],
                            default='metadata', help="Recommendation strategy")
    rec_parser.add_argument("--limit", type=int, default=20, help="Number of recommendations")
    rec_parser.add_argument("--min-year", type=int, help="Minimum release year")
    rec_parser.add_argument("--max-year", type=int, help="Maximum release year")
    rec_parser.add_argument("--genres", nargs="+", help="Filter by genres")
    rec_parser.add_argument("--exclude-genres", nargs="+", help="Exclude genres")
    rec_parser.add_argument("--min-rating", type=float, help="Minimum community rating (metadata only)")
    rec_parser.add_argument("--diversity", action="store_true", help="Enable diversity mode (metadata only)")
    rec_parser.add_argument("--max-per-director", type=int, default=2, help="Max films per director (diversity mode)")
    rec_parser.add_argument("--no-temporal-decay", action="store_true",
                            help="Disable temporal decay (treat old and new ratings equally)")
    rec_parser.add_argument(
        "--weighting-mode",
        choices=["absolute", "normalized", "blended"],
        default="absolute",
        help="Scoring weights: absolute (default), normalized (per-user z-score), or blended",
    )
    rec_parser.add_argument("--hybrid-meta-weight", type=float, help="Weight for metadata component in hybrid fusion")
    rec_parser.add_argument("--hybrid-collab-weight", type=float, help="Weight for collaborative component in hybrid fusion")
    rec_parser.add_argument("--hybrid-diversity", action="store_true", help="Apply diversity constraint to hybrid output")
    rec_parser.add_argument("--graph-alpha", type=float, default=0.15,
                            help="PPR restart probability for graph strategy")
    rec_parser.add_argument("--like-weight", type=float, help="Restart weight for liked films (graph)")
    rec_parser.add_argument("--watch-weight", type=float, help="Restart weight for watched-only films (graph)")
    rec_parser.add_argument("--watchlist-weight", type=float, help="Restart weight for watchlisted films (graph)")
    rec_parser.add_argument(
        "--relation-preset",
        choices=["balanced", "genre_heavy", "people_heavy", "social", "recent"],
        help="Graph relation-weight preset"
    )
    rec_parser.add_argument(
        "--relation-weights-path",
        type=str,
        help="Path to JSON file with custom graph relation weights",
    )
    rec_parser.add_argument(
        "--no-graph-idf",
        action="store_true",
        help="Disable IDF weighting for graph strategy",
    )
    rec_parser.add_argument(
        "--graph-idf-floor",
        type=float,
        help="Minimum IDF weight for graph attributes",
    )
    rec_parser.add_argument(
        "--graph-idf-ceiling",
        type=float,
        help="Maximum IDF weight for graph attributes",
    )
    rec_parser.add_argument("--graph-cache", type=str, help="Override graph cache path")
    rec_parser.add_argument("--rebuild-graph", action="store_true", help="Force rebuild of graph cache")
    rec_parser.add_argument("--format", choices=['text', 'json', 'markdown', 'csv'], default='text',
                            help="Output format")
    rec_parser.add_argument("--explain", action="store_true",
                            help="Show detailed explanation for each recommendation")
    rec_parser.add_argument("--diversity-report", action="store_true",
                            help="Show diversity metrics for the recommendation set")
    rec_parser.set_defaults(func=cmd_recommend)

    # Explore command
    explore_parser = subparsers.add_parser("explore", help="Explore films by attribute")
    explore_parser.add_argument("attribute_type", choices=["director", "genre", "actor"],
                                 help="Type of attribute to explore")
    explore_parser.add_argument("value", help="Attribute value (e.g., 'Bong Joon-ho' for director)")
    explore_parser.add_argument("--limit", type=int, default=20,
                                 help="Number of films to show (default: 20)")
    explore_parser.set_defaults(func=cmd_explore)

    # Similar users command
    similar_users_parser = subparsers.add_parser("similar-users",
                                                   help="Find users with similar taste")
    similar_users_parser.add_argument("username", help="Your Letterboxd username")
    similar_users_parser.add_argument("--limit", type=int, default=10,
                                       help="Number of similar users to find")
    similar_users_parser.add_argument("--verbose", "-v", action="store_true",
                                       help="Show additional statistics")
    similar_users_parser.set_defaults(func=cmd_similar_users)

    # Group recommendation command
    setup_jam_parser(subparsers)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.set_defaults(func=cmd_stats)

    # SVD command
    svd_parser = subparsers.add_parser("svd-info", help="Show SVD model diagnostics")
    svd_parser.set_defaults(func=cmd_svd_info)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export database to JSON")
    export_parser.add_argument("file", help="Output JSON file path")
    export_parser.set_defaults(func=cmd_export)

    # Import command
    import_parser = subparsers.add_parser("import", help="Import database from JSON")
    import_parser.add_argument("file", help="Input JSON file path")
    import_parser.add_argument("--maintenance", action="store_true", default=RUN_VACUUM_ANALYZE_DEFAULT,
                               help="Run VACUUM/ANALYZE after import")
    import_parser.set_defaults(func=cmd_import)

    # Profile command
    profile_parser = subparsers.add_parser("profile", help="Show user's preference profile")
    profile_parser.add_argument("username", help="Letterboxd username")
    profile_parser.set_defaults(func=cmd_profile)

    # Similar command
    similar_parser = subparsers.add_parser("similar", help="Find films similar to a specific film")
    similar_parser.add_argument("slug", help="Film slug (e.g., 'the-matrix')")
    similar_parser.add_argument("--limit", type=int, default=10, help="Number of similar films")
    similar_parser.set_defaults(func=cmd_similar)

    # Triage command
    triage_parser = subparsers.add_parser("triage", help="Rank watchlist by predicted enjoyment")
    triage_parser.add_argument("username", help="Letterboxd username")
    triage_parser.add_argument("--limit", type=int, default=20, help="Number of films to show")
    triage_parser.set_defaults(func=cmd_triage)

    # Gaps command
    gaps_parser = subparsers.add_parser("gaps", help="Find essential missing films from favorite directors")
    gaps_parser.add_argument("username", help="Letterboxd username")
    gaps_parser.add_argument("--min-score", type=float, default=2.0, help="Minimum director affinity score")
    gaps_parser.add_argument("--limit", type=int, default=3, help="Max films per director")
    gaps_parser.add_argument("--min-year", type=int, help="Minimum release year")
    gaps_parser.add_argument("--max-year", type=int, help="Maximum release year")
    gaps_parser.set_defaults(func=cmd_gaps)

    # Rebuild-IDF command
    rebuild_idf_parser = subparsers.add_parser("rebuild-idf", help="Rebuild IDF scores for attribute rarity weighting")
    rebuild_idf_parser.set_defaults(func=cmd_rebuild_idf)

    # Refresh metadata command
    refresh_meta_parser = subparsers.add_parser("refresh-metadata", help="Refresh missing/low-quality film metadata")
    refresh_meta_parser.add_argument("--limit", type=int, default=200, help="Max films to refresh")
    refresh_meta_parser.add_argument("--min-rating-count", type=int, default=50, help="Minimum rating count threshold to consider metadata low-quality")
    refresh_meta_parser.add_argument("--batch", type=int, default=DEFAULT_MAX_PER_BATCH, help="Batch size for metadata scraping")
    refresh_meta_parser.add_argument("--maintenance", action="store_true", default=False, help="Run ANALYZE after refresh")
    refresh_meta_parser.set_defaults(func=cmd_refresh_metadata)

    # Prune pending command
    prune_pending_parser = subparsers.add_parser("prune-pending", help="Prune stale pending users")
    prune_pending_parser.add_argument("--older-than", type=int, default=PENDING_STALE_DAYS, help="Age in days to prune pending entries")
    prune_pending_parser.add_argument("--max-priority", type=int, help="Remove pending entries at or below this priority")
    prune_pending_parser.set_defaults(func=cmd_prune_pending)

    # Queue management commands
    queue_status_parser = subparsers.add_parser("queue-status", help="Show pending queue status")
    queue_status_parser.add_argument("--verbose", action="store_true", help="Show next pending users")
    queue_status_parser.add_argument("--limit", type=int, default=10, help="Number of pending users to show with --verbose")
    queue_status_parser.set_defaults(func=cmd_queue_status)

    queue_add_parser = subparsers.add_parser("queue-add", help="Add usernames to queue")
    queue_add_parser.add_argument("usernames", nargs="*", help="Usernames to add")
    queue_add_parser.add_argument("--file", "-f", help="File with usernames (one per line)")
    queue_add_parser.add_argument("--priority", type=int, default=50, help="Priority for new queue entries")
    queue_add_parser.set_defaults(func=cmd_queue_add)

    queue_clear_parser = subparsers.add_parser("queue-clear", help="Clear pending queue")
    queue_clear_parser.add_argument("--source", help="Only clear users discovered from this source type")
    queue_clear_parser.set_defaults(func=cmd_queue_clear)

    # Discovery helpers
    refill_parser = subparsers.add_parser("discover-refill", help="Auto-refill queue from configured sources")
    refill_parser.add_argument("--min-queue", type=int, default=50, help="Only refill if queue size is below this")
    refill_parser.add_argument("--target", type=int, default=200, help="Target queue size after refill")
    refill_parser.add_argument("--source-refresh-days", type=int, default=7, help="Minimum age before reusing a discovery source")
    refill_parser.add_argument("--sources-file", help="JSON file with source definitions")
    refill_parser.add_argument("--min-films", type=int, default=50, help="Minimum films for activity filter")
    refill_parser.set_defaults(func=cmd_discover_refill)

    taste_parser = subparsers.add_parser("discover-taste", help="Discover users from your top films")
    taste_parser.add_argument("username", help="Your Letterboxd username")
    taste_parser.add_argument("--min-rating", type=float, default=4.0, help="Minimum rating to consider a film a favorite")
    taste_parser.add_argument("--film-limit", type=int, default=20, help="Number of top films to use")
    taste_parser.add_argument("--per-film", type=int, default=25, help="Reviewers per film to consider")
    taste_parser.add_argument("--min-films", type=int, default=50, help="Minimum film count for candidate users")
    taste_parser.set_defaults(func=cmd_discover_from_taste)

    # Session history
    session_parser = subparsers.add_parser("session-history", help="Show scraping session history")
    session_parser.add_argument("--limit", type=int, default=10, help="Number of sessions to display")
    session_parser.set_defaults(func=cmd_session_history)

    args = parser.parse_args()

    # Configure logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    args.func(args)


__all__ = [
    'main',
    'setup_jam_parser',
    # Database re-exports for backward compatibility
    'init_db',
    'get_db',
    'get_pending_queue_stats',
    'get_pending_users',
    'add_pending_users',
    'get_discovery_source',
    'update_discovery_source',
    # Config re-exports for backward compatibility
    'NOTIFICATION_WEBHOOK_URL',
    'DISCOVERY_PRIORITY_MAP',
    # Scraper re-exports for backward compatibility
    'LetterboxdScraper',
    # Commands
    'cmd_scrape',
    'cmd_scrape_social',
    'cmd_backfill_social',
    'cmd_scrape_daemon',
    'cmd_discover',
    'cmd_discover_refill',
    'cmd_discover_from_taste',
    'cmd_recommend',
    'cmd_similar_users',
    'cmd_jam',
    'cmd_stats',
    'cmd_export',
    'cmd_import',
    'cmd_refresh_metadata',
    'cmd_queue_status',
    'cmd_queue_add',
    'cmd_queue_clear',
    'cmd_prune_pending',
    'cmd_session_history',
    'cmd_explore',
    'cmd_profile',
    'cmd_similar',
    'cmd_triage',
    'cmd_svd_info',
    'cmd_gaps',
    'cmd_rebuild_idf',
    # Utilities (backward compat)
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
