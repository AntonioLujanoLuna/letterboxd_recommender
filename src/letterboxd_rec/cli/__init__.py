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
from ..lbpy_backend import BACKENDS, make_async_scraper, make_scraper, set_backend
from ..scrape_stats import STATS

from .commands.scrape import _cmd_scrape_daemon_async
from .utils.helpers import film_url
from .utils.strategies import _run_movielens_strategy, _run_ease_strategy, _run_content_strategy
from .commands.discover import _passes_activity, _filter_active_users, _enqueue_from_graph
from .commands import (
    cmd_scrape,
    cmd_scrape_social,
    cmd_backfill_social,
    cmd_scrape_daemon,
    cmd_backup,
    cmd_supervise,
    cmd_doctor,
    cmd_discover,
    cmd_discover_refill,
    cmd_discover_from_taste,
    cmd_crawl,
    cmd_recommend,
    cmd_evaluate,
    cmd_movielens_build,
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
    parser.add_argument("--backend", choices=BACKENDS, default=None,
                        help="Scraping backend for user/film pages (default: $LETTERBOXD_BACKEND or native). "
                             "'letterboxdpy' uses browser-impersonating requests and can bypass 403 blocks.")
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
    discover_parser.add_argument("--max-rps", type=float, default=None,
                                  help="Cap on requests started per second (default: $LETTERBOXD_MAX_RPS or 6)")
    discover_parser.add_argument("--auto-rps", type=float, default=None, metavar="CEILING",
                                  help="Probe upward from --max-rps toward CEILING req/s, halving on 429/403")
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
    daemon_parser.add_argument("--max-rps", type=float, default=None,
                               help="Cap on requests started per second (default: $LETTERBOXD_MAX_RPS or 6)")
    daemon_parser.add_argument("--auto-rps", type=float, default=None, metavar="CEILING",
                               help="Probe upward from --max-rps toward CEILING req/s, halving on 429/403")
    daemon_parser.add_argument("--claim-ttl-minutes", type=float, default=30,
                               help="On start, re-queue users left claimed longer than this by a crashed run (0 = all)")
    daemon_parser.add_argument("--include-social", action="store_true", help="Also scrape following/followers for each user")
    daemon_parser.add_argument("--social-limit", type=int, default=200, help="Max following/followers to scrape per user")
    daemon_parser.set_defaults(func=cmd_scrape_daemon)

    # Recommend command
    rec_parser = subparsers.add_parser("recommend", help="Generate recommendations")
    rec_parser.add_argument("username", help="Letterboxd username")
    rec_parser.add_argument("--novelty", type=float, default=0.0, metavar="0-1",
                            help="ease/movielens/blend: favor less mainstream films (0 = off; 0.2 is mild, 0.5 strong; "
                                 "costs some accuracy, see README)")
    rec_parser.add_argument("--strategy", choices=['metadata', 'collaborative', 'svd', 'graph', 'ease', 'content', 'movielens', 'blend'],
                            default=None,
                            help="Recommendation strategy (default: blend when the MovieLens model is built, "
                                 "else ease when the database has enough users, else metadata)")
    rec_parser.add_argument("--limit", type=int, default=20, help="Number of recommendations")
    rec_parser.add_argument("--min-year", type=int, help="Minimum release year")
    rec_parser.add_argument("--max-year", type=int, help="Maximum release year")
    rec_parser.add_argument("--genres", nargs="+", help="Filter by genres")
    rec_parser.add_argument("--exclude-genres", nargs="+", help="Exclude genres")
    rec_parser.add_argument("--min-rating", type=float, help="Minimum community rating")
    rec_parser.add_argument("--min-runtime", type=int, metavar="MIN", help="Minimum runtime in minutes")
    rec_parser.add_argument("--max-runtime", type=int, metavar="MIN", help="Maximum runtime in minutes")
    rec_parser.add_argument("--exclude-shorts", action="store_true", help="Leave out short films")
    rec_parser.add_argument("--exclude-animation", action="store_true", help="Leave out animated films")
    rec_parser.add_argument("--countries", nargs="+", help="Only films from these countries (e.g. Japan 'South Korea')")
    rec_parser.add_argument("--languages", nargs="+", help="Only films in these languages (e.g. Japanese French)")
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

    # Operations: backups, supervisor, health check
    backup_parser = subparsers.add_parser("backup", help="Back up the database (safe while a crawl is running)")
    backup_parser.add_argument("--dir", default="data/backups", help="Backup directory (default: data/backups)")
    backup_parser.add_argument("--keep", type=int, default=5, help="Backups to keep (default: 5)")
    backup_parser.set_defaults(func=cmd_backup)

    supervise_parser = subparsers.add_parser(
        "supervise",
        help="Run a command and restart it if it crashes, with periodic database backups",
        description="Example: python main.py supervise -- --backend letterboxdpy crawl --users 10000 --max-rps 2",
    )
    supervise_parser.add_argument("--backup-every", type=float, default=60.0, metavar="MIN",
                                  help="Minutes between backups; 0 disables (default: 60)")
    supervise_parser.add_argument("--backup-dir", default="data/backups", help="Backup directory")
    supervise_parser.add_argument("--keep", type=int, default=5, help="Backups to keep (default: 5)")
    supervise_parser.add_argument("--max-failures", type=int, default=20,
                                  help="Give up after this many crashes in a row (default: 20)")
    supervise_parser.add_argument("child", nargs=argparse.REMAINDER, help="-- then the command to run")
    supervise_parser.set_defaults(func=cmd_supervise)

    doctor_parser = subparsers.add_parser("doctor", help="Check the database for missing, orphaned or stuck data")
    doctor_parser.set_defaults(func=cmd_doctor)

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

    crawl_parser = subparsers.add_parser(
        "crawl", help="Unattended crawl: seed queue, then alternate scraping and refilling")
    crawl_parser.add_argument("--users", type=int, default=200, help="Stop after scraping this many new users")
    crawl_parser.add_argument("--seed-film", action="append", metavar="SLUG",
                              help="Seed from reviewers of this film (repeatable)")
    crawl_parser.add_argument("--seed-user", action="append", metavar="USER",
                              help="Seed from this user's followers (repeatable)")
    crawl_parser.add_argument("--popular", action="store_true", help="Also seed from popular members")
    crawl_parser.add_argument("--sources-file", help="JSON sources file (see discover-refill); used when no seeds given")
    crawl_parser.add_argument("--min-films", type=int, default=50, help="Minimum films for activity filter")
    crawl_parser.add_argument("--refill-below", type=int, default=50, help="Refill when queue drops below this")
    crawl_parser.add_argument("--queue-target", type=int, default=200, help="Queue size to refill up to")
    crawl_parser.add_argument("--source-refresh-days", type=int, default=7)
    crawl_parser.add_argument("--no-expand-graph", dest="expand_graph", action="store_false",
                              help="Don't fall back to unscraped users from the social graph")
    crawl_parser.add_argument("--chunk", type=int, default=50, help="Users to scrape between queue checks")
    crawl_parser.add_argument("--batch", type=int, default=DEFAULT_MAX_PER_BATCH, help="Film metadata batch size")
    crawl_parser.add_argument("--max-concurrent-users", type=int, default=DEFAULT_MAX_CONCURRENT_USERS)
    crawl_parser.add_argument("--max-concurrent-requests", type=int, default=DEFAULT_MAX_CONCURRENT)
    crawl_parser.add_argument("--async-delay", type=float, default=DEFAULT_ASYNC_DELAY)
    crawl_parser.add_argument("--max-rps", type=float, default=None,
                              help="Cap on requests started per second (default: $LETTERBOXD_MAX_RPS or 6)")
    crawl_parser.add_argument("--auto-rps", type=float, default=None, metavar="CEILING",
                              help="Probe upward from --max-rps toward CEILING req/s, halving on 429/403")
    crawl_parser.add_argument("--user-delay", type=float, default=0, help="Extra delay between users (seconds)")
    crawl_parser.add_argument("--no-social", dest="include_social", action="store_false",
                              help="Skip following/followers per user (default is to scrape them; feeds graph expansion)")
    crawl_parser.set_defaults(include_social=True, expand_graph=True)
    crawl_parser.add_argument("--social-limit", type=int, default=200)
    crawl_parser.add_argument("--claim-ttl-minutes", type=float, default=30,
                              help="Re-queue users left claimed longer than this by a crashed run (0 = all)")
    crawl_parser.add_argument("--max-stalls", type=int, default=3,
                              help="Abort after this many consecutive chunks that scraped nobody")
    crawl_parser.set_defaults(func=cmd_crawl)

    # Offline evaluation command
    eval_parser = subparsers.add_parser(
        "evaluate", help="Measure recommendation quality by hiding some of each user's liked films"
    )
    eval_parser.add_argument(
        "--strategies", nargs="+",
        choices=["popularity", "svd", "collaborative", "metadata", "ease", "content", "movielens", "blend"],
        help="Strategies to compare (default: all; the popularity baseline is always included)",
    )
    eval_parser.add_argument("--k", type=int, default=20, help="Cutoff for recall@k / NDCG@k")
    eval_parser.add_argument("--holdout", type=float, default=0.2,
                             help="Fraction of each user's positives to hide (default 0.2)")
    eval_parser.add_argument("--min-positives", type=int, default=10,
                             help="Only evaluate users with at least this many positives")
    eval_parser.add_argument("--max-users", type=int, default=200,
                             help="Sample at most this many users (metadata scoring is slow)")
    eval_parser.add_argument("--users", nargs="+", help="Evaluate exactly these users")
    eval_parser.add_argument("--seed", type=int, default=42, help="Random seed for the hold-out split")
    eval_parser.add_argument("--json", action="store_true", help="Print results as JSON")
    eval_parser.add_argument("--per-user", action="store_true", help="Include per-user metrics in JSON output")
    eval_parser.set_defaults(func=cmd_evaluate)

    ml_parser = subparsers.add_parser(
        "movielens-build", help="Train the MovieLens EASE model and report its held-out quality"
    )
    ml_parser.add_argument("--download", action="store_true", help="Download ML-32M if missing (~240 MB)")
    ml_parser.add_argument("--dir", help="Unzipped MovieLens directory (default data/movielens/ml-32m)")
    ml_parser.add_argument("--max-items", type=int, default=10_000,
                           help="Most-rated films to model (memory grows with the square; 10k ~ 3 GB peak)")
    ml_parser.add_argument("--lambdas", type=float, nargs="+", default=[200.0, 500.0, 1000.0, 3000.0],
                           help="EASE regularization values to compare; the best is saved")
    ml_parser.add_argument("--eval-users", type=int, default=2000, help="Held-out MovieLens users to score")
    ml_parser.add_argument("--k", type=int, default=20)
    ml_parser.add_argument("--seed", type=int, default=42)
    ml_parser.set_defaults(func=cmd_movielens_build)

    # Session history
    session_parser = subparsers.add_parser("session-history", help="Show scraping session history")
    session_parser.add_argument("--limit", type=int, default=10, help="Number of sessions to display")
    session_parser.set_defaults(func=cmd_session_history)

    args = parser.parse_args()
    if args.backend:
        set_backend(args.backend)

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
    'cmd_backup',
    'cmd_supervise',
    'cmd_doctor',
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
    '_run_graph_strategy',
    '_run_svd_strategy',
    '_output_recommendations',
    '_cmd_scrape_daemon_async',
    '_passes_activity',
    '_filter_active_users',
    '_enqueue_from_graph',
]
