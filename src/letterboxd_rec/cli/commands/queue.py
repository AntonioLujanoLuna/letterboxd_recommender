"""Queue management CLI commands."""

import argparse
import logging
from pathlib import Path

from ...database import (
    init_db,
    get_db,
    parse_timestamp_naive,
    add_pending_users,
    get_pending_users,
    get_pending_queue_stats,
    get_session_history,
)
from ...config import PENDING_STALE_DAYS
from ..utils.helpers import _validate_username

logger = logging.getLogger(__name__)


def cmd_queue_status(args: argparse.Namespace) -> None:
    """Show pending queue statistics."""
    init_db()
    stats = get_pending_queue_stats()

    logger.info("\nPending Queue Status")
    logger.info("-" * 30)
    logger.info(f"Total pending users: {stats['total']}")
    if stats['avg_priority'] is not None:
        logger.info(f"Average priority: {stats['avg_priority']:.1f}")

    if stats['breakdown']:
        logger.info("\nBy source:")
        for source_type, count in sorted(stats['breakdown'].items(), key=lambda x: -x[1]):
            logger.info(f"  {source_type}: {count}")

    if stats['total'] > 0:
        est_hours = (stats['total'] * 30) / 3600  # rough estimate: ~30s/user
        logger.info(f"\nEstimated time to drain: {est_hours:.1f} hours")

    if args.verbose:
        pending = get_pending_users(limit=args.limit)
        if pending:
            logger.info("\nNext in queue:")
            for p in pending:
                logger.info(f"  [{p['priority']}] {p['username']} (from {p['discovered_from_type']})")


def cmd_queue_add(args: argparse.Namespace) -> None:
    """Manually add usernames to the pending queue."""
    init_db()

    usernames: list[str] = []
    if args.file:
        path = Path(args.file)
        if not path.exists():
            logger.error(f"File not found: {args.file}")
            return
        usernames = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    else:
        usernames = args.usernames or []

    sanitized = [_validate_username(u) for u in usernames if u]
    if not sanitized:
        logger.error("No usernames provided.")
        return

    added = add_pending_users(sanitized, "manual", "cli", priority=args.priority)
    skipped = len(sanitized) - added
    logger.info(f"Added {added} users to queue ({skipped} already existed)")


def cmd_queue_clear(args: argparse.Namespace) -> None:
    """Clear pending queue (optionally by source type)."""
    init_db()
    with get_db() as conn:
        if args.source:
            count = conn.execute(
                "DELETE FROM pending_users WHERE discovered_from_type = ?",
                (args.source,),
            ).rowcount
        else:
            count = conn.execute("DELETE FROM pending_users").rowcount
    logger.info(f"Removed {count} users from queue")


def cmd_prune_pending(args: argparse.Namespace) -> None:
    """Prune stale or low-priority pending users."""
    init_db()
    cutoff = f"-{args.older_than} days"
    removed = 0
    with get_db() as conn:
        removed += conn.execute(
            "DELETE FROM pending_users WHERE discovered_at < datetime('now', ?)",
            (cutoff,),
        ).rowcount
        if args.max_priority is not None:
            removed += conn.execute(
                "DELETE FROM pending_users WHERE priority <= ?",
                (args.max_priority,),
            ).rowcount
    logger.info(f"Pruned {removed} pending users")


def cmd_session_history(args: argparse.Namespace) -> None:
    """Show scraping session history."""
    init_db()
    sessions = get_session_history(limit=args.limit)
    if not sessions:
        logger.info("No scraping sessions recorded yet.")
        return

    logger.info("\nRecent Scraping Sessions")
    logger.info("-" * 44)
    for s in sessions:
        started = s['started_at'][:16].replace('T', ' ')
        status = s['status']
        users = s.get('users_scraped') or 0
        films = s.get('films_added') or 0

        if s.get('completed_at'):
            start_dt = parse_timestamp_naive(s['started_at'])
            end_dt = parse_timestamp_naive(s['completed_at'])
            duration = (end_dt - start_dt).total_seconds() / 3600
            duration_str = f"{duration:.1f}h"
        else:
            duration_str = "ongoing"

        logger.info(f"  [{s['id']}] {started} | {status:10} | {users:4} users | {films:5} films | {duration_str}")
