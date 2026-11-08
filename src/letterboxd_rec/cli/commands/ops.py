"""Operations commands: database backups and the crawl supervisor."""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def cmd_backup(args: argparse.Namespace) -> None:
    """Take a consistent backup of the database (safe while a crawl is writing)."""
    from ...backup import backup_database, list_backups

    try:
        path = backup_database(Path(args.dir), keep=args.keep)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(str(e))
        return
    logger.info(f"Backups kept in {args.dir}: {len(list_backups(Path(args.dir)))} (newest first)")
    for p in list_backups(Path(args.dir)):
        logger.info(f"  {p.name}  {p.stat().st_size / 1e6:.1f} MB{'  <- new' if p == path else ''}")


def cmd_supervise(args: argparse.Namespace) -> None:
    """Run another command, restarting it if it crashes, with periodic database backups.

    Example:  python main.py supervise -- --backend letterboxdpy crawl --users 10000 --max-rps 2
    """
    from ...backup import backup_database
    from ...supervisor import AlreadyRunning, acquire_lock, release_lock, run_supervised

    child_args = list(args.child)
    if child_args and child_args[0] == "--":
        child_args = child_args[1:]
    if not child_args:
        logger.error("Nothing to run. Usage: python main.py supervise [options] -- <command and arguments>")
        sys.exit(2)

    try:
        acquire_lock()
    except AlreadyRunning as e:
        logger.error(str(e))
        sys.exit(1)

    backup_dir = Path(args.backup_dir)
    backup_fn = None
    if args.backup_every > 0:
        backup_fn = lambda: backup_database(backup_dir, keep=args.keep)  # noqa: E731
    cmd = [sys.executable, str(Path(sys.argv[0]).resolve()), *child_args]
    try:
        code = run_supervised(cmd, backup_fn=backup_fn, backup_every=args.backup_every * 60.0,
                              max_failures=args.max_failures)
    finally:
        release_lock()
    sys.exit(code)


def cmd_doctor(args: argparse.Namespace) -> None:
    """Report missing, orphaned or stuck data. Read-only."""
    from ... import cli
    from ...doctor import ERROR, format_report, run_checks, worst_level

    cli.init_db()
    with cli.get_db(read_only=True) as conn:
        findings = run_checks(conn)
    logger.info("\nDatabase health\n" + "-" * 30 + "\n" + format_report(findings))
    if worst_level(findings) == ERROR:
        sys.exit(1)


def cmd_diary(args: argparse.Namespace) -> None:
    """Backfill watch dates from users' diaries (opt-in; the main crawl does not fetch them)."""
    from ... import cli
    from ...diary import save_diary, scrape_diary, users_needing_diary
    from ...lbpy_backend import CurlClient

    cli.init_db()
    users = [u.strip().lower() for u in (args.usernames or []) if u.strip()]
    if args.crawled:
        with cli.get_db(read_only=True) as conn:
            users += [u for u in users_needing_diary(conn, args.crawled) if u not in users]
    if not users:
        logger.error("Name users to fetch, or use --crawled N for the N biggest crawled users without a diary yet.")
        sys.exit(2)

    client = CurlClient()
    done = skipped = total = 0
    for i, user in enumerate(users, 1):
        entries = scrape_diary(client, user, delay=args.delay)
        if entries is None:
            skipped += 1
            continue
        with cli.get_db() as conn:
            save_diary(conn, user, entries)
        done += 1
        total += len(entries)
        logger.info(f"[{i}/{len(users)}] {user}: {len(entries)} dated viewings")
    logger.info(f"Diaries stored for {done} users ({total} viewings); {skipped} skipped (private or blocked).")
