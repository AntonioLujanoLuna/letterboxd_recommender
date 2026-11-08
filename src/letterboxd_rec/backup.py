"""Consistent SQLite backups that are safe while the crawler is writing."""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from .config import DB_PATH

logger = logging.getLogger(__name__)

DEFAULT_BACKUP_DIR = DB_PATH.parent / "backups"
DEFAULT_KEEP = 5
_PREFIX = "letterboxd-"
_SUFFIX = ".db"


def backup_database(
    dest_dir: Path = DEFAULT_BACKUP_DIR,
    keep: int = DEFAULT_KEEP,
    db_path: Path = DB_PATH,
    now: datetime | None = None,
) -> Path:
    """Copy the database into `dest_dir` and prune to the newest `keep` backups.

    Uses SQLite's online backup API, which takes a consistent snapshot even while another process
    writes (a plain file copy of a WAL database can miss committed rows or capture a torn state).
    The copy is written to a temporary name and renamed, so a crash never leaves a half-written
    file that looks like a good backup. Returns the path of the new backup.
    """
    if keep < 1:
        raise ValueError("keep must be at least 1")
    if not Path(db_path).exists():
        raise FileNotFoundError(f"No database at {db_path}")

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    final = dest_dir / f"{_PREFIX}{stamp}{_SUFFIX}"
    partial = final.with_suffix(".partial")

    src = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True, timeout=60)
    try:
        dst = sqlite3.connect(partial)
        try:
            src.backup(dst)
            check = dst.execute("PRAGMA quick_check").fetchone()[0]
        finally:
            dst.close()
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    finally:
        src.close()

    if check != "ok":
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Backup failed its integrity check: {check}")
    partial.replace(final)

    for old in list_backups(dest_dir)[keep:]:
        old.unlink(missing_ok=True)
    logger.info(f"Backed up database to {final} ({final.stat().st_size / 1e6:.1f} MB)")
    return final


def list_backups(dest_dir: Path = DEFAULT_BACKUP_DIR) -> list[Path]:
    """Backups in `dest_dir`, newest first."""
    dest_dir = Path(dest_dir)
    if not dest_dir.exists():
        return []
    return sorted(dest_dir.glob(f"{_PREFIX}*{_SUFFIX}"), reverse=True)
