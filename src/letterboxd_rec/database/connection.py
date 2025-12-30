"""Database connection management with pooling and retry logic."""

import sqlite3
import logging
import threading
import time
import random
from typing import Any, Callable
from contextlib import contextmanager

from ..config import DB_PATH

logger = logging.getLogger(__name__)

_LOCK_RETRY_ATTEMPTS = 5
_LOCK_RETRY_BASE_DELAY = 0.1
_LOCK_RETRY_MAX_DELAY = 1.0


def _is_lock_error(exc: sqlite3.OperationalError) -> bool:
    """Check if the OperationalError corresponds to a SQLite lock/busy."""
    msg = str(exc).lower()
    return "database is locked" in msg or "database is busy" in msg


def _execute_with_retry(fn: Callable[[], Any], *, attempts: int = _LOCK_RETRY_ATTEMPTS) -> Any:
    """
    Run a DB operation with small exponential backoff on SQLite lock.

    Retries only for OperationalError that indicates the database is locked/busy.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except sqlite3.OperationalError as exc:  # pragma: no cover - timing dependent
            if not _is_lock_error(exc):
                raise
            if attempt == attempts - 1:
                raise

            delay = min(
                _LOCK_RETRY_MAX_DELAY,
                _LOCK_RETRY_BASE_DELAY * (2**attempt),
            ) + random.uniform(0, _LOCK_RETRY_BASE_DELAY)
            logger.warning(
                f"SQLite locked; retrying in {delay:.2f}s "
                f"(attempt {attempt + 1}/{attempts})"
            )
            time.sleep(delay)


class RetryConnection(sqlite3.Connection):
    """SQLite connection that retries on database-locked errors."""

    def execute(self, sql, parameters=(), /):
        return _execute_with_retry(lambda: super(RetryConnection, self).execute(sql, parameters))

    def executemany(self, sql, seq_of_parameters, /):
        return _execute_with_retry(lambda: super(RetryConnection, self).executemany(sql, seq_of_parameters))

    def executescript(self, sql_script, /):
        return _execute_with_retry(lambda: super(RetryConnection, self).executescript(sql_script))

    def commit(self):
        return _execute_with_retry(lambda: super(RetryConnection, self).commit())


from datetime import datetime


def parse_timestamp_naive(timestamp_str: str) -> datetime:
    """
    Parse ISO format timestamp string to naive datetime.

    Ensures consistency by always returning naive datetime regardless of
    whether the stored timestamp had timezone info.
    This prevents timezone comparison bugs when mixing naive and aware datetimes.
    """
    dt = datetime.fromisoformat(timestamp_str)
    # Always return naive datetime for consistency
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


class ConnectionPool:
    """
    Thread-safe SQLite connection pool with health checks and automatic cleanup.

    Features:
    - One connection per thread (SQLite threading requirement)
    - Periodic health checks via SELECT 1
    - Automatic cleanup of dead thread connections
    - Explicit transaction nesting tracking
    """

    def __init__(self, db_path, max_size: int = 50, health_check_interval: int = 300):
        self._db_path = db_path
        self._max_size = max_size
        self._health_check_interval = health_check_interval

        self._lock = threading.Lock()
        self._connections: dict[int, sqlite3.Connection] = {}
        self._last_used: dict[int, float] = {}
        self._last_health_check: dict[int, float] = {}
        self._transaction_depth: dict[int, int] = {}  # Track nested transactions
        self._last_cleanup = time.time()
        self._cleanup_interval = 60  # Cleanup every 60 seconds max

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with optimal settings."""
        conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            factory=RetryConnection,
        )
        conn.row_factory = sqlite3.Row

        # Performance optimizations
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")  # Better concurrency
        conn.execute("PRAGMA synchronous = NORMAL")  # Faster, still safe with WAL
        conn.execute("PRAGMA cache_size = -64000")  # 64MB cache

        return conn

    def _health_check(self, conn: sqlite3.Connection) -> bool:
        """Verify connection is still valid."""
        try:
            conn.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    def _maybe_cleanup(self):
        """Periodically cleanup dead thread connections."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now
        alive_threads = {t.ident for t in threading.enumerate()}
        dead_threads = set(self._connections.keys()) - alive_threads

        for thread_id in dead_threads:
            conn = self._connections.pop(thread_id, None)
            self._last_used.pop(thread_id, None)
            self._last_health_check.pop(thread_id, None)
            self._transaction_depth.pop(thread_id, None)

            if conn:
                try:
                    conn.close()
                    logger.debug(f"Cleaned up connection for dead thread {thread_id}")
                except Exception as e:
                    logger.warning(f"Error closing connection for thread {thread_id}: {e}")

        if dead_threads:
            logger.info(f"Connection pool cleanup: removed {len(dead_threads)} dead connections, {len(self._connections)} remaining")

    def get_connection(self) -> sqlite3.Connection:
        """Get a connection for the current thread, creating if necessary."""
        thread_id = threading.get_ident()
        now = time.time()

        with self._lock:
            self._maybe_cleanup()

            conn = self._connections.get(thread_id)

            # Check if connection needs health check
            if conn is not None:
                last_check = self._last_health_check.get(thread_id, 0)
                if now - last_check > self._health_check_interval:
                    if not self._health_check(conn):
                        logger.warning(f"Connection for thread {thread_id} failed health check, replacing")
                        try:
                            conn.close()
                        except Exception:
                            pass
                        conn = None
                    else:
                        self._last_health_check[thread_id] = now

            # Create new connection if needed
            if conn is None:
                if len(self._connections) >= self._max_size:
                    # Force cleanup before creating new connection
                    self._last_cleanup = 0
                    self._maybe_cleanup()

                    if len(self._connections) >= self._max_size:
                        raise RuntimeError(
                            f"Connection pool exhausted ({self._max_size} connections). "
                            f"Possible connection leak or too many threads."
                        )

                conn = self._create_connection()
                self._connections[thread_id] = conn
                self._last_health_check[thread_id] = now
                self._transaction_depth[thread_id] = 0
                logger.debug(f"Created connection for thread {thread_id} (pool size: {len(self._connections)})")

            self._last_used[thread_id] = now
            return conn

    def get_transaction_depth(self) -> int:
        """Get current transaction nesting depth for this thread."""
        return self._transaction_depth.get(threading.get_ident(), 0)

    def increment_transaction_depth(self):
        """Increment transaction depth (called on context entry)."""
        thread_id = threading.get_ident()
        with self._lock:
            self._transaction_depth[thread_id] = self._transaction_depth.get(thread_id, 0) + 1

    def decrement_transaction_depth(self):
        """Decrement transaction depth (called on context exit)."""
        thread_id = threading.get_ident()
        with self._lock:
            depth = self._transaction_depth.get(thread_id, 1)
            self._transaction_depth[thread_id] = max(0, depth - 1)

    def close_all(self):
        """Close all connections (call on application shutdown)."""
        with self._lock:
            for thread_id, conn in list(self._connections.items()):
                try:
                    conn.close()
                except Exception as e:
                    logger.warning(f"Error closing connection for thread {thread_id}: {e}")

            self._connections.clear()
            self._last_used.clear()
            self._last_health_check.clear()
            self._transaction_depth.clear()
            logger.info("Connection pool closed")

    def stats(self) -> dict:
        """Get pool statistics."""
        with self._lock:
            return {
                'active_connections': len(self._connections),
                'max_size': self._max_size,
                'thread_ids': list(self._connections.keys()),
            }


# Global pool instance
_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def _get_pool() -> ConnectionPool:
    """Get or create the global connection pool."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                DB_PATH.parent.mkdir(exist_ok=True, parents=True)
                _pool = ConnectionPool(DB_PATH)
    return _pool


@contextmanager
def get_db(read_only: bool = False):
    """
    Get database connection with proper transaction handling.

    Args:
        read_only: If True, skip commit on exit (optimization for read operations)

    Handles nested calls correctly:
    - Only the outermost context commits/rollbacks
    - Inner contexts are no-ops for transaction control
    """
    pool = _get_pool()
    conn = pool.get_connection()

    is_outermost = pool.get_transaction_depth() == 0
    pool.increment_transaction_depth()

    try:
        yield conn

        # Only commit on outermost context exit
        if is_outermost and not read_only:
            conn.commit()

    except Exception:
        # Only rollback on outermost context
        if is_outermost:
            conn.rollback()
        raise

    finally:
        pool.decrement_transaction_depth()


def close_pool():
    """Close the connection pool. Call on application shutdown."""
    global _pool
    if _pool is not None:
        _pool.close_all()
        _pool = None


def cleanup_connection_pool() -> None:
    """Legacy function - cleanup is now automatic."""
    pool = _get_pool()
    pool._maybe_cleanup()
