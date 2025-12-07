import sqlite3
import json
import logging
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from .config import DB_PATH, MIGRATIONS_PATH, MIGRATION_VERSION_TABLE

logger = logging.getLogger(__name__)


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
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
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


def init_db() -> None:
    DB_PATH.parent.mkdir(exist_ok=True, parents=True)
    with get_db() as conn:
        _ensure_migration_table(conn)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS films (
                slug TEXT PRIMARY KEY,
                title TEXT,
                year INTEGER,
                directors TEXT,     -- JSON list (kept for backward compat)
                genres TEXT,        -- JSON list
                cast TEXT,          -- JSON list
                themes TEXT,        -- JSON list
                runtime INTEGER,
                avg_rating REAL,
                rating_count INTEGER,
                countries TEXT,     -- JSON list
                languages TEXT,     -- JSON list
                writers TEXT,       -- JSON list
                cinematographers TEXT,  -- JSON list
                composers TEXT      -- JSON list
            );
            
            CREATE TABLE IF NOT EXISTS user_films (
                username TEXT,
                film_slug TEXT,
                rating REAL,
                watched INTEGER DEFAULT 0,
                watchlisted INTEGER DEFAULT 0,
                liked INTEGER DEFAULT 0,
                scraped_at TEXT,
                PRIMARY KEY (username, film_slug)
            );
            
            CREATE TABLE IF NOT EXISTS user_lists (
                username TEXT,
                list_slug TEXT,
                list_name TEXT,
                is_ranked INTEGER DEFAULT 0,
                is_favorites INTEGER DEFAULT 0,
                position INTEGER,
                film_slug TEXT,
                scraped_at TEXT,
                PRIMARY KEY (username, list_slug, film_slug)
            );
            
            -- New table for caching user profiles
            CREATE TABLE IF NOT EXISTS user_profiles (
                username TEXT PRIMARY KEY,
                profile_data TEXT,  -- JSON blob of profile stats
                updated_at TEXT,
                schema_version INTEGER DEFAULT 0
            );
            
            -- Normalized tables for better querying
            CREATE TABLE IF NOT EXISTS film_directors (film_slug TEXT, director TEXT, PRIMARY KEY (film_slug, director));
            CREATE TABLE IF NOT EXISTS film_genres (film_slug TEXT, genre TEXT, PRIMARY KEY (film_slug, genre));
            CREATE TABLE IF NOT EXISTS film_cast (film_slug TEXT, actor TEXT, PRIMARY KEY (film_slug, actor));
            CREATE TABLE IF NOT EXISTS film_themes (film_slug TEXT, theme TEXT, PRIMARY KEY (film_slug, theme));
            
            CREATE INDEX IF NOT EXISTS idx_user ON user_films(username);
            CREATE INDEX IF NOT EXISTS idx_user_film_slug ON user_films(film_slug);
            CREATE INDEX IF NOT EXISTS idx_film_year ON films(year);
            CREATE INDEX IF NOT EXISTS idx_lists_user ON user_lists(username);
            CREATE INDEX IF NOT EXISTS idx_lists_film ON user_lists(film_slug);
            CREATE INDEX IF NOT EXISTS idx_lists_favorites ON user_lists(is_favorites);

            -- Additional indexes for filter queries in triage and profile building
            CREATE INDEX IF NOT EXISTS idx_user_films_watched ON user_films(watched);
            CREATE INDEX IF NOT EXISTS idx_user_films_liked ON user_films(liked);
            CREATE INDEX IF NOT EXISTS idx_user_films_watchlisted ON user_films(watchlisted);
            CREATE INDEX IF NOT EXISTS idx_user_films_rating ON user_films(rating);

            -- Composite indexes for common query patterns
            CREATE INDEX IF NOT EXISTS idx_user_watchlisted ON user_films(username, watchlisted);
            CREATE INDEX IF NOT EXISTS idx_user_rating ON user_films(username, rating);

            -- Covering indexes for common film filter patterns (year + rating)
            CREATE INDEX IF NOT EXISTS idx_film_year_rating ON films(year, avg_rating);
            CREATE INDEX IF NOT EXISTS idx_film_rating ON films(avg_rating);

            CREATE INDEX IF NOT EXISTS idx_fd_director ON film_directors(director);
            CREATE INDEX IF NOT EXISTS idx_fg_genre ON film_genres(genre);
            CREATE INDEX IF NOT EXISTS idx_fc_actor ON film_cast(actor);
            CREATE INDEX IF NOT EXISTS idx_ft_theme ON film_themes(theme);

            -- Discovery source caching tables
            CREATE TABLE IF NOT EXISTS discovery_sources (
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                last_page_scraped INTEGER DEFAULT 0,
                total_users_found INTEGER DEFAULT 0,
                scraped_at TEXT,
                PRIMARY KEY (source_type, source_id)
            );

            CREATE TABLE IF NOT EXISTS pending_users (
                username TEXT PRIMARY KEY,
                discovered_from_type TEXT NOT NULL,
                discovered_from_id TEXT NOT NULL,
                discovered_at TEXT NOT NULL,
                priority INTEGER DEFAULT 50
            );

            CREATE INDEX IF NOT EXISTS idx_pending_priority ON pending_users(priority DESC, discovered_at ASC);

            -- IDF (Inverse Document Frequency) table for rarity weighting
            CREATE TABLE IF NOT EXISTS attribute_idf (
                attribute_type TEXT NOT NULL,
                attribute_value TEXT NOT NULL,
                doc_count INTEGER NOT NULL,
                idf_score REAL NOT NULL,
                PRIMARY KEY (attribute_type, attribute_value)
            );

            CREATE INDEX IF NOT EXISTS idx_idf_type ON attribute_idf(attribute_type);

            -- Scraping session tracking for daemon/resume visibility
            CREATE TABLE IF NOT EXISTS scrape_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT DEFAULT 'running',
                users_scraped INTEGER DEFAULT 0,
                films_added INTEGER DEFAULT 0,
                last_activity TEXT
            );
        """)

        # Migration: Add new columns to existing films table if they don't exist
        _migrate_films_table(conn)
        _migrate_user_profiles_table(conn)
        _record_baseline_migration(conn)

    # Apply any versioned migrations that are newer than baseline
    run_versioned_migrations()


def _migrate_films_table(conn):
    """Add new columns to films table if they don't exist."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(films)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    new_columns = {
        'countries': 'TEXT',
        'languages': 'TEXT',
        'writers': 'TEXT',
        'cinematographers': 'TEXT',
        'composers': 'TEXT'
    }

    for col_name, col_type in new_columns.items():
        if col_name not in existing_columns:
            try:
                conn.execute(f"ALTER TABLE films ADD COLUMN {col_name} {col_type}")
                logger.info(f"Added column '{col_name}' to films table")
            except sqlite3.Error as e:
                logger.warning(f"Could not add column '{col_name}': {e}")


def _migrate_user_profiles_table(conn):
    """Add schema_version column to user_profiles table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(user_profiles)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    if 'schema_version' not in existing_columns:
        try:
            conn.execute("ALTER TABLE user_profiles ADD COLUMN schema_version INTEGER DEFAULT 0")
            logger.info("Added column 'schema_version' to user_profiles table")
        except sqlite3.Error as e:
            logger.warning(f"Could not add column 'schema_version': {e}")


def _ensure_migration_table(conn):
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {MIGRATION_VERSION_TABLE} (
            version TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
    """)


def _record_baseline_migration(conn):
    """Mark baseline schema as applied to align with versioned migrations."""
    baseline = "0000_baseline"
    cursor = conn.execute(
        f"SELECT version FROM {MIGRATION_VERSION_TABLE} WHERE version = ?",
        (baseline,)
    )
    if cursor.fetchone():
        return
    conn.execute(
        f"INSERT INTO {MIGRATION_VERSION_TABLE} (version, applied_at) VALUES (?, ?)",
        (baseline, datetime.now().isoformat())
    )


def run_versioned_migrations() -> int:
    """
    Apply versioned migrations found under MIGRATIONS_PATH/versions.
    Files are applied in lexicographic order and tracked in MIGRATION_VERSION_TABLE.
    """
    versions_dir = MIGRATIONS_PATH / "versions"
    if not versions_dir.exists():
        return 0

    applied = set()
    with get_db(read_only=True) as conn:
        _ensure_migration_table(conn)
        rows = conn.execute(f"SELECT version FROM {MIGRATION_VERSION_TABLE}").fetchall()
        applied = {r['version'] for r in rows}

    migration_files = sorted(p for p in versions_dir.glob("*.sql"))
    applied_count = 0

    for path in migration_files:
        version = path.stem
        if version in applied:
            continue
        sql = path.read_text()
        with get_db() as conn:
            conn.executescript(sql)
            conn.execute(
                f"INSERT OR REPLACE INTO {MIGRATION_VERSION_TABLE} (version, applied_at) VALUES (?, ?)",
                (version, datetime.now().isoformat())
            )
        applied_count += 1
        logger.info(f"Applied migration {version}")

    return applied_count


def populate_normalized_tables(conn, film_metadata):
    """
    Populate normalized tables from film metadata for a single film.

    This is a convenience wrapper around populate_normalized_tables_batch
    for single-film operations.

    Args:
        conn: Database connection
        film_metadata: FilmMetadata object or dict with slug and attribute lists
    """
    populate_normalized_tables_batch(conn, [film_metadata])


def populate_normalized_tables_batch(conn, film_metadata_list: list) -> None:
    """
    Populate normalized tables for multiple films efficiently using batch operations.

    Performs bulk DELETEs and INSERTs to minimize database round-trips.
    Handles SQLite's parameter limit (999) by chunking large batches.

    Args:
        conn: Database connection (caller manages transaction)
        film_metadata_list: List of FilmMetadata objects or dicts
    """
    if not film_metadata_list:
        return

    # Collect all slugs for batch DELETE
    slugs = []
    for fm in film_metadata_list:
        slug = fm.slug if hasattr(fm, 'slug') else fm['slug']
        slugs.append(slug)

    # Batch DELETE using IN clause (SQLite supports up to 999 parameters, chunk if needed)
    CHUNK_SIZE = 900  # Leave room for safety
    for i in range(0, len(slugs), CHUNK_SIZE):
        chunk = slugs[i:i + CHUNK_SIZE]
        placeholders = ','.join('?' * len(chunk))
        conn.execute(f"DELETE FROM film_directors WHERE film_slug IN ({placeholders})", chunk)
        conn.execute(f"DELETE FROM film_genres WHERE film_slug IN ({placeholders})", chunk)
        conn.execute(f"DELETE FROM film_cast WHERE film_slug IN ({placeholders})", chunk)
        conn.execute(f"DELETE FROM film_themes WHERE film_slug IN ({placeholders})", chunk)

    # Collect all inserts
    director_rows = []
    genre_rows = []
    cast_rows = []
    theme_rows = []

    for fm in film_metadata_list:
        slug = fm.slug if hasattr(fm, 'slug') else fm['slug']

        directors = fm.directors if hasattr(fm, 'directors') else load_json(fm.get('directors', []))
        genres = fm.genres if hasattr(fm, 'genres') else load_json(fm.get('genres', []))
        cast = fm.cast if hasattr(fm, 'cast') else load_json(fm.get('cast', []))
        themes = fm.themes if hasattr(fm, 'themes') else load_json(fm.get('themes', []))

        director_rows.extend((slug, d) for d in directors)
        genre_rows.extend((slug, g) for g in genres)
        cast_rows.extend((slug, a) for a in cast)
        theme_rows.extend((slug, t) for t in themes)

    # Batch INSERT
    if director_rows:
        conn.executemany("INSERT OR IGNORE INTO film_directors (film_slug, director) VALUES (?, ?)", director_rows)
    if genre_rows:
        conn.executemany("INSERT OR IGNORE INTO film_genres (film_slug, genre) VALUES (?, ?)", genre_rows)
    if cast_rows:
        conn.executemany("INSERT OR IGNORE INTO film_cast (film_slug, actor) VALUES (?, ?)", cast_rows)
    if theme_rows:
        conn.executemany("INSERT OR IGNORE INTO film_themes (film_slug, theme) VALUES (?, ?)", theme_rows)


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


def load_json(val):
    """Safely load JSON from db field."""
    if not val:
        return []
    if isinstance(val, list):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse JSON '{val[:50]}...': {e}")
        return []


def load_user_lists(username: str) -> list[dict]:
    """Load all list entries for a user from database."""
    with get_db(read_only=True) as conn:
        cursor = conn.execute("""
            SELECT username, list_slug, list_name, is_ranked, is_favorites, position, film_slug
            FROM user_lists
            WHERE username = ?
        """, (username,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def load_cached_profile(username: str, max_age_days: int = 7) -> dict | None:
    """
    Load cached user profile if it exists and is recent enough.

    Cache is invalidated if:
    - Profile schema version doesn't match current version
    - Profile is older than max_age_days
    - User's lists have been updated more recently than the profile
    - User's films have been updated more recently than the profile
    """
    from .config import PROFILE_SCHEMA_VERSION

    with get_db(read_only=True) as conn:
        cursor = conn.execute("""
            SELECT profile_data, updated_at, schema_version
            FROM user_profiles
            WHERE username = ?
        """, (username,))
        row = cursor.fetchone()

        if not row:
            return None

        # Check schema version FIRST before any other validation
        cached_version = row['schema_version'] if 'schema_version' in row.keys() else 0
        if cached_version != PROFILE_SCHEMA_VERSION:
            logger.debug(f"Profile cache invalidated for {username} - schema version mismatch ({cached_version} != {PROFILE_SCHEMA_VERSION})")
            return None

        # Use helper to ensure naive datetime for consistent comparisons
        profile_updated_at = parse_timestamp_naive(row['updated_at'])
        now = datetime.now()

        # Check age
        if (now - profile_updated_at).days > max_age_days:
            return None

        # Check if user_lists have been updated more recently
        lists_cursor = conn.execute("""
            SELECT MAX(scraped_at) as last_list_update
            FROM user_lists
            WHERE username = ?
        """, (username,))
        lists_row = lists_cursor.fetchone()

        if lists_row and lists_row['last_list_update']:
            last_list_update = parse_timestamp_naive(lists_row['last_list_update'])
            if last_list_update > profile_updated_at:
                logger.debug(f"Profile cache invalidated for {username} - lists updated more recently")
                return None

        # Check if user_films have been updated more recently
        films_cursor = conn.execute("""
            SELECT MAX(scraped_at) as last_film_update
            FROM user_films
            WHERE username = ?
        """, (username,))
        films_row = films_cursor.fetchone()

        if films_row and films_row['last_film_update']:
            last_film_update = parse_timestamp_naive(films_row['last_film_update'])
            if last_film_update > profile_updated_at:
                logger.debug(f"Profile cache invalidated for {username} - films updated more recently")
                return None

        return json.loads(row['profile_data'])


def save_user_profile(username: str, profile_data: dict) -> None:
    """Save user profile to cache with naive datetime timestamp and schema version."""
    from .config import PROFILE_SCHEMA_VERSION

    with get_db() as conn:
        # Always use naive datetime for consistency
        conn.execute("""
            INSERT OR REPLACE INTO user_profiles (username, profile_data, updated_at, schema_version)
            VALUES (?, ?, ?, ?)
        """, (username, json.dumps(profile_data), datetime.now().isoformat(), PROFILE_SCHEMA_VERSION))


def purge_stale_profile_caches() -> int:
    """
    Delete all cached profiles with outdated schema versions.

    Returns:
        Count of profiles deleted
    """
    from .config import PROFILE_SCHEMA_VERSION

    with get_db() as conn:
        cursor = conn.execute("""
            DELETE FROM user_profiles
            WHERE schema_version IS NULL OR schema_version != ?
        """, (PROFILE_SCHEMA_VERSION,))
        return cursor.rowcount


def get_discovery_source(source_type: str, source_id: str) -> dict | None:
    """
    Get cached discovery source information.

    Returns dict with last_page_scraped, total_users_found, and scraped_at,
    or None if source not found.
    """
    with get_db(read_only=True) as conn:
        cursor = conn.execute("""
            SELECT last_page_scraped, total_users_found, scraped_at
            FROM discovery_sources
            WHERE source_type = ? AND source_id = ?
        """, (source_type, source_id))
        row = cursor.fetchone()
        return dict(row) if row else None


def update_discovery_source(
    source_type: str,
    source_id: str,
    last_page: int,
    total_users: int
) -> None:
    """Update discovery source cache with new page and user count."""
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO discovery_sources
            (source_type, source_id, last_page_scraped, total_users_found, scraped_at)
            VALUES (?, ?, ?, ?, ?)
        """, (source_type, source_id, last_page, total_users, datetime.now().isoformat()))


def add_pending_users(
    usernames: list[str],
    source_type: str,
    source_id: str,
    priority: int = 50
) -> int:
    """
    Add discovered usernames to pending_users queue.

    Returns count of newly added users (ignores duplicates and existing users).
    """
    with get_db() as conn:
        # Get existing users in user_films and pending_users
        existing_scraped = {
            r['username'] for r in conn.execute(
                "SELECT DISTINCT username FROM user_films"
            )
        }
        existing_pending = {
            r['username'] for r in conn.execute(
                "SELECT username FROM pending_users"
            )
        }

        # Filter to truly new users and de-duplicate this batch while preserving order
        seen = set()
        new_users = []
        for u in usernames:
            if u in seen:
                continue
            seen.add(u)
            if u in existing_scraped or u in existing_pending:
                continue
            new_users.append(u)

        if not new_users:
            return 0

        # Insert new pending users
        timestamp = datetime.now().isoformat()
        conn.executemany("""
            INSERT OR IGNORE INTO pending_users
            (username, discovered_from_type, discovered_from_id, discovered_at, priority)
            VALUES (?, ?, ?, ?, ?)
        """, [(u, source_type, source_id, timestamp, priority) for u in new_users])

        return len(new_users)


def get_pending_users(limit: int = 50) -> list[dict]:
    """
    Get pending users from queue, ordered by priority (desc) then discovery time (asc).

    Returns list of dicts with username, discovered_from_type, discovered_from_id,
    discovered_at, and priority.
    """
    with get_db(read_only=True) as conn:
        cursor = conn.execute("""
            SELECT username, discovered_from_type, discovered_from_id, discovered_at, priority
            FROM pending_users
            ORDER BY priority DESC, discovered_at ASC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]


def remove_pending_user(username: str) -> None:
    """Remove a user from pending_users queue after successful scrape."""
    with get_db() as conn:
        conn.execute("DELETE FROM pending_users WHERE username = ?", (username,))


def get_pending_queue_stats() -> dict:
    """Get statistics about the pending users queue."""
    with get_db(read_only=True) as conn:
        cursor = conn.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(DISTINCT discovered_from_type) as source_types,
                AVG(priority) as avg_priority
            FROM pending_users
        """)
        row = cursor.fetchone()

        # Get breakdown by source type
        breakdown_cursor = conn.execute("""
            SELECT discovered_from_type, COUNT(*) as count
            FROM pending_users
            GROUP BY discovered_from_type
            ORDER BY count DESC
        """)
        breakdown = {r['discovered_from_type']: r['count'] for r in breakdown_cursor.fetchall()}

        return {
            'total': row['total'],
            'source_types': row['source_types'],
            'avg_priority': row['avg_priority'],
            'breakdown': breakdown
        }


def create_scrape_session() -> int:
    """Create a new scraping session record and return its ID."""
    with get_db() as conn:
        cursor = conn.execute("""
            INSERT INTO scrape_sessions (started_at, status, last_activity)
            VALUES (?, 'running', ?)
        """, (datetime.now().isoformat(), datetime.now().isoformat()))
        return cursor.lastrowid


def update_session_progress(session_id: int, users_scraped: int, films_added: int) -> None:
    """Update session progress counters and heartbeat."""
    with get_db() as conn:
        conn.execute("""
            UPDATE scrape_sessions
            SET users_scraped = ?, films_added = ?, last_activity = ?
            WHERE id = ?
        """, (users_scraped, films_added, datetime.now().isoformat(), session_id))


def complete_session(session_id: int, status: str = "completed") -> None:
    """Mark a scraping session as completed/interrupted."""
    with get_db() as conn:
        conn.execute("""
            UPDATE scrape_sessions
            SET status = ?, completed_at = ?, last_activity = ?
            WHERE id = ?
        """, (status, datetime.now().isoformat(), datetime.now().isoformat(), session_id))


def get_session_history(limit: int = 10) -> list[dict]:
    """Return recent scraping sessions for visibility."""
    with get_db(read_only=True) as conn:
        rows = conn.execute("""
            SELECT id, started_at, completed_at, status, users_scraped, films_added, last_activity
            FROM scrape_sessions
            ORDER BY started_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]


def compute_and_store_idf() -> dict[str, int]:
    """
    Compute and store IDF (Inverse Document Frequency) for all attribute values.

    IDF measures how distinctive an attribute value is across the film corpus.
    Formula: IDF(value) = log(N / (1 + doc_count))
    where N = total films, doc_count = films with this value

    Returns dict with counts of attributes processed per type.
    """
    import math

    with get_db() as conn:
        # Get total film count
        cursor = conn.execute("SELECT COUNT(*) as total FROM films")
        N = cursor.fetchone()['total']

        if N == 0:
            return {}

        logger = logging.getLogger(__name__)
        logger.info(f"Computing IDF for {N} films...")

        results = {}

        # Process genres
        cursor = conn.execute("""
            SELECT genre as value, COUNT(*) as doc_count
            FROM film_genres
            GROUP BY genre
        """)
        genre_data = [(r['value'], r['doc_count']) for r in cursor.fetchall()]
        conn.executemany("""
            INSERT OR REPLACE INTO attribute_idf (attribute_type, attribute_value, doc_count, idf_score)
            VALUES ('genre', ?, ?, ?)
        """, [(value, count, math.log(N / (1 + count))) for value, count in genre_data])
        results['genre'] = len(genre_data)

        # Process directors
        cursor = conn.execute("""
            SELECT director as value, COUNT(*) as doc_count
            FROM film_directors
            GROUP BY director
        """)
        director_data = [(r['value'], r['doc_count']) for r in cursor.fetchall()]
        conn.executemany("""
            INSERT OR REPLACE INTO attribute_idf (attribute_type, attribute_value, doc_count, idf_score)
            VALUES ('director', ?, ?, ?)
        """, [(value, count, math.log(N / (1 + count))) for value, count in director_data])
        results['director'] = len(director_data)

        # Process actors
        cursor = conn.execute("""
            SELECT actor as value, COUNT(*) as doc_count
            FROM film_cast
            GROUP BY actor
        """)
        actor_data = [(r['value'], r['doc_count']) for r in cursor.fetchall()]
        conn.executemany("""
            INSERT OR REPLACE INTO attribute_idf (attribute_type, attribute_value, doc_count, idf_score)
            VALUES ('actor', ?, ?, ?)
        """, [(value, count, math.log(N / (1 + count))) for value, count in actor_data])
        results['actor'] = len(actor_data)

        # Process themes
        cursor = conn.execute("""
            SELECT theme as value, COUNT(*) as doc_count
            FROM film_themes
            GROUP BY theme
        """)
        theme_data = [(r['value'], r['doc_count']) for r in cursor.fetchall()]
        conn.executemany("""
            INSERT OR REPLACE INTO attribute_idf (attribute_type, attribute_value, doc_count, idf_score)
            VALUES ('theme', ?, ?, ?)
        """, [(value, count, math.log(N / (1 + count))) for value, count in theme_data])
        results['theme'] = len(theme_data)

        # Process countries - single grouped query
        cursor = conn.execute("""
            WITH country_films AS (
                SELECT json_each.value as country, slug
                FROM films, json_each(films.countries)
                WHERE countries IS NOT NULL
            )
            SELECT country as value, COUNT(DISTINCT slug) as doc_count
            FROM country_films
            GROUP BY country
        """)
        country_data = [(r['value'], r['doc_count']) for r in cursor.fetchall()]
        conn.executemany("""
            INSERT OR REPLACE INTO attribute_idf (attribute_type, attribute_value, doc_count, idf_score)
            VALUES ('country', ?, ?, ?)
        """, [(value, count, math.log(N / (1 + count))) for value, count in country_data])
        results['country'] = len(country_data)

        # Process languages - single grouped query
        cursor = conn.execute("""
            WITH language_films AS (
                SELECT json_each.value as language, slug
                FROM films, json_each(films.languages)
                WHERE languages IS NOT NULL
            )
            SELECT language as value, COUNT(DISTINCT slug) as doc_count
            FROM language_films
            GROUP BY language
        """)
        language_data = [(r['value'], r['doc_count']) for r in cursor.fetchall()]
        conn.executemany("""
            INSERT OR REPLACE INTO attribute_idf (attribute_type, attribute_value, doc_count, idf_score)
            VALUES ('language', ?, ?, ?)
        """, [(value, count, math.log(N / (1 + count))) for value, count in language_data])
        results['language'] = len(language_data)

        logger.info(f"IDF computation complete: {results}")
        return results


def load_idf() -> dict[str, dict[str, float]]:
    """
    Load all IDF scores from database.

    Returns nested dict: {"genre": {"drama": 0.5, ...}, "director": {...}, ...}
    """
    with get_db(read_only=True) as conn:
        cursor = conn.execute("""
            SELECT attribute_type, attribute_value, idf_score
            FROM attribute_idf
        """)

        idf = {}
        for row in cursor.fetchall():
            attr_type = row['attribute_type']
            if attr_type not in idf:
                idf[attr_type] = {}
            idf[attr_type][row['attribute_value']] = row['idf_score']

        return idf


def run_maintenance(vacuum: bool = True, analyze: bool = True) -> None:
    """
    Run optional VACUUM/ANALYZE after bulk loads.
    Uses a dedicated connection to avoid interfering with pooled transactions.
    """
    import sqlite3
    if not vacuum and not analyze:
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        if vacuum:
            conn.execute("VACUUM")
        if analyze:
            conn.execute("ANALYZE")
        conn.commit()
    finally:
        conn.close()


def load_user_films_batch(usernames: list[str]) -> dict[str, list[dict]]:
    """
    Load films for multiple users in a single query.

    Much more efficient than N separate queries for collaborative filtering.
    """
    if not usernames:
        return {}

    with get_db(read_only=True) as conn:
        placeholders = ','.join('?' * len(usernames))
        rows = conn.execute(f"""
            SELECT username, film_slug as slug, rating, watched, watchlisted, liked, scraped_at
            FROM user_films
            WHERE username IN ({placeholders})
        """, usernames).fetchall()

    result = defaultdict(list)
    for row in rows:
        row_dict = dict(row)
        username = row_dict.pop('username')
        result[username].append(row_dict)

    return dict(result)


def load_films_by_attribute(
    attribute_type: str,
    attribute_values: list[str],
    limit_per_value: int = 50
) -> dict[str, list[dict]]:
    """
    Efficiently load films grouped by attribute (genre, director, etc).
    Useful for "find more films by X" queries.
    """
    if not attribute_values:
        return {}

    with get_db(read_only=True) as conn:
        if attribute_type == 'director':
            table, col = 'film_directors', 'director'
        elif attribute_type == 'genre':
            table, col = 'film_genres', 'genre'
        elif attribute_type == 'actor':
            table, col = 'film_cast', 'actor'
        else:
            raise ValueError(f"Unknown attribute type: {attribute_type}")

        placeholders = ','.join('?' * len(attribute_values))

        rows = conn.execute(f"""
            WITH ranked AS (
                SELECT 
                    a.{col} as attr_value,
                    f.*,
                    ROW_NUMBER() OVER (PARTITION BY a.{col} ORDER BY f.rating_count DESC) as rn
                FROM {table} a
                JOIN films f ON a.film_slug = f.slug
                WHERE a.{col} IN ({placeholders})
            )
            SELECT * FROM ranked WHERE rn <= ?
        """, [*attribute_values, limit_per_value]).fetchall()

    result = defaultdict(list)
    for row in rows:
        row_dict = dict(row)
        attr_value = row_dict.pop('attr_value')
        row_dict.pop('rn', None)
        result[attr_value].append(row_dict)

    return dict(result)


def update_idf_incremental(new_film_slugs: list[str]) -> None:
    """
    Incrementally update IDF scores when new films are added.
    Much faster than full recompute for small additions.
    """
    import math

    if not new_film_slugs:
        return

    with get_db() as conn:
        current_total = conn.execute("SELECT COUNT(*) FROM films").fetchone()[0]
        new_total = current_total  # Already includes new films

        for attr_type, table, col in [
            ('genre', 'film_genres', 'genre'),
            ('director', 'film_directors', 'director'),
            ('actor', 'film_cast', 'actor'),
            ('theme', 'film_themes', 'theme'),
        ]:
            placeholders = ','.join('?' * len(new_film_slugs))

            new_values = conn.execute(f"""
                SELECT {col}, COUNT(*) as new_count
                FROM {table}
                WHERE film_slug IN ({placeholders})
                GROUP BY {col}
            """, new_film_slugs).fetchall()

            for row in new_values:
                value, added_count = row[col], row['new_count']

                existing = conn.execute("""
                    SELECT doc_count FROM attribute_idf
                    WHERE attribute_type = ? AND attribute_value = ?
                """, (attr_type, value)).fetchone()

                if existing:
                    new_doc_count = existing['doc_count'] + added_count
                else:
                    new_doc_count = added_count

                new_idf = math.log(new_total / (1 + new_doc_count))

                conn.execute("""
                    INSERT OR REPLACE INTO attribute_idf 
                    (attribute_type, attribute_value, doc_count, idf_score)
                    VALUES (?, ?, ?, ?)
                """, (attr_type, value, new_doc_count, new_idf))