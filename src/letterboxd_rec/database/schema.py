"""Database schema initialization and migrations."""

import sqlite3
import logging
from datetime import datetime

from ..config import DB_PATH, MIGRATIONS_PATH, MIGRATION_VERSION_TABLE
from .connection import get_db
from .loaders import load_json

logger = logging.getLogger(__name__)


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
                fan_count INTEGER,
                is_short INTEGER,
                is_animation INTEGER,
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

            -- Optional social graph edges (used when follow scraping is enabled)
            CREATE TABLE IF NOT EXISTS user_relationships (
                follower TEXT NOT NULL,
                followee TEXT NOT NULL,
                scraped_at TEXT,
                source TEXT,
                PRIMARY KEY (follower, followee)
            );
            CREATE INDEX IF NOT EXISTS idx_user_relationships_follower ON user_relationships(follower);
            CREATE INDEX IF NOT EXISTS idx_user_relationships_followee ON user_relationships(followee);

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

            -- Social graph: who follows whom
            CREATE TABLE IF NOT EXISTS user_follows (
                follower TEXT NOT NULL,
                followee TEXT NOT NULL,
                scraped_at TEXT,
                PRIMARY KEY (follower, followee)
            );
            CREATE INDEX IF NOT EXISTS idx_follows_follower ON user_follows(follower);
            CREATE INDEX IF NOT EXISTS idx_follows_followee ON user_follows(followee);
        """)

        _record_baseline_migration(conn)

    # Apply any versioned migrations that are newer than baseline
    run_versioned_migrations()


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
        'composers': 'TEXT',
        'fan_count': 'INTEGER',
        'is_short': 'INTEGER',
        'is_animation': 'INTEGER',
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


def _ensure_user_relationships_table(conn):
    """Ensure the social edges table exists for follow-based graph edges."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS user_relationships (
            follower TEXT NOT NULL,
            followee TEXT NOT NULL,
            scraped_at TEXT,
            source TEXT,
            PRIMARY KEY (follower, followee)
        );
        CREATE INDEX IF NOT EXISTS idx_user_relationships_follower ON user_relationships(follower);
        CREATE INDEX IF NOT EXISTS idx_user_relationships_followee ON user_relationships(followee);
        """
    )


def run_versioned_migrations() -> int:
    """
    Apply versioned migrations found under MIGRATIONS_PATH/versions.
    Files are applied in lexicographic order and tracked in MIGRATION_VERSION_TABLE.
    """
    versions_dir = MIGRATIONS_PATH / "versions"

    applied = set()
    with get_db(read_only=True) as conn:
        _ensure_migration_table(conn)
        rows = conn.execute(f"SELECT version FROM {MIGRATION_VERSION_TABLE}").fetchall()
        applied = {r['version'] for r in rows}

    migration_files = sorted(p for p in versions_dir.glob("*.sql")) if versions_dir.exists() else []
    applied_count = 0

    # Apply lightweight Python migrations (idempotent/conditional) first
    python_migrations = [
        ("0001_add_film_columns", _migrate_films_table),
        ("0002_add_user_profile_schema_version", _migrate_user_profiles_table),
        ("0003_ensure_user_relationships_table", _ensure_user_relationships_table),
    ]

    for version, fn in python_migrations:
        if version in applied:
            continue
        with get_db() as conn:
            fn(conn)
            conn.execute(
                f"INSERT OR REPLACE INTO {MIGRATION_VERSION_TABLE} (version, applied_at) VALUES (?, ?)",
                (version, datetime.now().isoformat())
            )
        applied_count += 1

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
