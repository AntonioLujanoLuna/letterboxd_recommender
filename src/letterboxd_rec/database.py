import sqlite3
import json
import logging
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Adjust DB_PATH to be relative to the project root or a specific location
DB_PATH = Path("data/letterboxd.db")


def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS films (
                slug TEXT PRIMARY KEY,
                title TEXT,
                year INTEGER,
                directors TEXT,     -- JSON list
                genres TEXT,        -- JSON list
                cast TEXT,          -- JSON list
                themes TEXT,        -- JSON list (from Letterboxd tags)
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
            
            CREATE INDEX IF NOT EXISTS idx_user ON user_films(username);
            CREATE INDEX IF NOT EXISTS idx_user_film_slug ON user_films(film_slug);
            CREATE INDEX IF NOT EXISTS idx_film_year ON films(year);
            CREATE INDEX IF NOT EXISTS idx_lists_user ON user_lists(username);
            CREATE INDEX IF NOT EXISTS idx_lists_film ON user_lists(film_slug);
            CREATE INDEX IF NOT EXISTS idx_lists_favorites ON user_lists(is_favorites);
        """)
        
        # Migration: Add new columns to existing films table if they don't exist
        _migrate_films_table(conn)


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


@contextmanager
def get_db(readonly: bool = False):
    """
    Get database connection context manager.
    
    Args:
        readonly: If True, skip commit on exit (for read-only operations)
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        if not readonly:
            conn.commit()
    finally:
        conn.close()


def load_json(val):
    """Safely load JSON from db field."""
    if not val:
        return []
    if isinstance(val, list):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError) as e:
        logger.debug(f"Failed to parse JSON '{val[:50]}...': {e}")
        return []


def load_user_lists(username: str) -> list[dict]:
    """Load all list entries for a user from database."""
    with get_db(readonly=True) as conn:
        cursor = conn.execute("""
            SELECT username, list_slug, list_name, is_ranked, is_favorites, position, film_slug
            FROM user_lists
            WHERE username = ?
        """, (username,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]