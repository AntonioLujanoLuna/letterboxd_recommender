import sqlite3
import json
import logging
import os
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Adjust DB_PATH to be relative to the project root or a specific location
DB_PATH = Path(os.environ.get("LETTERBOXD_DB", "data/letterboxd.db"))


def init_db():
    DB_PATH.parent.mkdir(exist_ok=True, parents=True)
    with get_db() as conn:
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
                updated_at TEXT
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
            
            CREATE INDEX IF NOT EXISTS idx_fd_director ON film_directors(director);
            CREATE INDEX IF NOT EXISTS idx_fg_genre ON film_genres(genre);
            CREATE INDEX IF NOT EXISTS idx_fc_actor ON film_cast(actor);
            CREATE INDEX IF NOT EXISTS idx_ft_theme ON film_themes(theme);
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


def populate_normalized_tables(conn, film_metadata):
    """
    Populate normalized tables from film metadata.
    
    Args:
        conn: Database connection
        film_metadata: FilmMetadata object or dict with slug and attribute lists
    """
    slug = film_metadata.slug if hasattr(film_metadata, 'slug') else film_metadata['slug']
    
    # Clear existing entries for this film
    conn.execute("DELETE FROM film_directors WHERE film_slug = ?", (slug,))
    conn.execute("DELETE FROM film_genres WHERE film_slug = ?", (slug,))
    conn.execute("DELETE FROM film_cast WHERE film_slug = ?", (slug,))
    conn.execute("DELETE FROM film_themes WHERE film_slug = ?", (slug,))
    
    # Get attributes (handle both dataclass and dict)
    directors = film_metadata.directors if hasattr(film_metadata, 'directors') else load_json(film_metadata.get('directors', []))
    genres = film_metadata.genres if hasattr(film_metadata, 'genres') else load_json(film_metadata.get('genres', []))
    cast = film_metadata.cast if hasattr(film_metadata, 'cast') else load_json(film_metadata.get('cast', []))
    themes = film_metadata.themes if hasattr(film_metadata, 'themes') else load_json(film_metadata.get('themes', []))
    
    # Insert into normalized tables
    for director in directors:
        conn.execute("INSERT OR IGNORE INTO film_directors (film_slug, director) VALUES (?, ?)", (slug, director))
    
    for genre in genres:
        conn.execute("INSERT OR IGNORE INTO film_genres (film_slug, genre) VALUES (?, ?)", (slug, genre))
    
    for actor in cast:
        conn.execute("INSERT OR IGNORE INTO film_cast (film_slug, actor) VALUES (?, ?)", (slug, actor))
    
    for theme in themes:
        conn.execute("INSERT OR IGNORE INTO film_themes (film_slug, theme) VALUES (?, ?)", (slug, theme))


@contextmanager
def get_db(exclude_commit: bool = False):
    """
    Get database connection context manager.
    
    Args:
        exclude_commit: If True, skip commit on exit (for read-only operations)
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        if not exclude_commit:
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
    with get_db(exclude_commit=True) as conn:
        cursor = conn.execute("""
            SELECT username, list_slug, list_name, is_ranked, is_favorites, position, film_slug
            FROM user_lists
            WHERE username = ?
        """, (username,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]