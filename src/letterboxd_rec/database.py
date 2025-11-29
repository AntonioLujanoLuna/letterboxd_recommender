import sqlite3
import json
from pathlib import Path
from contextlib import contextmanager

# Adjust DB_PATH to be relative to the project root or a specific location
# Assuming running from project root
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
                rating_count INTEGER
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
            
            CREATE INDEX IF NOT EXISTS idx_user ON user_films(username);
            CREATE INDEX IF NOT EXISTS idx_film_year ON films(year);
        """)

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
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
        print(f"Warning: Failed to parse JSON: {e}")
        return []
