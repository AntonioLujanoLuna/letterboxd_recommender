"""IDF (Inverse Document Frequency) computation and storage."""

import math
import logging

from .connection import get_db

logger = logging.getLogger(__name__)


def compute_and_store_idf() -> dict[str, int]:
    """
    Compute and store IDF (Inverse Document Frequency) for all attribute values.

    IDF measures how distinctive an attribute value is across the film corpus.
    Formula: IDF(value) = log(N / (1 + doc_count))
    where N = total films, doc_count = films with this value

    Returns dict with counts of attributes processed per type.
    """
    with get_db() as conn:
        # Get total film count
        cursor = conn.execute("SELECT COUNT(*) as total FROM films")
        N = cursor.fetchone()['total']

        if N == 0:
            return {}

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


def update_idf_incremental(new_film_slugs: list[str]) -> None:
    """
    Incrementally update IDF scores when new films are added.
    Much faster than full recompute for small additions.
    """
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
