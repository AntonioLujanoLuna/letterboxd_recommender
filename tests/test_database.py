import json

from letterboxd_rec import database, cli


def test_init_db_creates_expected_tables(fresh_db):
    db = fresh_db
    db.init_db()

    with db.get_db(read_only=True) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }

    expected = {
        "films",
        "user_films",
        "user_lists",
        "user_profiles",
        "discovery_sources",
        "pending_users",
        "attribute_idf",
    }
    assert expected.issubset(tables)


def test_nested_transactions_commit_once(fresh_db):
    db = fresh_db
    db.init_db()

    with db.get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO films (slug, title) VALUES (?, ?)", ("film-a", "Film A"))
        with db.get_db() as inner:
            inner.execute(
                "INSERT OR REPLACE INTO user_films (username, film_slug, watched) VALUES (?, ?, ?)",
                ("alice", "film-a", 1),
            )

    with db.get_db(read_only=True) as conn:
        count = conn.execute("SELECT COUNT(*) FROM user_films").fetchone()[0]
        assert count == 1


def test_add_pending_users_filters_existing(fresh_db):
    db = fresh_db
    db.init_db()

    with db.get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO user_films (username, film_slug, watched) VALUES (?, ?, ?)",
            ("alice", "film-a", 1),
        )

    added = db.add_pending_users(["alice", "bob", "bob", "carol"], "followers", "alice", priority=80)
    assert added == 2  # alice is already scraped; bob duplicated

    # Second attempt should not add duplicates
    assert db.add_pending_users(["bob", "carol"], "followers", "alice", priority=80) == 0

    pending = db.get_pending_users()
    assert {p["username"] for p in pending} == {"bob", "carol"}


def test_load_json_handles_invalid_payload():
    assert database.load_json("not-json") == []
    assert database.load_json(None) == []


def test_compute_and_store_idf_populates_attribute_tables(fresh_db):
    db = fresh_db
    db.init_db()

    films = [
        {
            "slug": "film-a",
            "title": "A",
            "year": 2000,
            "genres": json.dumps(["drama"]),
            "directors": json.dumps(["DirA"]),
            "cast": json.dumps(["ActorA"]),
            "themes": json.dumps(["ThemeA"]),
            "runtime": 100,
            "avg_rating": 4.0,
            "rating_count": 1000,
            "countries": json.dumps(["USA"]),
            "languages": json.dumps(["English"]),
            "writers": json.dumps(["WriterA"]),
            "cinematographers": json.dumps(["CineA"]),
            "composers": json.dumps(["CompA"]),
        },
        {
            "slug": "film-b",
            "title": "B",
            "year": 2001,
            "genres": json.dumps(["drama", "thriller"]),
            "directors": json.dumps(["DirB"]),
            "cast": json.dumps(["ActorB"]),
            "themes": json.dumps(["ThemeB"]),
            "runtime": 90,
            "avg_rating": 3.5,
            "rating_count": 500,
            "countries": json.dumps(["UK"]),
            "languages": json.dumps(["English"]),
            "writers": json.dumps(["WriterB"]),
            "cinematographers": json.dumps(["CineB"]),
            "composers": json.dumps(["CompB"]),
        },
    ]

    with db.get_db() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO films
            (slug, title, year, directors, genres, cast, themes, runtime, avg_rating, rating_count,
             countries, languages, writers, cinematographers, composers)
            VALUES (:slug, :title, :year, :directors, :genres, :cast, :themes, :runtime, :avg_rating, :rating_count,
                    :countries, :languages, :writers, :cinematographers, :composers)
            """,
            films,
        )
        db.populate_normalized_tables_batch(conn, films)

    results = db.compute_and_store_idf()

    assert results["genre"] >= 2
    assert results["director"] >= 2

    with db.get_db(read_only=True) as conn:
        row = conn.execute(
            "SELECT idf_score FROM attribute_idf WHERE attribute_type='genre' AND attribute_value='thriller'"
        ).fetchone()
        assert row is not None


def test_populate_normalized_tables_batch_replaces_entries(fresh_db):
    db = fresh_db
    db.init_db()

    film = {
        "slug": "film-a",
        "directors": ["DirA"],
        "genres": ["Drama"],
        "cast": ["ActorA"],
        "themes": ["ThemeA"],
    }

    with db.get_db() as conn:
        db.populate_normalized_tables(conn, film)
        # Update attributes and ensure old ones are replaced
        film_updated = {
            "slug": "film-a",
            "directors": ["DirB"],
            "genres": ["Comedy"],
            "cast": ["ActorB"],
            "themes": ["ThemeB"],
        }
        db.populate_normalized_tables(conn, film_updated)

    with db.get_db(read_only=True) as conn:
        director = conn.execute(
            "SELECT director FROM film_directors WHERE film_slug='film-a'"
        ).fetchone()[0]
        genre = conn.execute(
            "SELECT genre FROM film_genres WHERE film_slug='film-a'"
        ).fetchone()[0]
        actor = conn.execute(
            "SELECT actor FROM film_cast WHERE film_slug='film-a'"
        ).fetchone()[0]
        theme = conn.execute(
            "SELECT theme FROM film_themes WHERE film_slug='film-a'"
        ).fetchone()[0]

    assert director == "DirB"
    assert genre == "Comedy"
    assert actor == "ActorB"
    assert theme == "ThemeB"


def test_discovery_source_roundtrip(fresh_db):
    db = fresh_db
    db.init_db()

    assert db.get_discovery_source("film_reviews", "slug-x") is None

    db.update_discovery_source("film_reviews", "slug-x", last_page=2, total_users=5)
    cached = db.get_discovery_source("film_reviews", "slug-x")

    assert cached["last_page_scraped"] == 2
    assert cached["total_users_found"] == 5
    assert "scraped_at" in cached


def test_pending_queue_stats_and_removal(fresh_db):
    db = fresh_db
    db.init_db()

    db.add_pending_users(["bob", "carol"], "followers", "alice", priority=80)
    db.add_pending_users(["dave"], "popular", "members", priority=50)

    pending = db.get_pending_users()
    assert pending[0]["priority"] == 80
    assert {p["username"] for p in pending} == {"bob", "carol", "dave"}

    stats = db.get_pending_queue_stats()
    assert stats["total"] == 3
    assert stats["breakdown"]["followers"] == 2
    assert stats["breakdown"]["popular"] == 1
    assert stats["avg_priority"] >= 60  # average of priorities present

    db.remove_pending_user("bob")
    remaining = {p["username"] for p in db.get_pending_users()}
    assert "bob" not in remaining


def test_load_films_with_filters(fresh_db):
    db = fresh_db
    db.init_db()

    with db.get_db() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO films (slug, title, year, avg_rating) VALUES (?, ?, ?, ?)",
            [
                ("old-film", "Old", 1999, 3.0),
                ("mid-film", "Mid", 2010, 3.6),
                ("new-film", "New", 2018, 4.0),
            ],
        )

        filtered = cli._load_films_with_filters(conn, min_year=2005, max_year=2015, min_rating=3.5)

    assert set(filtered.keys()) == {"mid-film"}
