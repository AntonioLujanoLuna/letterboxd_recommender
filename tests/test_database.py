from letterboxd_rec import database


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

