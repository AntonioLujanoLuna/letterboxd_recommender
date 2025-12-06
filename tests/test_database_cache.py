from datetime import datetime, timedelta

from letterboxd_rec import database


def test_profile_cache_round_trip(fresh_db):
    db = fresh_db
    db.init_db()

    profile_data = {"n_films": 1, "genres": {"drama": 1.0}}
    database.save_user_profile("alice", profile_data)

    cached = database.load_cached_profile("alice", max_age_days=7)
    assert cached is not None
    assert cached["genres"]["drama"] == 1.0


def test_profile_cache_invalidates_on_schema_mismatch(fresh_db):
    db = fresh_db
    db.init_db()

    profile_data = {"n_films": 1, "genres": {}}
    database.save_user_profile("bob", profile_data)

    # Force a schema_version mismatch
    with db.get_db() as conn:
        conn.execute(
            "UPDATE user_profiles SET schema_version = schema_version + 1 WHERE username = ?",
            ("bob",),
        )

    assert database.load_cached_profile("bob", max_age_days=7) is None


def test_profile_cache_invalidates_on_age(fresh_db):
    db = fresh_db
    db.init_db()

    profile_data = {"n_films": 1, "genres": {}}
    database.save_user_profile("carol", profile_data)

    old_timestamp = (datetime.now() - timedelta(days=10)).isoformat()
    with db.get_db() as conn:
        conn.execute(
            "UPDATE user_profiles SET updated_at = ? WHERE username = ?",
            (old_timestamp, "carol"),
        )

    assert database.load_cached_profile("carol", max_age_days=7) is None

