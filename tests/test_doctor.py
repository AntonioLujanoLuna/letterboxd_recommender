import sqlite3

from letterboxd_rec.doctor import ERROR, INFO, OK, WARN, format_report, run_checks, worst_level


def _db():
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE films (slug TEXT PRIMARY KEY, year INTEGER, tmdb_id INTEGER, imdb_id TEXT);
        CREATE TABLE user_films (username TEXT, film_slug TEXT, PRIMARY KEY (username, film_slug));
        CREATE TABLE film_genres (film_slug TEXT, genre TEXT);
        CREATE TABLE film_directors (film_slug TEXT, director TEXT);
        CREATE TABLE pending_users (username TEXT PRIMARY KEY, attempts INTEGER DEFAULT 0);
        CREATE TABLE user_follows (follower TEXT, followee TEXT);
    """)
    return conn


def by_check(findings):
    return {f.check: f for f in findings}


def test_clean_database_reports_no_problems():
    conn = _db()
    conn.execute("INSERT INTO films VALUES ('a', 2000, 1, 'tt1')")
    conn.executemany("INSERT INTO user_films VALUES (?, 'a')", [(f"u{i}",) for i in range(3)])
    conn.executemany("INSERT INTO user_follows VALUES (?, 'zz')", [(f"u{i}",) for i in range(3)])
    findings = run_checks(conn)
    assert worst_level(findings) in (OK, INFO)
    assert by_check(findings)["integrity"].level == OK


def test_problems_are_found_and_come_with_a_fix():
    conn = _db()
    conn.execute("INSERT INTO films VALUES ('bare', NULL, NULL, NULL)")
    conn.execute("INSERT INTO films VALUES ('ok', 1999, NULL, NULL)")
    conn.executemany("INSERT INTO user_films VALUES (?, ?)",
                     [("u1", "bare"), ("u1", "ghost"), ("u1", "ok"), ("u2", "ok")])
    conn.execute("INSERT INTO film_genres VALUES ('gone', 'drama')")
    conn.execute("INSERT INTO pending_users VALUES ('stuck', 3)")
    conn.execute("INSERT INTO pending_users VALUES ('u1', 0)")        # already scraped
    f = by_check(run_checks(conn))
    assert f["films without a row"].level == WARN and "1 " in f["films without a row"].detail
    assert f["films without metadata"].level == WARN
    assert f["orphaned film_genres"].level == WARN
    assert f["parked users"].level == WARN
    assert f["queued but already scraped"].level == WARN
    assert f["users without social edges"].level == INFO
    assert f["films without a row"].fix
    assert worst_level(run_checks(conn)) == WARN
    assert "fix:" in format_report(run_checks(conn))


def test_doctor_never_modifies_the_database():
    conn = _db()
    conn.execute("INSERT INTO pending_users VALUES ('stuck', 3)")
    before = conn.total_changes
    run_checks(conn)
    assert conn.total_changes == before
