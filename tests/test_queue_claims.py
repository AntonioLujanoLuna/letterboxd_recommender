import argparse
import asyncio
import importlib
import time

import pytest

from letterboxd_rec.scraper import FilmInteraction


@pytest.fixture
def db(fresh_db):
    fresh_db.init_db()
    return fresh_db


def _queue(db, *names):
    db.add_pending_users(list(names), "test", "t", priority=50)


def _row(db, name):
    with db.get_db(read_only=True) as conn:
        r = conn.execute("SELECT * FROM pending_users WHERE username = ?", (name,)).fetchone()
        return dict(r) if r else None


def test_claim_is_exclusive_and_keeps_row_until_removed(db):
    _queue(db, "a", "b")
    first, second = db.claim_pending_user(), db.claim_pending_user()
    assert {first["username"], second["username"]} == {"a", "b"}
    assert db.claim_pending_user() is None  # nothing left to claim
    assert _row(db, "a") is not None  # still queued while in flight
    assert db.get_pending_users() == []  # but not offered to other consumers
    assert db.get_pending_queue_stats()["total"] == 2


def test_claim_respects_priority(db):
    db.add_pending_users(["low"], "t", "t", priority=10)
    db.add_pending_users(["high"], "t", "t", priority=90)
    assert db.claim_pending_user()["username"] == "high"


def test_release_requeues_and_counts_attempts(db):
    _queue(db, "a")
    db.claim_pending_user()
    db.release_pending_user("a")
    row = _row(db, "a")
    assert row["claimed_at"] is None and row["attempts"] == 1
    db.claim_pending_user()
    db.release_pending_user("a", count_attempt=False)
    assert _row(db, "a")["attempts"] == 1


def test_user_is_parked_after_max_attempts(db):
    _queue(db, "a")
    for _ in range(db.MAX_QUEUE_ATTEMPTS):
        assert db.claim_pending_user()["username"] == "a"
        db.release_pending_user("a")
    assert db.claim_pending_user() is None
    stats = db.get_pending_queue_stats()
    assert stats["total"] == 0 and stats["parked"] == 1
    assert _row(db, "a") is not None  # kept for inspection, not silently dropped


def test_stale_claims_are_released_fresh_ones_kept(db):
    _queue(db, "old", "new")
    db.claim_pending_user()
    db.claim_pending_user()
    with db.get_db() as conn:
        conn.execute("UPDATE pending_users SET claimed_at = ? WHERE username = 'old'",
                     (time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 7200)),))
    assert db.release_stale_claims(30 * 60) == 1
    assert _row(db, "old")["claimed_at"] is None and _row(db, "new")["claimed_at"] is not None
    assert db.release_stale_claims(0) == 1  # 0 releases everything


def test_migration_adds_columns_to_existing_table(fresh_db):
    db = fresh_db
    db.init_db()
    with db.get_db() as conn:
        conn.execute("DROP TABLE pending_users")
        conn.execute("""CREATE TABLE pending_users (username TEXT PRIMARY KEY, discovered_from_type TEXT NOT NULL,
            discovered_from_id TEXT NOT NULL, discovered_at TEXT NOT NULL, priority INTEGER DEFAULT 50)""")
        conn.execute("INSERT INTO pending_users VALUES ('legacy','t','t','2026-01-01',50)")
        conn.execute("DELETE FROM schema_migrations WHERE version = '0005_pending_user_claims'")
    db.run_versioned_migrations()
    assert db.claim_pending_user()["username"] == "legacy"


class _FakeAsyncScraper:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def scrape_user(self, username, **_):
        if username.startswith("bad"):
            raise RuntimeError("blocked")
        return [FilmInteraction(f"film-{username}", 4.0, True, False, False)]

    async def scrape_films_batch(self, slugs):
        return []


def _daemon_args(**over):
    base = dict(target=None, delay=0, user_delay=0, wait_for_queue=False, wait_seconds=1, batch=10,
                remove_on_error=False, max_concurrent_users=1, max_concurrent_requests=1, async_delay=0,
                include_social=False, social_limit=1, claim_ttl_minutes=30, max_rps=None)
    base.update(over)
    return argparse.Namespace(**base)


@pytest.fixture
def cli(db, monkeypatch):
    from letterboxd_rec import cli as cli_mod

    cli_mod = importlib.reload(cli_mod)  # rebind to the reloaded (temp-DB) database module
    monkeypatch.setattr(cli_mod, "make_async_scraper", lambda **_: _FakeAsyncScraper())
    return cli_mod


def test_daemon_keeps_failed_users_queued_then_parks_them(db, cli):
    _queue(db, "good1", "bad1", "good2")
    asyncio.run(cli._cmd_scrape_daemon_async(_daemon_args()))
    assert _row(db, "good1") is None and _row(db, "good2") is None  # succeeded -> removed
    bad = _row(db, "bad1")
    assert bad is not None and bad["claimed_at"] is None and bad["attempts"] == 1

    for _ in range(db.MAX_QUEUE_ATTEMPTS - 1):
        asyncio.run(cli._cmd_scrape_daemon_async(_daemon_args()))
    assert _row(db, "bad1")["attempts"] == db.MAX_QUEUE_ATTEMPTS
    assert db.get_pending_queue_stats()["parked"] == 1


def test_daemon_remove_on_error_still_drops_user(db, cli):
    _queue(db, "bad1")
    asyncio.run(cli._cmd_scrape_daemon_async(_daemon_args(remove_on_error=True)))
    assert _row(db, "bad1") is None


def test_daemon_recovers_users_claimed_by_a_crashed_run(db, cli):
    _queue(db, "good1")
    db.claim_pending_user()  # simulate a run that died holding the claim
    with db.get_db() as conn:
        conn.execute("UPDATE pending_users SET claimed_at = '2020-01-01T00:00:00'")
    asyncio.run(cli._cmd_scrape_daemon_async(_daemon_args()))
    assert _row(db, "good1") is None
    with db.get_db(read_only=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM user_films WHERE username='good1'").fetchone()[0] == 1


def test_sync_metadata_helper_fetches_every_slug_in_chunks(monkeypatch):
    from letterboxd_rec import cli as cli_mod

    saved, scraped = [], []

    class Scraper:
        def scrape_film(self, slug):
            scraped.append(slug)
            return type("M", (), {"slug": slug})()

    monkeypatch.setattr("letterboxd_rec.cli.utils.scrapers._persist_metadata_batch", lambda batch: saved.append([m.slug for m in batch]))
    slugs = [f"f{i}" for i in range(25)]
    cli_mod._scrape_film_metadata(Scraper(), slugs, max_per_batch=10, use_async=False)
    assert scraped == slugs
    assert [len(c) for c in saved] == [10, 10, 5]  # saved incrementally, nothing dropped
