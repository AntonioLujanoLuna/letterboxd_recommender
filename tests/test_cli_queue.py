import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from letterboxd_rec import cli


def test_cmd_queue_status_verbose(monkeypatch, caplog):
    monkeypatch.setattr(cli, "init_db", lambda: None)
    monkeypatch.setattr(
        cli,
        "get_pending_queue_stats",
        lambda: {"total": 2, "avg_priority": 75.0, "breakdown": {"followers": 1, "popular": 1}},
    )
    monkeypatch.setattr(
        cli,
        "get_pending_users",
        lambda limit=5: [
            {"username": "u1", "priority": 90, "discovered_from_type": "followers"},
            {"username": "u2", "priority": 70, "discovered_from_type": "popular"},
        ],
    )

    caplog.set_level(logging.INFO)
    cli.cmd_queue_status(SimpleNamespace(verbose=True, limit=5))

    assert "Total pending users: 2" in caplog.text
    assert "followers" in caplog.text


def test_cmd_queue_add_file_not_found(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(cli, "init_db", lambda: None)
    caplog.set_level(logging.ERROR)

    missing = tmp_path / "missing.txt"
    cli.cmd_queue_add(SimpleNamespace(file=str(missing), usernames=None, priority=50))

    assert "File not found" in caplog.text


def test_cmd_queue_add_manual_sanitizes(monkeypatch):
    monkeypatch.setattr(cli, "init_db", lambda: None)
    recorded = {}

    def fake_add(usernames, source_type, source_id, priority):
        recorded["usernames"] = usernames
        recorded["priority"] = priority
        return len(usernames)

    monkeypatch.setattr(cli, "add_pending_users", fake_add)

    cli.cmd_queue_add(SimpleNamespace(file=None, usernames=["User-One", "Bad Name!", ""], priority=70))

    assert recorded["usernames"] == ["user-one", "badname"]
    assert recorded["priority"] == 70


def test_cmd_queue_clear_by_source(monkeypatch, caplog):
    monkeypatch.setattr(cli, "init_db", lambda: None)

    class DummyConn:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, sql, params=()):
            self.last = (sql, params)
            return SimpleNamespace(rowcount=2 if params else 5)

    monkeypatch.setattr(cli, "get_db", lambda: DummyConn())

    caplog.set_level(logging.INFO)
    cli.cmd_queue_clear(SimpleNamespace(source="followers"))

    assert "Removed 2 users from queue" in caplog.text


def test_cmd_discover_refill_adds_users(monkeypatch):
    monkeypatch.setattr(cli, "init_db", lambda: None)
    monkeypatch.setattr(cli, "get_pending_queue_stats", lambda: {"total": 0, "avg_priority": None, "breakdown": {}})
    monkeypatch.setattr(cli, "get_discovery_source", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "update_discovery_source", lambda *args, **kwargs: None)

    added_calls = []

    def fake_add(users, source_type, source_id, priority):
        added_calls.append((users, source_type, source_id, priority))
        return len(users)

    monkeypatch.setattr(cli, "add_pending_users", fake_add)

    class DummyScraper:
        def __init__(self, delay):
            self.closed = False

        def scrape_film_reviewers(self, slug, limit=50):
            return [{"username": f"r{i}"} for i in range(limit)]

        def scrape_popular_members(self, limit=50):
            return [f"p{i}" for i in range(limit)]

        def scrape_followers(self, username, limit=50):
            return [f"f{i}" for i in range(limit)]

        def check_user_activity(self, username):
            return {"film_count": 100, "has_ratings": True}

        def close(self):
            self.closed = True

    created = []
    monkeypatch.setattr(cli, "LetterboxdScraper", lambda delay=1.0: created.append(DummyScraper(delay)) or created[-1])

    args = SimpleNamespace(
        min_queue=5,
        target=3,
        source_refresh_days=7,
        min_films=50,
        sources_file=None,
    )

    cli.cmd_discover_refill(args)

    assert added_calls, "Expected add_pending_users to be called"
    assert len(added_calls[0][0]) == 3  # three reviewers limited by target
    assert created and created[0].closed is True


def test_cmd_discover_continue_empty_queue(monkeypatch, caplog):
    monkeypatch.setattr(cli, "init_db", lambda: None)
    monkeypatch.setattr(cli, "get_pending_queue_stats", lambda: {"total": 0, "breakdown": {}, "avg_priority": None})
    monkeypatch.setattr(cli, "get_pending_users", lambda limit=5: [])

    class DummyScraper:
        def __init__(self, delay=1.0):
            self.closed = False

        def close(self):
            self.closed = True

    created = []
    monkeypatch.setattr(cli, "LetterboxdScraper", lambda delay=1.0: created.append(DummyScraper(delay)) or created[-1])

    caplog.set_level(logging.INFO)
    cli.cmd_discover(SimpleNamespace(continue_mode=True, limit=5, source_refresh_days=7))

    assert "No pending users to scrape" in caplog.text
    assert created and created[0].closed is True


def test_cmd_discover_queue_only_popular(monkeypatch):
    monkeypatch.setattr(cli, "init_db", lambda: None)
    monkeypatch.setattr(cli, "DISCOVERY_PRIORITY_MAP", {"popular": 70})
    monkeypatch.setattr(cli, "get_discovery_source", lambda *args, **kwargs: None)

    queue_stats = {"total": 0, "avg_priority": None, "breakdown": {}}
    monkeypatch.setattr(cli, "get_pending_queue_stats", lambda: queue_stats)

    added_calls = []
    updated_calls = []

    monkeypatch.setattr(cli, "add_pending_users", lambda users, st, sid, pr: added_calls.append((users, st, sid, pr)) or len(users))
    monkeypatch.setattr(cli, "update_discovery_source", lambda st, sid, lp, total: updated_calls.append((st, sid, lp, total)))

    class DummyScraper:
        def __init__(self, delay):
            self.closed = False
            self.calls = []

        def scrape_popular_members(self, limit=50):
            self.calls.append(limit)
            if limit == 50:
                return ["u1", "u2", "u3"]
            return []

        def check_user_activity(self, username):
            return {"film_count": 100, "has_ratings": True}

        def close(self):
            self.closed = True

    created = []
    monkeypatch.setattr(cli, "LetterboxdScraper", lambda delay=1.0: created.append(DummyScraper(delay)) or created[-1])

    args = SimpleNamespace(
        source="popular",
        username=None,
        film_slug=None,
        limit=2,
        queue_only=True,
        continue_mode=False,
        source_refresh_days=7,
        min_films=50,
        dry_run=False,
    )

    cli.cmd_discover(args)

    assert added_calls and added_calls[0][1:] == ("popular", "members", 70)
    assert updated_calls and updated_calls[0][0] == "popular"
    assert created and created[0].closed is True


def test_cmd_queue_add_reads_file(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "init_db", lambda: None)
    recorded = {}

    def fake_add(usernames, source_type, source_id, priority):
        recorded["usernames"] = usernames
        recorded["priority"] = priority
        return len(usernames)

    monkeypatch.setattr(cli, "add_pending_users", fake_add)

    file_path = tmp_path / "names.txt"
    file_path.write_text("User-One\n\nsecond_user\n")

    cli.cmd_queue_add(SimpleNamespace(file=str(file_path), usernames=None, priority=55))

    assert recorded["usernames"] == ["user-one", "second_user"]
    assert recorded["priority"] == 55


def test_cmd_discover_refill_skips_when_queue_full(monkeypatch):
    monkeypatch.setattr(cli, "init_db", lambda: None)
    monkeypatch.setattr(cli, "get_pending_queue_stats", lambda: {"total": 10, "avg_priority": None, "breakdown": {}})

    called = {}
    monkeypatch.setattr(cli, "LetterboxdScraper", lambda delay=1.0: called.setdefault("scraper", True))

    args = SimpleNamespace(min_queue=5, target=3, source_refresh_days=7, min_films=50, sources_file=None)

    cli.cmd_discover_refill(args)

    assert "scraper" not in called, "Should skip refill when queue already above min_queue"

