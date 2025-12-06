import asyncio
from types import SimpleNamespace

import pytest

from letterboxd_rec import cli


class DummyScraper:
    def __init__(self):
        self.scraped_users = []
        self.films = []
        self.lists_scraped = []

    def scrape_user(self, username, existing_slugs=None, stop_on_existing=False):
        self.scraped_users.append(username)
        return []

    def scrape_favorites(self, username):
        return []

    def scrape_user_lists(self, username, limit=50):
        self.lists_scraped.append((username, limit))
        return []

    def scrape_list_films(self, username, list_slug):
        return []

    def scrape_film(self, slug):
        self.films.append(slug)
        return None

    def scrape_followers(self, username, limit=100):
        return []

    def scrape_following(self, username, limit=100):
        return []

    def scrape_popular_members(self, limit=100):
        return []

    def check_user_activity(self, username):
        # Pretend everyone has enough films/ratings
        return {"film_count": 100, "has_ratings": True, "recent_activity": True}

    def close(self):
        pass


@pytest.mark.asyncio
async def test_cmd_scrape_skips_refresh(monkeypatch, tmp_path):
    # Use temporary DB path
    monkeypatch.setenv("LETTERBOXD_DB", str(tmp_path / "test.db"))

    # Avoid real LetterboxdScraper network calls
    dummy = DummyScraper()
    monkeypatch.setattr(cli, "LetterboxdScraper", lambda delay=1.0: dummy)

    # Seed last_scrape to be recent so refresh skip is triggered
    from letterboxd_rec import database

    database.init_db()
    with database.get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO user_films (username, film_slug, scraped_at) VALUES (?, ?, datetime('now'))",
            ("alice", "film-a"),
        )

    args = SimpleNamespace(
        username="alice",
        refresh=7,
        include_lists=False,
        max_lists=1,
        incremental=False,
    )

    cli.cmd_scrape(args)
    # Because last scrape was 0 days ago and refresh=7, we skip scraping
    assert dummy.scraped_users == []


def test_cmd_discover_dry_run(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("LETTERBOXD_DB", str(tmp_path / "test.db"))

    # Dummy scraper that returns a fixed list
    dummy = DummyScraper()

    def fake_followers(username, limit=100):
        return ["u1", "u2"]

    # Patch constructor to return dummy instance
    monkeypatch.setattr(cli, "LetterboxdScraper", lambda delay=1.0: dummy)
    monkeypatch.setattr(cli, "get_discovery_source", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "add_pending_users", lambda users, *args, **kwargs: len(users))
    monkeypatch.setattr(cli, "get_pending_users", lambda limit=50: [{"username": "u1"}, {"username": "u2"}])
    monkeypatch.setattr(cli, "DISCOVERY_PRIORITY_MAP", {"followers": 80})

    # Replace the scrape_followers method on the dummy instance
    dummy.scrape_followers = fake_followers

    args = SimpleNamespace(
        command="discover",
        source="followers",
        username="alice",
        limit=2,
        dry_run=True,
        continue_mode=False,
        source_refresh_days=7,
        min_films=50,
        maintenance=False,
    )

    cli.cmd_discover(args)
    # Dry-run should not enqueue scraping
    assert dummy.scraped_users == []


def test_cmd_discover_queue_only_followers(monkeypatch, tmp_path):
    monkeypatch.setenv("LETTERBOXD_DB", str(tmp_path / "test.db"))

    # Dummy scraper that returns 3 followers and passes activity filter
    dummy = DummyScraper()
    dummy.scrape_followers = lambda username, limit=100: ["u1", "u2", "u3"]
    dummy.check_user_activity = lambda username: {"film_count": 200, "has_ratings": True}

    monkeypatch.setattr(cli, "LetterboxdScraper", lambda delay=1.0: dummy)
    monkeypatch.setattr(cli, "get_discovery_source", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "update_discovery_source", lambda *args, **kwargs: None)
    queue_stats = {"total": 0, "avg_priority": None, "breakdown": {}}
    monkeypatch.setattr(cli, "get_pending_queue_stats", lambda: queue_stats)

    added = {}
    monkeypatch.setattr(
        cli,
        "add_pending_users",
        lambda users, st, sid, pr: added.update({"users": users, "st": st, "sid": sid, "pr": pr}) or len(users),
    )

    args = SimpleNamespace(
        command="discover",
        source="followers",
        username="alice",
        film_slug=None,
        limit=2,
        dry_run=False,
        continue_mode=False,
        queue_only=True,
        source_refresh_days=7,
        min_films=50,
        maintenance=False,
    )

    cli.cmd_discover(args)

    assert added["st"] == "followers"
    assert added["sid"] == "alice"
    assert len(added["users"]) >= 2


def test_cmd_scrape_incremental_passes_existing(monkeypatch, tmp_path):
    monkeypatch.setenv("LETTERBOXD_DB", str(tmp_path / "test.db"))

    from letterboxd_rec import database

    database.init_db()
    with database.get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO user_films (username, film_slug, watched) VALUES (?, ?, ?)",
            ("alice", "film-old", 1),
        )

    dummy = DummyScraper()
    called = {}

    def fake_scrape_user(username, existing_slugs=None, stop_on_existing=False):
        called["existing_slugs"] = existing_slugs
        called["stop_on_existing"] = stop_on_existing
        return []

    dummy.scrape_user = fake_scrape_user
    monkeypatch.setattr(cli, "LetterboxdScraper", lambda delay=1.0: dummy)

    args = SimpleNamespace(
        username="alice",
        refresh=None,
        include_lists=False,
        max_lists=1,
        incremental=True,
    )

    cli.cmd_scrape(args)

    assert called["stop_on_existing"] is True
    assert "film-old" in called["existing_slugs"]