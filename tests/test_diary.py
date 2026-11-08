import sqlite3
from types import SimpleNamespace

from letterboxd_rec.diary import (
    has_next_page,
    parse_diary_page,
    save_diary,
    scrape_diary,
    users_needing_diary,
)


def _row(viewing, slug, ymd, rewatch=False):
    y, m, d = ymd.split("-")
    status = "icon-status-on" if rewatch else "icon-status-off"
    return (
        f'<tr class="diary-entry-row viewing-poster-container" data-viewing-id="{viewing}" data-owner="u">'
        f'<td class="col-monthdate"><a class="month" href="/u/diary/films/for/{y}/{m}/">x</a></td>'
        f'<td class="col-daydate"><a class="daydate" href="/u/diary/films/for/{y}/{m}/{d}/">{d}</a></td>'
        f'<td><div data-item-slug="{slug}" data-item-link="/film/{slug}/"></div></td>'
        f'<td class="td-rewatch {status}"></td></tr>'
    )


def _page(rows, nxt=False):
    more = '<a class="next" href="/u/diary/films/page/2/">Older</a>' if nxt else ""
    return "<table>" + "".join(rows) + "</table>" + more


class FakeClient:
    def __init__(self, pages, flaky=0):
        self.pages, self.calls, self.flaky = pages, [], flaky

    def get(self, url):
        self.calls.append(url)
        if self.flaky > 0:
            self.flaky -= 1
            return SimpleNamespace(status_code=403, text="")
        n = int(url.rstrip("/").rsplit("/", 1)[1])
        status, text = self.pages.get(n, (404, ""))
        return SimpleNamespace(status_code=status, text=text)


def test_parse_diary_page_reads_slug_date_viewing_and_rewatch():
    html = _page([_row(11, "perfect-blue", "2026-11-07"), _row(12, "heat-1995", "2026-10-02", rewatch=True)])
    assert parse_diary_page(html) == [
        {"viewing_id": 11, "film_slug": "perfect-blue", "watched_date": "2026-11-07", "rewatch": 0},
        {"viewing_id": 12, "film_slug": "heat-1995", "watched_date": "2026-10-02", "rewatch": 1},
    ]
    assert parse_diary_page("<html>nothing</html>") == []


def test_has_next_page_only_when_an_older_link_exists():
    assert has_next_page(_page([_row(1, "a", "2026-01-01")], nxt=True))
    assert not has_next_page(_page([_row(1, "a", "2026-01-01")]))


def test_scrape_diary_follows_pages_until_the_last():
    client = FakeClient({1: (200, _page([_row(1, "a", "2026-01-02")], nxt=True)),
                         2: (200, _page([_row(2, "b", "2026-01-01")]))})
    slept = []
    entries = scrape_diary(client, "u", delay=0.7, sleep=slept.append)
    assert [e["film_slug"] for e in entries] == ["a", "b"]
    assert len(client.calls) == 2 and slept == [0.7]


def test_scrape_diary_returns_none_instead_of_a_partial_diary_on_failure():
    client = FakeClient({1: (200, _page([_row(1, "a", "2026-01-02")], nxt=True)), 2: (403, "")})
    assert scrape_diary(client, "u", sleep=lambda s: None) is None
    assert scrape_diary(FakeClient({1: (403, "")}), "u", sleep=lambda s: None) is None


def test_save_diary_is_idempotent_and_records_empty_diaries():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE user_films (username TEXT, film_slug TEXT);"
        "CREATE TABLE user_diary (username TEXT NOT NULL, viewing_id INTEGER NOT NULL, film_slug TEXT NOT NULL, "
        "watched_date TEXT NOT NULL, rewatch INTEGER DEFAULT 0, scraped_at TEXT, PRIMARY KEY (username, viewing_id));"
        "CREATE TABLE user_diary_status (username TEXT PRIMARY KEY, entries INTEGER NOT NULL, scraped_at TEXT);"
    )
    entries = [{"viewing_id": 1, "film_slug": "a", "watched_date": "2026-01-01", "rewatch": 0}]
    save_diary(conn, "u", entries)
    save_diary(conn, "u", entries)  # re-run replaces rather than duplicates
    assert conn.execute("SELECT COUNT(*) FROM user_diary").fetchone()[0] == 1
    save_diary(conn, "empty", [])
    assert conn.execute("SELECT entries FROM user_diary_status WHERE username='empty'").fetchone()[0] == 0

    conn.executemany("INSERT INTO user_films VALUES (?, ?)",
                     [("u", "x"), ("big", "x"), ("big", "y"), ("empty", "x"), ("new", "x")])
    # u and empty already fetched; biggest remaining first
    assert users_needing_diary(conn, 10) == ["big", "new"]
    assert users_needing_diary(conn, 1) == ["big"]


def test_scrape_diary_retries_a_blocked_page_before_giving_up():
    client = FakeClient({1: (200, _page([_row(1, "a", "2026-01-02")]))}, flaky=2)
    slept = []
    entries = scrape_diary(client, "u", sleep=slept.append)
    assert [e["film_slug"] for e in entries] == ["a"] and slept == [5.0, 15.0]
