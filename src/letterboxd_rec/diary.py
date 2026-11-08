"""Optional diary scraping: the date each film was watched (one row per logged viewing).

The crawl's film lists carry no dates; Letterboxd keeps them only in each user's diary
(`/user/films/diary/`). This is a separate, opt-in backfill stored in `user_diary`, so the main
crawl, its schema and its throughput are untouched.
"""
import logging
import re
import time
from datetime import datetime, timezone
from typing import Callable

logger = logging.getLogger(__name__)

_ROW_RE = re.compile(r'<tr class="diary-entry-row.*?</tr>', re.S)
_VIEWING_RE = re.compile(r'data-viewing-id="(\d+)"')
_SLUG_RE = re.compile(r'data-item-slug="([^"]+)"')
_DATE_RE = re.compile(r'/diary/films/for/(\d{4})/(\d{2})/(\d{2})/')
_REWATCH_RE = re.compile(r'td-rewatch[^"]*')
_NEXT_RE = re.compile(r'<a class="next" href="')
MAX_PAGES = 400
RETRYABLE = (403, 429, 500, 502, 503, 504)   # throttling is time-varying: wait and retry a page
RETRY_WAITS = (5.0, 15.0, 30.0, 60.0)


def parse_diary_page(html: str) -> list[dict]:
    """Entries on one diary page: viewing_id, film_slug, watched_date (YYYY-MM-DD), rewatch."""
    entries = []
    for row in _ROW_RE.findall(html):
        viewing, slug, date = _VIEWING_RE.search(row), _SLUG_RE.search(row), _DATE_RE.search(row)
        if not (viewing and slug and date):
            continue
        rewatch = _REWATCH_RE.search(row)
        entries.append({
            "viewing_id": int(viewing.group(1)),
            "film_slug": slug.group(1),
            "watched_date": "-".join(date.groups()),
            "rewatch": 1 if rewatch and "icon-status-on" in rewatch.group(0) else 0,
        })
    return entries


def has_next_page(html: str) -> bool:
    return bool(_NEXT_RE.search(html))


def scrape_diary(client, username: str, delay: float = 1.5, sleep: Callable[[float], None] = time.sleep,
                 max_pages: int = MAX_PAGES) -> list[dict] | None:
    """Every diary entry for `username`, or None if the diary could not be read (private, blocked).

    A failure on any page returns None rather than a partial list, so a half-read diary is never
    stored as if it were complete.
    """
    entries: list[dict] = []
    for page in range(1, max_pages + 1):
        url = f"https://letterboxd.com/{username}/films/diary/page/{page}/"
        resp = client.get(url)
        for wait in RETRY_WAITS:
            if resp.status_code not in RETRYABLE:
                break
            sleep(wait)
            resp = client.get(url)
        if resp.status_code == 404 and page > 1:
            break
        if resp.status_code != 200:
            logger.warning(f"{username}: diary page {page} returned HTTP {resp.status_code}; skipping user")
            return None
        found = parse_diary_page(resp.text)
        entries.extend(found)
        if not found or not has_next_page(resp.text):
            break
        sleep(delay)
    else:
        logger.warning(f"{username}: diary longer than {max_pages} pages; skipping user")
        return None
    return entries


def save_diary(conn, username: str, entries: list[dict]) -> int:
    """Replace this user's stored diary with `entries` (idempotent re-runs)."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.execute("DELETE FROM user_diary WHERE username = ?", (username,))
    conn.executemany(
        "INSERT OR REPLACE INTO user_diary (username, viewing_id, film_slug, watched_date, rewatch, scraped_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [(username, e["viewing_id"], e["film_slug"], e["watched_date"], e["rewatch"], now) for e in entries],
    )
    # Even a user with an empty diary is recorded, so they are not re-fetched on every run.
    conn.execute("INSERT OR REPLACE INTO user_diary_status (username, entries, scraped_at) VALUES (?, ?, ?)",
                 (username, len(entries), now))
    return len(entries)


def users_needing_diary(conn, limit: int) -> list[str]:
    """Crawled users with no diary fetched yet, biggest histories first."""
    rows = conn.execute(
        "SELECT uf.username FROM user_films uf "
        "LEFT JOIN user_diary_status s ON s.username = uf.username "
        "WHERE s.username IS NULL GROUP BY uf.username ORDER BY COUNT(*) DESC LIMIT ?", (limit,)
    ).fetchall()
    return [r[0] for r in rows]
