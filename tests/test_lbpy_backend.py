import argparse
import asyncio
from contextlib import contextmanager

import httpx
import pytest

from letterboxd_rec import cli, lbpy_backend
from letterboxd_rec.lbpy_backend import (
    AsyncCurlClient,
    AsyncLetterboxdPyScraper,
    CurlClient,
    LetterboxdPyScraper,
)
from letterboxd_rec.scraper import AsyncLetterboxdScraper, LetterboxdScraper, ScrapeIncompleteError

GRID = """<html><body><ul>
<li class="griditem"><div class="react-component" data-item-slug="film-a"></div>
<p class="poster-viewingdata"><span class="rating rated-9"></span><span class="like"></span></p></li>
<li class="griditem"><div class="react-component" data-item-slug="film-b"></div>
<p class="poster-viewingdata"></p></li>
</ul></body></html>"""


class FakeCurlResponse:
    def __init__(self, status=200, text="", headers=None):
        self.status_code = status
        self.text = text
        self.headers = headers or {}


@pytest.fixture
def fake_fetch(monkeypatch):
    """Fake letterboxdpy's curl session. `routes` maps URL substring -> FakeCurlResponse, or an
    Exception to raise; unmatched URLs 404. `routes["calls"]` lists every attempted URL."""
    from letterboxdpy.core.scraper import Scraper

    routes: dict = {}
    calls: list = []

    class Cookies:
        def set(self, *a, **k):
            pass

    class FakeSession:
        cookies = Cookies()

        def get(self, url, headers=None, timeout=None, impersonate=None):
            calls.append(url)
            for needle, result in routes.items():
                if needle in url and needle != "calls":
                    if isinstance(result, list):  # a sequence of outcomes, one per attempt
                        result = result.pop(0) if len(result) > 1 else result[0]
                    if isinstance(result, Exception):
                        raise result
                    return result
            return FakeCurlResponse(404)

    session = FakeSession()
    monkeypatch.setattr(Scraper, "instance", classmethod(lambda cls: session))
    monkeypatch.setattr(lbpy_backend.time, "sleep", lambda s: None)  # no real retry waits in tests
    routes["calls"] = calls
    return routes


def test_curl_client_returns_httpx_response_without_encoding_headers(fake_fetch):
    fake_fetch["/film/x/"] = FakeCurlResponse(
        200, "<p>hi</p>", {"content-encoding": "gzip", "content-length": "999", "retry-after": "7"})
    resp = CurlClient().get("https://letterboxd.com/film/x/")
    assert isinstance(resp, httpx.Response)
    assert resp.status_code == 200 and resp.text == "<p>hi</p>"
    assert resp.headers["retry-after"] == "7"
    assert "content-encoding" not in resp.headers


def test_curl_client_maps_errors_to_httpx(fake_fetch):
    from curl_cffi.requests import errors

    fake_fetch["/slow/"] = errors.RequestsError("Operation timed out after 30000 milliseconds")
    fake_fetch["/down/"] = errors.RequestsError("Failed to connect: connection refused")
    client = CurlClient()
    with pytest.raises(httpx.TimeoutException):
        client.get("https://letterboxd.com/slow/")
    with pytest.raises(httpx.HTTPError):
        client.get("https://letterboxd.com/down/")


def test_status_codes_flow_through(fake_fetch):
    fake_fetch["/gone/"] = FakeCurlResponse(404)
    fake_fetch["/limited/"] = FakeCurlResponse(429, headers={"retry-after": "3"})
    client = CurlClient()
    assert client.get("https://letterboxd.com/gone/").status_code == 404
    r = client.get("https://letterboxd.com/limited/")
    assert r.status_code == 429 and r.headers["retry-after"] == "3"


def test_sync_scraper_parses_through_transport(fake_fetch):
    fake_fetch["/someone/films/"] = FakeCurlResponse(200, GRID)
    scraper = LetterboxdPyScraper(delay=0)
    scraper._current_delay = 0
    try:
        by_slug = {f.film_slug: f for f in scraper.scrape_user("someone")}
    finally:
        scraper.close()
    assert by_slug["film-a"].liked and by_slug["film-a"].rating == 4.5
    assert by_slug["film-b"].watched and not by_slug["film-b"].liked


def test_sync_scraper_blocked_listing_raises(fake_fetch):
    fake_fetch["/someone/films/"] = FakeCurlResponse(403, "denied")
    scraper = LetterboxdPyScraper(delay=0)
    scraper._current_delay = 0
    with pytest.raises(ScrapeIncompleteError):
        scraper.scrape_user("someone")


def test_async_scraper_parses_through_transport(fake_fetch):
    fake_fetch["/someone/films/"] = FakeCurlResponse(200, GRID)

    async def run():
        async with AsyncLetterboxdPyScraper(delay=0, max_concurrent=2) as s:
            assert isinstance(s.client, AsyncCurlClient)
            return await s.scrape_user("someone")

    films = {f.film_slug: f for f in asyncio.run(run())}
    assert set(films) == {"film-a", "film-b"} and films["film-a"].liked


def test_async_film_batch_uses_curl_client_standalone(fake_fetch):
    fake_fetch["/film/bad/"] = FakeCurlResponse(404)

    async def run():
        s = AsyncLetterboxdPyScraper(delay=0, max_concurrent=2)
        return await s.scrape_films_batch(["bad"])

    assert asyncio.run(run()) == []
    assert any("/film/bad/" in u for u in fake_fetch["calls"])


def test_factories_respect_backend():
    old = lbpy_backend.get_backend()
    try:
        lbpy_backend.set_backend("native")
        assert type(lbpy_backend.make_scraper(0)) is LetterboxdScraper
        assert type(lbpy_backend.make_async_scraper(0, 2)) is AsyncLetterboxdScraper
        lbpy_backend.set_backend("letterboxdpy")
        assert isinstance(lbpy_backend.make_scraper(0), LetterboxdPyScraper)
        assert isinstance(lbpy_backend.make_async_scraper(0, 2), AsyncLetterboxdPyScraper)
        with pytest.raises(ValueError):
            lbpy_backend.set_backend("nope")
    finally:
        lbpy_backend.set_backend(old)


# -- crawl -------------------------------------------------------------------------------

def _crawl_args(**over):
    base = dict(
        users=10, seed_film=["parasite"], seed_user=None, popular=False, sources_file=None,
        min_films=50, refill_below=5, queue_target=20, source_refresh_days=7, expand_graph=False,
        chunk=4, batch=10, max_concurrent_users=1, max_concurrent_requests=1, async_delay=0,
        user_delay=0, include_social=False, social_limit=10, max_stalls=2, claim_ttl_minutes=30, max_rps=None, auto_rps=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


@pytest.fixture
def crawl_env(monkeypatch):
    state = {"scraped": 0, "queue": 0, "refills": [], "daemon_targets": [], "per_chunk": 4}

    class Conn:
        def execute(self, *_a, **_k):
            class Cur:
                def fetchone(_s):
                    return [state["scraped"]]
            return Cur()

    @contextmanager
    def fake_get_db(read_only=False):
        yield Conn()

    def fake_refill(ns):
        state["refills"].append(ns.sources)
        state["queue"] += 10

    def fake_daemon(ns):
        state["daemon_targets"].append(ns.target)
        got = min(ns.target, state["per_chunk"], state["queue"])
        state["scraped"] += got
        state["queue"] -= got

    monkeypatch.setattr(cli, "init_db", lambda: None)
    monkeypatch.setattr(cli, "get_db", fake_get_db)
    monkeypatch.setattr(cli, "get_pending_queue_stats", lambda: {"total": state["queue"]})
    monkeypatch.setattr(cli, "cmd_discover_refill", fake_refill)
    monkeypatch.setattr(cli, "cmd_scrape_daemon", fake_daemon)
    return state


def test_crawl_alternates_refill_and_drain_until_target(crawl_env):
    cli.cmd_crawl(_crawl_args(users=10))
    assert crawl_env["scraped"] == 10
    assert crawl_env["refills"][0] == [("film_reviews", "parasite", 100, 2)]
    assert crawl_env["daemon_targets"] == [4, 4, 2]  # last chunk clipped to what's left


def test_crawl_stops_after_repeated_empty_chunks(crawl_env):
    crawl_env["per_chunk"] = 0
    cli.cmd_crawl(_crawl_args(users=10, max_stalls=2))
    assert crawl_env["scraped"] == 0 and len(crawl_env["daemon_targets"]) == 2


def test_crawl_stops_when_no_candidates(crawl_env, monkeypatch):
    monkeypatch.setattr(cli, "cmd_discover_refill", lambda ns: None)  # finds nobody
    cli.cmd_crawl(_crawl_args())
    assert crawl_env["daemon_targets"] == []


def test_async_metadata_helper_fetches_every_slug_in_chunks(monkeypatch):
    seen_chunks, saved = [], []

    class Fake:
        async def scrape_films_batch(self, slugs):
            seen_chunks.append(list(slugs))
            return [type("M", (), {"slug": s})() for s in slugs]

    monkeypatch.setattr(cli, "_persist_metadata_batch", lambda batch: saved.extend(m.slug for m in batch))
    slugs = [f"f{i}" for i in range(25)]
    asyncio.run(cli._scrape_film_metadata_async(Fake(), slugs, max_per_batch=10))
    assert [len(c) for c in seen_chunks] == [10, 10, 5]
    assert saved == slugs


# -- block circuit breaker ---------------------------------------------------------------

class _Clock:
    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def _breaker(**kw):
    clock = _Clock()
    b = lbpy_backend.BlockBreaker(threshold=3, cooldown=60, max_cooldown=200, clock=clock.time,
                                  sleep=clock.sleep, **kw)
    return b, clock


def test_breaker_stays_closed_below_threshold_and_success_resets():
    b, clock = _breaker()
    b.record_block(); b.record_block()
    b.record_ok()
    b.record_block(); b.record_block()
    b.wait()
    assert clock.slept == []


def test_breaker_opens_after_threshold_and_waiters_sleep_the_cooldown():
    b, clock = _breaker()
    for _ in range(3):
        b.record_block()
    b.wait()
    assert clock.slept == [60]
    b.wait()  # cooldown over: no further sleeping
    assert clock.slept == [60]


def test_breaker_half_open_reopens_on_first_block_and_backs_off():
    b, clock = _breaker()
    for _ in range(3):
        b.record_block()
    b.wait()
    b.record_block()  # half-open: a single block re-opens, with a doubled cooldown
    b.wait()
    b.record_block(); b.wait()
    b.record_block(); b.wait()
    assert clock.slept == [60, 120, 200, 200]  # capped at max_cooldown


def test_breaker_success_closes_and_resets_backoff():
    b, clock = _breaker()
    for _ in range(3):
        b.record_block()
    b.wait()
    b.record_ok()
    for _ in range(3):
        b.record_block()
    b.wait()
    assert clock.slept == [60, 60]


def test_curl_client_feeds_breaker_but_ignores_private_pages(fake_fetch):
    b, _ = _breaker()
    client = CurlClient(breaker=b)
    fake_fetch["/private/"] = FakeCurlResponse(403, "<section class='message'>Falken's maze: classified</section>")
    fake_fetch["/blocked/"] = FakeCurlResponse(403, "Access denied")
    fake_fetch["/ok/"] = FakeCurlResponse(200, "<p>ok</p>")

    for _ in range(5):
        client.get("https://letterboxd.com/private/")
    assert b._until == 0.0  # private pages never trip it

    client.get("https://letterboxd.com/blocked/")
    client.get("https://letterboxd.com/blocked/")
    client.get("https://letterboxd.com/ok/")  # success resets the streak
    client.get("https://letterboxd.com/blocked/")
    client.get("https://letterboxd.com/blocked/")
    assert b._until == 0.0
    client.get("https://letterboxd.com/blocked/")  # third in a row
    assert b._until > 0.0


def test_curl_client_waits_while_breaker_open(fake_fetch):
    b, clock = _breaker()
    for _ in range(3):
        b.record_block()
    fake_fetch["/ok/"] = FakeCurlResponse(200, "x")
    CurlClient(breaker=b).get("https://letterboxd.com/ok/")
    assert clock.slept == [60]


# -- transport retry policy ---------------------------------------------------------------

def _attempts(fake, needle):
    return sum(1 for u in fake["calls"] if needle in u)


def test_private_403_is_returned_immediately_without_retries(fake_fetch):
    fake_fetch["/priv/"] = FakeCurlResponse(403, "<section class='message'>Falken's maze: classified</section>")
    resp = CurlClient().get("https://letterboxd.com/priv/")
    assert resp.status_code == 403 and _attempts(fake_fetch, "/priv/") == 1


def test_blocking_403_is_retried_briefly_then_returned(fake_fetch):
    fake_fetch["/blocked/"] = FakeCurlResponse(403, "Access denied")
    resp = CurlClient().get("https://letterboxd.com/blocked/")
    assert resp.status_code == 403
    assert _attempts(fake_fetch, "/blocked/") == CurlClient.ATTEMPTS  # 3, not letterboxdpy's 5 x ~20s


def test_transient_403_then_success_returns_the_page(fake_fetch):
    fake_fetch["/flaky/"] = [FakeCurlResponse(403, "Access denied"), FakeCurlResponse(200, "<p>ok</p>")]
    resp = CurlClient().get("https://letterboxd.com/flaky/")
    assert resp.status_code == 200 and resp.text == "<p>ok</p>" and _attempts(fake_fetch, "/flaky/") == 2


def test_network_errors_are_retried_then_mapped(fake_fetch):
    from curl_cffi.requests import errors

    fake_fetch["/flaky/"] = [errors.RequestsError("connection reset"), FakeCurlResponse(200, "ok")]
    assert CurlClient().get("https://letterboxd.com/flaky/").status_code == 200
    fake_fetch["/dead/"] = errors.RequestsError("connection refused")
    with pytest.raises(httpx.ConnectError):
        CurlClient().get("https://letterboxd.com/dead/")
    assert _attempts(fake_fetch, "/dead/") == CurlClient.ATTEMPTS


def test_every_attempt_is_recorded_in_stats_but_breaker_sees_the_outcome_once(fake_fetch, monkeypatch):
    seen = []
    monkeypatch.setattr(lbpy_backend.STATS, "record", lambda status, secs: seen.append(status))
    b, _ = _breaker()
    fake_fetch["/blocked/"] = FakeCurlResponse(403, "Access denied")
    CurlClient(breaker=b).get("https://letterboxd.com/blocked/")
    assert seen == [403, 403, 403]
    assert b._consecutive == 1  # one blocked *request*, not three
