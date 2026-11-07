"""Pagination integrity, parallel page fetching, request pacing, metrics and film-fetch dedup."""
import asyncio
import time

import httpx
import pytest
from selectolax.parser import HTMLParser

from letterboxd_rec import cli, scraper as sc
from letterboxd_rec.scrape_stats import ScrapeStats, TimedAsyncClient, TimedClient
from letterboxd_rec.scraper import (
    AsyncLetterboxdScraper,
    FilmInteraction,
    LetterboxdScraper,
    ScrapeIncompleteError,
)

PAGE = 72  # what Letterboxd really serves per page


def grid(slugs):
    items = "".join(
        f'<li class="griditem"><div class="react-component" data-item-slug="{s}"></div>'
        f'<p class="poster-viewingdata"><span class="rating rated-8"></span></p></li>'
        for s in slugs
    )
    return HTMLParser(f"<html><body><ul>{items}</ul></body></html>")


def make_site(total_films: int, watchlist: int = 0, user: str = "u"):
    """Fake Letterboxd: films/watchlist listings paged by PAGE; pages past the end are empty."""
    films = [f"film-{i}" for i in range(total_films)]
    wl = [f"wish-{i}" for i in range(watchlist)]

    def page_of(url):
        kind = "films" if "/films/page/" in url else "watchlist"
        n = int(url.rstrip("/").rsplit("/", 1)[1])
        data = films if kind == "films" else wl
        return grid(data[(n - 1) * PAGE:n * PAGE])

    return films, wl, page_of


@pytest.fixture
def fast_sleep(monkeypatch):
    real = asyncio.sleep

    async def quick(delay, *a, **k):
        await real(0)

    monkeypatch.setattr(asyncio, "sleep", quick)
    monkeypatch.setattr(time, "sleep", lambda s: None)


# -- async scraping ----------------------------------------------------------------------

def test_async_scrape_user_collects_all_pages_in_parallel(monkeypatch):
    films, wl, page_of = make_site(200, watchlist=80)
    in_flight = {"now": 0, "max": 0}
    requested = []

    async def fake_get(url):
        requested.append(url)
        in_flight["now"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["now"])
        await asyncio.sleep(0.01)
        in_flight["now"] -= 1
        return page_of(url)

    s = AsyncLetterboxdScraper(delay=0)
    monkeypatch.setattr(s, "_get_with_soft_block_recovery", fake_get)
    result = asyncio.run(s.scrape_user("u"))

    got = {f.film_slug: f for f in result}
    assert set(got) == set(films) | set(wl)
    assert got["film-0"].watched and got["wish-0"].watchlisted and not got["wish-0"].watched
    assert in_flight["max"] > 1  # pages really were fetched concurrently
    # 200 films = 3 pages; waves may fetch a few empty pages past the end, but not many
    film_pages = [u for u in requested if "/films/" in u]
    assert 3 <= len(film_pages) <= 3 + s.PAGE_WAVE


def test_small_profile_costs_one_request_per_listing(monkeypatch):
    films, wl, page_of = make_site(10, watchlist=0)
    requested = []

    async def fake_get(url):
        requested.append(url)
        return page_of(url)

    s = AsyncLetterboxdScraper(delay=0)
    monkeypatch.setattr(s, "_get_with_soft_block_recovery", fake_get)
    asyncio.run(s.scrape_user("u"))
    assert len(requested) == 2  # page 1 of films + page 1 of watchlist, no speculative extras


def test_async_failed_middle_page_raises_instead_of_truncating(monkeypatch, fast_sleep):
    """Regression: a transient failure on page 2 used to be read as 'end of list', silently
    keeping 72 of 505 films for a user."""
    films, wl, page_of = make_site(300)

    async def fake_get(url):
        if "/films/page/2/" in url:
            s._failed.add(url)  # what _get does for a transient failure
            return None
        return page_of(url)

    s = AsyncLetterboxdScraper(delay=0)
    monkeypatch.setattr(s, "_get_with_soft_block_recovery", fake_get)
    with pytest.raises(ScrapeIncompleteError):
        asyncio.run(s.scrape_user("u"))
    with pytest.raises(ScrapeIncompleteError):
        asyncio.run(s.scrape_user_smart("u"))


def test_async_transient_page_failure_is_retried(monkeypatch, fast_sleep):
    films, wl, page_of = make_site(150)
    attempts = {"n": 0}

    async def fake_get(url):
        if "/films/page/2/" in url:
            attempts["n"] += 1
            if attempts["n"] < 3:
                s._failed.add(url)
                return None
            s._failed.discard(url)
        return page_of(url)

    s = AsyncLetterboxdScraper(delay=0)
    monkeypatch.setattr(s, "_get_with_soft_block_recovery", fake_get)
    result = asyncio.run(s.scrape_user("u"))
    assert {f.film_slug for f in result} == set(films)
    assert attempts["n"] == 3


def test_async_watchlist_failure_raises_but_private_watchlist_is_fine(monkeypatch, fast_sleep):
    films, wl, page_of = make_site(5, watchlist=5)

    async def private(url):
        return None if "/watchlist/" in url else page_of(url)  # 403 'classified': not marked failed

    s = AsyncLetterboxdScraper(delay=0)
    monkeypatch.setattr(s, "_get_with_soft_block_recovery", private)
    assert {f.film_slug for f in asyncio.run(s.scrape_user("u"))} == set(films)

    async def failing(url):
        if "/watchlist/" in url:
            s._failed.add(url)
            return None
        return page_of(url)

    monkeypatch.setattr(s, "_get_with_soft_block_recovery", failing)
    with pytest.raises(ScrapeIncompleteError):
        asyncio.run(s.scrape_user("u"))


def test_watchlist_that_repeats_past_the_end_terminates(monkeypatch):
    """Letterboxd was seen serving the same watchlist content for pages beyond the end."""
    films, _, page_of = make_site(3)
    full = [f"wish-{i}" for i in range(PAGE)]

    async def fake_get(url):
        if "/watchlist/" in url:
            return grid(full)  # every page identical and "full"
        return page_of(url)

    s = AsyncLetterboxdScraper(delay=0)
    monkeypatch.setattr(s, "_get_with_soft_block_recovery", fake_get)
    result = asyncio.run(asyncio.wait_for(s.scrape_user("u"), timeout=5))
    assert len([f for f in result if f.watchlisted]) == PAGE


def test_async_smart_variant_also_reads_every_page(monkeypatch):
    films, _, page_of = make_site(180)

    async def fake_get(url):
        return page_of(url)

    s = AsyncLetterboxdScraper(delay=0)
    monkeypatch.setattr(s, "_get_with_soft_block_recovery", fake_get)
    assert {f.film_slug for f in asyncio.run(s.scrape_user_smart("u", known_film_count=180))} == set(films)


def test_incremental_mode_still_stops_early(monkeypatch):
    films, _, page_of = make_site(300)

    async def fake_get(url):
        return page_of(url)

    s = AsyncLetterboxdScraper(delay=0)
    monkeypatch.setattr(s, "_get_with_soft_block_recovery", fake_get)
    result = asyncio.run(s.scrape_user("u", existing_slugs=set(films), stop_on_existing=True))
    assert [f for f in result if f.watched] == []  # everything already known


# -- sync scraping -----------------------------------------------------------------------

def test_sync_failed_middle_page_raises(monkeypatch, fast_sleep):
    films, _, page_of = make_site(300)
    lb = LetterboxdScraper(delay=0)

    def fake_get(url):
        if "/films/page/2/" in url:
            lb._failed.add(url)
            return None
        return page_of(url)

    monkeypatch.setattr(lb, "_get_with_soft_block_recovery", fake_get)
    try:
        with pytest.raises(ScrapeIncompleteError):
            lb.scrape_user("u")
        with pytest.raises(ScrapeIncompleteError):
            lb.scrape_user_smart("u")
    finally:
        lb.close()


def test_sync_scrape_user_reads_all_pages(monkeypatch):
    films, wl, page_of = make_site(150, watchlist=5)
    lb = LetterboxdScraper(delay=0)
    monkeypatch.setattr(lb, "_get_with_soft_block_recovery", page_of)
    try:
        assert {f.film_slug for f in lb.scrape_user("u")} == set(films) | set(wl)
    finally:
        lb.close()


def test_sync_get_marks_transient_failures_but_not_404s_or_private(monkeypatch):
    lb = LetterboxdScraper(delay=0)
    lb._current_delay = 0

    def respond(status, text=""):
        monkeypatch.setattr(lb.client._client, "get", lambda url: httpx.Response(
            status, text=text, request=httpx.Request("GET", url)))

    try:
        respond(404)
        assert lb._get("https://letterboxd.com/x/") is None and not lb._failed
        respond(403, "Falken's maze: classified")
        assert lb._get("https://letterboxd.com/x/") is None and not lb._failed
        respond(403, "Access denied")
        assert lb._get("https://letterboxd.com/x/") is None and "https://letterboxd.com/x/" in lb._failed
        respond(200, "<p>ok</p>")
        assert lb._get("https://letterboxd.com/x/") is not None and not lb._failed
    finally:
        lb.close()


# -- pacing ------------------------------------------------------------------------------

def test_pacing_spaces_request_starts_by_rps_cap():
    async def run():
        s = AsyncLetterboxdScraper(delay=0, max_concurrent=10, max_rps=50)
        starts = []

        async def one():
            await s._pace()
            starts.append(time.monotonic())

        await asyncio.gather(*(one() for _ in range(10)))
        return sorted(starts)

    starts = asyncio.run(run())
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert min(gaps) >= 0.010  # 1/50s = 0.02s; generous slack for coarse Windows timers
    assert starts[-1] - starts[0] >= 9 / 50 * 0.9


def test_no_cap_and_no_delay_means_no_waiting():
    async def run():
        s = AsyncLetterboxdScraper(delay=0, max_concurrent=5)
        t = time.monotonic()
        await asyncio.gather(*(s._pace() for _ in range(200)))
        return time.monotonic() - t

    assert asyncio.run(run()) < 0.2


def test_pace_interval_uses_adaptive_delay_when_slower_than_cap():
    s = AsyncLetterboxdScraper(delay=0.2, max_concurrent=4, max_rps=100)
    assert s._pace_interval() == pytest.approx(0.05)  # delay/concurrency beats the 0.01s cap
    s.delay = 2.0  # as after repeated 429s
    assert s._pace_interval() == pytest.approx(0.5)


# -- metrics -----------------------------------------------------------------------------

class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_stats_window_report_counts_rate_latency_and_problems():
    clock = _Clock()
    st = ScrapeStats(clock=clock)
    for i in range(20):
        clock.now += 0.5
        st.record(200, 0.1 * (i + 1))
    st.record(403, 0.2)
    st.record(429, 0.2)
    st.record(None, 0.2)
    st.record(404, 0.2)  # a missing page is a normal answer, not a problem
    clock.now = 60.0
    report = st.maybe_report(60)
    assert "24 requests" in report and "0.4 req/s" in report
    assert "403:1" in report and "429:1" in report and "err:1" in report and "404" not in report
    assert "p50" in report and "p95" in report


def test_stats_report_respects_interval_and_empty_windows():
    clock = _Clock()
    st = ScrapeStats(clock=clock)
    st.record(200, 0.1)
    clock.now = 10.0
    assert st.maybe_report(60) is None  # too soon
    clock.now = 61.0
    assert st.maybe_report(60) is not None
    clock.now = 130.0
    assert st.maybe_report(60) is None  # nothing happened in that window
    assert "1 in" in st.summary()


def test_timed_clients_record_status_and_errors():
    st = ScrapeStats()
    seen = []
    st.record = lambda status, seconds: seen.append(status)

    class Sync:
        def get(self, url):
            if "boom" in url:
                raise httpx.ConnectError("x")
            return httpx.Response(429, request=httpx.Request("GET", url))

        def close(self):
            seen.append("closed")

    client = TimedClient(Sync(), st)
    assert client.get("http://a").status_code == 429
    with pytest.raises(httpx.ConnectError):
        client.get("http://boom")
    client.close()  # passes through to the wrapped client
    assert seen == [429, None, "closed"]

    class Async:
        async def get(self, url):
            return httpx.Response(200, request=httpx.Request("GET", url))

    asyncio.run(TimedAsyncClient(Async(), st).get("http://a"))
    assert seen[-1] == 200


# -- film fetch dedup in the daemon ------------------------------------------------------

def test_daemon_does_not_fetch_the_same_film_twice_across_concurrent_users(fresh_db, monkeypatch):
    import argparse
    import importlib

    fresh_db.init_db()
    fresh_db.add_pending_users(["a", "b", "c"], "t", "t", priority=50)
    cli_mod = importlib.reload(cli)

    fetched = []

    class FakeAsync:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def scrape_user(self, username, **_):
            await asyncio.sleep(0)
            return [FilmInteraction(s, 4.0, True, False, False) for s in ("shared-1", "shared-2", f"own-{username}")]

        async def scrape_films_batch(self, slugs):
            fetched.extend(slugs)
            await asyncio.sleep(0.02)  # long enough for the other users to overlap
            return [type("M", (), {"slug": s})() for s in slugs if s != "own-c"]  # one film fails

    monkeypatch.setattr(cli_mod, "make_async_scraper", lambda **_: FakeAsync())
    monkeypatch.setattr("letterboxd_rec.cli.utils.scrapers._persist_metadata_batch", lambda batch: None)
    args = argparse.Namespace(
        target=None, delay=0, user_delay=0, wait_for_queue=False, wait_seconds=1, batch=10,
        remove_on_error=False, max_concurrent_users=3, max_concurrent_requests=3, async_delay=0,
        include_social=False, social_limit=1, claim_ttl_minutes=30, max_rps=None)
    asyncio.run(cli_mod._cmd_scrape_daemon_async(args))

    assert sorted(fetched) == sorted(set(fetched))  # no film requested twice
    assert {"shared-1", "shared-2", "own-a", "own-b", "own-c"} == set(fetched)


def test_metadata_helper_reports_which_slugs_were_saved(monkeypatch):
    class Fake:
        async def scrape_films_batch(self, slugs):
            return [type("M", (), {"slug": s})() for s in slugs if s != "bad"]

    monkeypatch.setattr("letterboxd_rec.cli.utils.scrapers._persist_metadata_batch", lambda batch: None)
    saved = asyncio.run(cli._scrape_film_metadata_async(Fake(), ["a", "bad", "b"], max_per_batch=2))
    assert saved == {"a", "b"}


# -- adaptive rate (AIMD) ----------------------------------------------------------------

from letterboxd_rec.rate_control import AimdRate  # noqa: E402


class _Tick:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_aimd_ramps_up_only_after_clean_windows_and_stops_at_ceiling():
    t = _Tick()
    r = AimdRate(start=6, ceiling=8, step=1.0, increase_after=20, clock=t)
    r.on_ok()
    assert r.rps == 6  # too soon
    t.now = 20; r.on_ok(); assert r.rps == 7
    t.now = 25; r.on_ok(); assert r.rps == 7  # needs another full clean window
    t.now = 40; r.on_ok(); assert r.rps == 8
    t.now = 100; r.on_ok(); assert r.rps == 8  # ceiling


def test_aimd_halves_on_throttle_floors_debounces_and_pauses_probing():
    t = _Tick()
    r = AimdRate(start=8, ceiling=12, floor=2, increase_after=20, debounce=10, clock=t)
    t.now = 30; r.on_throttle(); assert r.rps == 4
    t.now = 31; r.on_throttle(); assert r.rps == 4  # burst of 429s counts once
    t.now = 45; r.on_throttle(); assert r.rps == 2  # floored at 2
    t.now = 60; r.on_throttle(); assert r.rps == 2
    t.now = 70; r.on_ok(); assert r.rps == 2  # a throttle resets the clean-window clock
    t.now = 85; r.on_ok(); assert r.rps == 3


def test_aimd_validates_bounds():
    with pytest.raises(ValueError):
        AimdRate(start=10, ceiling=5)


def test_scraper_uses_adaptive_rate_for_pacing_and_reacts_to_throttles():
    s = AsyncLetterboxdScraper(delay=0, max_concurrent=8, max_rps=6, rps_ceiling=12)
    assert s._pace_interval() == pytest.approx(1 / 6)
    s._note_throttle()  # a 429 / blocking 403
    assert s._pace_interval() == pytest.approx(1 / 3)  # rate halved -> spacing doubled
    fixed = AsyncLetterboxdScraper(delay=0, max_concurrent=8, max_rps=6)
    fixed._note_throttle()  # without --auto-rps nothing changes
    assert fixed._pace_interval() == pytest.approx(1 / 6)
    # a ceiling that isn't above the start disables probing
    assert AsyncLetterboxdScraper(max_rps=6, rps_ceiling=6)._aimd is None


def test_429_and_blocking_403_reach_the_rate_controller(monkeypatch, fast_sleep):
    s = AsyncLetterboxdScraper(delay=0, max_concurrent=2, max_rps=6, rps_ceiling=12)
    calls = []
    monkeypatch.setattr(s, "_note_throttle", lambda: calls.append(1))

    class Client:
        def __init__(self, responses):
            self.responses = list(responses)

        async def get(self, url):
            return self.responses.pop(0)

    def resp(status, text=""):
        return httpx.Response(status, text=text, request=httpx.Request("GET", "http://x"),
                              headers={"retry-after": "0"})

    async def run(responses, fn):
        s.client = Client(responses)
        return await fn()

    asyncio.run(run([resp(429), resp(200, "<html></html>")], lambda: s._get("http://x/a")))
    assert len(calls) == 1
    asyncio.run(run([resp(403, "Access denied")], lambda: s._get("http://x/b")))
    assert len(calls) == 2
    asyncio.run(run([resp(403, "Falken's maze: classified")], lambda: s._get("http://x/c")))
    assert len(calls) == 2  # a private page is not a throttle signal


# -- daemon: stop at the target without wasting in-flight users ---------------------------

def test_daemon_target_never_cancels_users_mid_scrape(fresh_db, monkeypatch):
    import argparse
    import importlib

    fresh_db.init_db()
    fresh_db.add_pending_users([f"u{i}" for i in range(6)], "t", "t", priority=50)
    cli_mod = importlib.reload(cli)
    started, finished = [], []

    class FakeAsync:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def scrape_user(self, username, **_):
            started.append(username)
            await asyncio.sleep(0.02 * (1 + len(started) % 3))  # uneven durations
            finished.append(username)
            return [FilmInteraction(f"film-{username}", 4.0, True, False, False)]

        async def scrape_films_batch(self, slugs):
            return []

    monkeypatch.setattr(cli_mod, "make_async_scraper", lambda **_: FakeAsync())
    monkeypatch.setattr("letterboxd_rec.cli.utils.scrapers._persist_metadata_batch", lambda batch: None)
    args = argparse.Namespace(
        target=3, delay=0, user_delay=0, wait_for_queue=True, wait_seconds=1, batch=10,
        remove_on_error=False, max_concurrent_users=2, max_concurrent_requests=2, async_delay=0,
        include_social=False, social_limit=1, claim_ttl_minutes=30, max_rps=None)
    asyncio.run(asyncio.wait_for(cli_mod._cmd_scrape_daemon_async(args), timeout=10))  # wait_for_queue must not hang

    assert len(started) == 3 and sorted(started) == sorted(finished)  # exactly the target, none cancelled
    assert fresh_db.get_pending_queue_stats()["total"] == 3  # the other three stay queued, unclaimed
    assert len(fresh_db.get_pending_users()) == 3


# -- concurrent activity filter ----------------------------------------------------------

def test_activity_filter_is_concurrent_ordered_and_skips_failures():
    import threading

    live = {"now": 0, "max": 0}
    lock = threading.Lock()

    class Scraper:
        def check_user_activity(self, username):
            with lock:
                live["now"] += 1
                live["max"] = max(live["max"], live["now"])
            time.sleep(0.03)
            with lock:
                live["now"] -= 1
            if username == "boom":
                raise RuntimeError("blocked")
            return {
                "good1": {"film_count": 200, "has_ratings": True},
                "few": {"film_count": 10, "has_ratings": True},
                "norate": {"film_count": 500, "has_ratings": False},
                "ghost": None,
                "good2": {"film_count": 50, "has_ratings": True},
            }.get(username, {"film_count": 100, "has_ratings": True})

    names = ["good1", "few", "norate", "ghost", "boom", "good2"] + [f"x{i}" for i in range(10)]
    t = time.monotonic()
    kept = cli._filter_active_users(Scraper(), names, min_films=50, workers=8)
    elapsed = time.monotonic() - t

    assert kept == ["good1", "good2"] + [f"x{i}" for i in range(10)]  # original order preserved
    assert live["max"] > 1
    assert elapsed < len(names) * 0.03 * 0.6  # clearly faster than serial
    assert cli._filter_active_users(Scraper(), [], 50) == []
