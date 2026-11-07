"""Optional transport backend built on `letterboxdpy` (https://github.com/nmcassa/letterboxdpy).

letterboxdpy fetches pages through curl_cffi with Chrome TLS impersonation, so it often gets
through where the plain httpx client in `scraper.py` is blocked (403 / Cloudflare).

Rather than re-implementing parsing, this module swaps only the *HTTP layer*: `CurlClient` /
`AsyncCurlClient` expose the same `.get(url)` shape as httpx and return real `httpx.Response`
objects. The native scrapers' parsers, 429/soft-block handling, async batching, fan counts,
discovery sources, favorites and lists therefore all work unchanged on top of it.

Select it with `--backend letterboxdpy` or `LETTERBOXD_BACKEND=letterboxdpy`. It applies to the
sync scraper, the async scraper and the scrape-daemon.

Uses letterboxdpy's curl_cffi session, headers and Chrome impersonation, but not its `_fetch`
retry loop (which sleeps ~20s on *every* 403, private pages included). Retries live in
`CurlClient.get`: private 403s return at once, real blocks get a couple of short retries.
"""
import asyncio
import logging
import os
import random
import threading
import time

import httpx

from .config import (
    DEFAULT_MAX_RPS,
    BLOCK_BREAKER_COOLDOWN,
    BLOCK_BREAKER_MAX_COOLDOWN,
    BLOCK_BREAKER_THRESHOLD,
    LETTERBOXD_COOKIE,
)
from .scrape_stats import STATS
from .scraper import AsyncLetterboxdScraper, LetterboxdScraper, _looks_private, _parse_cookie_header

logger = logging.getLogger(__name__)

BACKENDS = ("native", "letterboxdpy")

_backend = os.environ.get("LETTERBOXD_BACKEND", "native").strip().lower()
if _backend not in BACKENDS:
    logger.warning(f"Invalid LETTERBOXD_BACKEND='{_backend}', using 'native'")
    _backend = "native"


def get_backend() -> str:
    return _backend


def set_backend(name: str) -> None:
    global _backend
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; choose from {BACKENDS}")
    _backend = name


def make_scraper(delay: float = 1.0) -> LetterboxdScraper:
    """Build the sync scraper for the selected backend."""
    if _backend == "letterboxdpy":
        return LetterboxdPyScraper(delay=delay)
    return LetterboxdScraper(delay=delay)


def make_async_scraper(delay: float = 0.2, max_concurrent: int = 5, max_rps: float | None = None,
                       rps_ceiling: float | None = None) -> AsyncLetterboxdScraper:
    """Build the async scraper for the selected backend (max_rps defaults to LETTERBOXD_MAX_RPS)."""
    rps = DEFAULT_MAX_RPS if max_rps is None else max_rps
    cls = AsyncLetterboxdPyScraper if _backend == "letterboxdpy" else AsyncLetterboxdScraper
    return cls(delay=delay, max_concurrent=max_concurrent, max_rps=rps, rps_ceiling=rps_ceiling)


# -- block circuit breaker ---------------------------------------------------------------

class BlockBreaker:
    """Pauses *all* workers when Letterboxd starts blocking us.

    Every blocked request already costs ~20s of internal retries inside letterboxdpy, and with
    several workers each one keeps hammering a blocked IP. After `threshold` consecutive blocks
    the breaker opens: every caller sleeps until the cooldown ends. The cooldown doubles each
    time it re-opens without a success in between (up to `max_cooldown`). After a cooldown the
    breaker is half-open: a single further block re-opens it immediately; one success resets it.
    """

    def __init__(self, threshold: int = BLOCK_BREAKER_THRESHOLD, cooldown: float = BLOCK_BREAKER_COOLDOWN,
                 max_cooldown: float = BLOCK_BREAKER_MAX_COOLDOWN, clock=time.monotonic, sleep=time.sleep):
        self.threshold = threshold
        self.cooldown = cooldown
        self.max_cooldown = max_cooldown
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._consecutive = 0
        self._opens = 0
        self._until = 0.0
        self._half_open = False

    def wait(self) -> None:
        """Block while the breaker is open."""
        while True:
            with self._lock:
                remaining = self._until - self._clock()
            if remaining <= 0:
                return
            self._sleep(remaining)

    def record_ok(self) -> None:
        with self._lock:
            self._consecutive = 0
            self._opens = 0
            self._half_open = False

    def record_block(self) -> None:
        with self._lock:
            self._consecutive += 1
            if not (self._half_open or self._consecutive >= self.threshold):
                return
            cooldown = min(self.cooldown * (2 ** self._opens), self.max_cooldown)
            self._opens += 1
            self._consecutive = 0
            self._half_open = True
            self._until = self._clock() + cooldown
        logger.warning(f"Letterboxd is blocking requests; pausing all scraping for {cooldown:.0f}s "
                       f"(pause #{self._opens})")


def _is_block(resp) -> bool:
    """A 403 that isn't just a private/classified page."""
    if resp.status_code != 403:
        return False
    text = (resp.text or "").lower()
    return not ("classified" in text or "falken" in text)


BREAKER = BlockBreaker()
_announced_auth = False  # log the auth mode once per process

# -- transport ---------------------------------------------------------------------------

class CurlClient:
    """httpx-like sync client (`.get`, `.close`) backed by letterboxdpy's fetcher.

    curl_cffi sessions are not guaranteed thread-safe and the async scraper calls this from
    worker threads, so each thread gets its own `Scraper` subclass (hence its own session).
    """

    def __init__(self, breaker: BlockBreaker | None = None, on_block=None):
        self._breaker = breaker or BREAKER
        # Called for each blocked (non-private 403) attempt that gets retried. Without it a block
        # that succeeds on retry is invisible to the adaptive rate controller.
        self.on_block = on_block
        try:
            from letterboxdpy.core.scraper import Scraper  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "The 'letterboxdpy' backend needs the letterboxdpy package: "
                "pip install -r requirements.txt"
            ) from e
        self._local = threading.local()
        self._cookies = _parse_cookie_header(LETTERBOXD_COOKIE)
        global _announced_auth
        if not _announced_auth:
            _announced_auth = True
            logger.info(f"letterboxdpy transport: {'logged-in session (%d cookies)' % len(self._cookies) if self._cookies else 'anonymous (no cookie)'}")

    def _fetcher(self):
        fetcher = getattr(self._local, "fetcher", None)
        if fetcher is None:
            from letterboxdpy.core.scraper import Scraper

            # A subclass with its own `_session` slot => a private session per thread.
            fetcher = type("ThreadScraper", (Scraper,), {"_session": None})
            for name, value in self._cookies.items():
                fetcher.instance().cookies.set(name, value, domain=".letterboxd.com")
            self._local.fetcher = fetcher
        return fetcher

    ATTEMPTS = 3          # per request: the first try plus two retries
    IMPERSONATE = "chrome120"  # same browser fingerprint letterboxdpy uses

    def _request_once(self, url: str):
        """One HTTP attempt on this thread's letterboxdpy session."""
        from letterboxdpy.utils.utils_url import normalize_letterboxd_url

        fetcher = self._fetcher()
        return fetcher.instance().get(
            normalize_letterboxd_url(url), headers=fetcher.headers, timeout=fetcher.timeout,
            impersonate=self.IMPERSONATE,
        )

    def get(self, url: str) -> httpx.Response:
        from curl_cffi.requests import errors as curl_errors

        request = httpx.Request("GET", url)
        self._breaker.wait()

        resp = None
        for attempt in range(self.ATTEMPTS):
            last = attempt == self.ATTEMPTS - 1
            started = time.monotonic()
            try:
                resp = self._request_once(url)
            except curl_errors.RequestsError as e:
                STATS.record(None, time.monotonic() - started)
                if not last:
                    time.sleep(1.0 * (attempt + 1) + random.random() * 0.5)
                    continue
                # Map to an httpx error so the native retry/skip handlers apply unchanged.
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    raise httpx.ReadTimeout(str(e), request=request) from e
                raise httpx.ConnectError(str(e), request=request) from e

            STATS.record(resp.status_code, time.monotonic() - started)
            # A 403 may be a transient Cloudflare challenge, so retry briefly. A private/classified
            # page (also a 403) never gets better: return it immediately.
            if resp.status_code == 403 and not _looks_private(resp.text) and not last:
                if self.on_block:
                    self.on_block()
                time.sleep(1.5 * (attempt + 1) + random.random() * 0.5)
                continue
            break

        if _is_block(resp):
            self._breaker.record_block()
        elif resp.status_code in (200, 404):
            self._breaker.record_ok()

        # Only forward headers the callers read. Forwarding content-encoding/length would make
        # httpx try to decompress the (already decoded) text.
        headers = {}
        for key in ("retry-after", "content-type"):
            value = resp.headers.get(key)
            if value:
                headers[key] = value
        return httpx.Response(resp.status_code, headers=headers, text=resp.text, request=request)

    def close(self) -> None:
        pass


class AsyncCurlClient:
    """httpx.AsyncClient-like (`await .get`, `await .aclose`) running CurlClient in threads."""

    def __init__(self, sync_client: CurlClient | None = None):
        self._sync = sync_client or CurlClient()

    async def get(self, url: str) -> httpx.Response:
        return await asyncio.to_thread(self._sync.get, url)

    async def aclose(self) -> None:
        pass


# -- scrapers ----------------------------------------------------------------------------

class LetterboxdPyScraper(LetterboxdScraper):
    """LetterboxdScraper whose HTTP goes through letterboxdpy's browser-impersonating fetcher."""

    def __init__(self, delay: float = 1.0):
        super().__init__(delay=delay)
        self.client.close()  # drop the unused httpx client built by the parent
        self.client = CurlClient()


class AsyncLetterboxdPyScraper(AsyncLetterboxdScraper):
    """AsyncLetterboxdScraper whose HTTP goes through letterboxdpy's fetcher (in worker threads)."""

    async def __aenter__(self):
        self.client = AsyncCurlClient(CurlClient(on_block=self._note_throttle))
        return self

    async def scrape_films_batch(self, slugs):
        # The parent builds a temporary httpx client when used outside `async with`;
        # always use the curl-backed one instead.
        client = self.client or AsyncCurlClient()
        return await self._scrape_batch_with_client(client, slugs)
