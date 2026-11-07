"""Lightweight request metrics so scraping speed can be tuned from data instead of guesses.

Every HTTP request records (status, latency). The daemon logs a one-line window report about once a
minute (requests/s, p50/p95 latency, 403/429/error counts) and a cumulative summary at the end.
Both backends feed it: the native httpx clients through the Timed* wrappers below, the
letterboxdpy transport directly (so circuit-breaker waits are never counted as latency).
"""
import threading
import time
from collections import deque


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[idx]


class ScrapeStats:
    def __init__(self, window_seconds: float = 60.0, clock=time.monotonic):
        self._clock = clock
        self._window = window_seconds
        self._lock = threading.Lock()
        self._events: deque[tuple[float, int | None, float]] = deque()
        self._started = clock()
        self._last_report = self._started
        self._total = 0
        self._by_class: dict[str, int] = {}

    @staticmethod
    def _classify(status: int | None) -> str:
        if status is None:
            return "err"
        if status in (403, 429):
            return str(status)
        if status >= 400 and status != 404:
            return "other"
        return "ok"  # 2xx/3xx and 404 (a missing page is a normal answer)

    def record(self, status: int | None, seconds: float) -> None:
        """status=None means the request failed without a response (timeout, connection error)."""
        now = self._clock()
        with self._lock:
            self._events.append((now, status, seconds))
            self._total += 1
            cls = self._classify(status)
            self._by_class[cls] = self._by_class.get(cls, 0) + 1
            cutoff = now - self._window * 2  # keep a little more than one window
            while self._events and self._events[0][0] < cutoff:
                self._events.popleft()

    def _format(self, events, span: float, label: str) -> str:
        latencies = sorted(e[2] for e in events)
        classes: dict[str, int] = {}
        for _, status, _ in events:
            c = self._classify(status)
            classes[c] = classes.get(c, 0) + 1
        problems = " ".join(f"{k}:{classes[k]}" for k in ("403", "429", "other", "err") if classes.get(k)) or "none"
        rps = len(events) / span if span > 0 else 0.0
        return (f"Throughput ({label}): {rps:.1f} req/s over {len(events)} requests | "
                f"latency p50 {_percentile(latencies, .5):.2f}s p95 {_percentile(latencies, .95):.2f}s | "
                f"problems: {problems}")

    def maybe_report(self, interval: float | None = None) -> str | None:
        """A window report if `interval` seconds passed since the last one (and there was traffic)."""
        interval = self._window if interval is None else interval
        now = self._clock()
        with self._lock:
            if now - self._last_report < interval:
                return None
            span = now - self._last_report
            self._last_report = now
            events = [e for e in self._events if e[0] >= now - span]
        if not events:
            return None
        return self._format(events, span, f"last {span:.0f}s")

    def summary(self) -> str:
        now = self._clock()
        with self._lock:
            if not self._total:
                return "No HTTP requests recorded."
            span = now - self._started
            total, by_class = self._total, dict(self._by_class)
        problems = " ".join(f"{k}:{by_class[k]}" for k in ("403", "429", "other", "err") if by_class.get(k)) or "none"
        return (f"Requests this session: {total} in {span:.0f}s ({total / span:.1f} req/s average) | "
                f"problems: {problems}" if span > 0 else f"Requests this session: {total}")


STATS = ScrapeStats()


class TimedClient:
    """Wraps a sync httpx client, recording every `.get` into `stats`; other attributes pass through."""

    def __init__(self, client, stats: ScrapeStats | None = None):
        self._client = client
        self._stats = stats or STATS

    def get(self, *args, **kwargs):
        start = time.monotonic()
        try:
            resp = self._client.get(*args, **kwargs)
        except Exception:
            self._stats.record(None, time.monotonic() - start)
            raise
        self._stats.record(resp.status_code, time.monotonic() - start)
        return resp

    def __getattr__(self, name):
        return getattr(self._client, name)


class TimedAsyncClient:
    """Async twin of TimedClient (wraps httpx.AsyncClient)."""

    def __init__(self, client, stats: ScrapeStats | None = None):
        self._client = client
        self._stats = stats or STATS

    async def get(self, *args, **kwargs):
        start = time.monotonic()
        try:
            resp = await self._client.get(*args, **kwargs)
        except Exception:
            self._stats.record(None, time.monotonic() - start)
            raise
        self._stats.record(resp.status_code, time.monotonic() - start)
        return resp

    def __getattr__(self, name):
        return getattr(self._client, name)
