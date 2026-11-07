"""Adaptive request-rate control (AIMD: additive increase, multiplicative decrease).

A fixed `--max-rps` is a guess. With `--auto-rps CEILING` the scraper instead starts at `--max-rps`
and probes upward: +`step` req/s after every `increase_after` seconds without a throttle signal
(429, or a 403 that isn't a private page), and halves immediately on a signal. It therefore settles
near the highest rate Letterboxd tolerates from this IP, and backs off before the block circuit
breaker (which costs minutes) has to trip.
"""
import logging
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)


class AimdRate:
    def __init__(self, start: float, ceiling: float, floor: float | None = None, step: float = 1.0,
                 increase_after: float = 20.0, decrease_factor: float = 0.5, debounce: float = 10.0,
                 soft_threshold: int = 5, soft_window: float = 30.0,
                 clock=time.monotonic):
        if start <= 0 or ceiling < start:
            raise ValueError("need 0 < start <= ceiling")
        self.ceiling = ceiling
        self.floor = floor if floor is not None else max(0.5, start / 4)
        self.step = step
        self.increase_after = increase_after
        self.decrease_factor = decrease_factor
        self.debounce = debounce  # a burst of simultaneous throttles counts once
        self._clock = clock
        self._lock = threading.Lock()  # called from the async loop and from transport threads
        # Soft signals: blocked attempts that a retry recovered. One or two are normal noise; only a
        # cluster (soft_threshold within soft_window seconds) means we are actually going too fast.
        self.soft_threshold = soft_threshold
        self.soft_window = soft_window
        self._soft: deque[float] = deque()
        self._rps = start
        now = clock()
        self._last_change = now
        self._last_throttle = -float("inf")

    @property
    def rps(self) -> float:
        return self._rps

    def on_ok(self) -> None:
        """Called after each successful response."""
        with self._lock:
            now = self._clock()
            if (self._rps < self.ceiling
                    and now - self._last_change >= self.increase_after
                    and now - self._last_throttle >= self.increase_after):
                self._rps = min(self.ceiling, self._rps + self.step)
                self._last_change = now
                logger.info(f"Auto-rate: no throttling for {self.increase_after:.0f}s, raising to {self._rps:.1f} req/s")

    def on_soft_block(self) -> None:
        """A blocked attempt that the retry recovered. Cuts the rate only when they cluster."""
        with self._lock:
            now = self._clock()
            self._soft.append(now)
            while self._soft and now - self._soft[0] > self.soft_window:
                self._soft.popleft()
            if len(self._soft) < self.soft_threshold:
                return
            self._soft.clear()
        self.on_throttle()

    def on_throttle(self) -> None:
        """Called on a 429 or a 403 that survived every retry: cut immediately."""
        with self._lock:
            now = self._clock()
            if now - self._last_throttle < self.debounce:
                return
            before = self._rps
            self._rps = max(self.floor, self._rps * self.decrease_factor)
            self._last_throttle = now
            self._last_change = now
            logger.warning(f"Auto-rate: throttled by Letterboxd, cutting {before:.1f} -> {self._rps:.1f} req/s")
