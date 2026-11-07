"""Adaptive request-rate control (AIMD: additive increase, multiplicative decrease).

A fixed `--max-rps` is a guess. With `--auto-rps CEILING` the scraper instead starts at `--max-rps`
and probes upward: +`step` req/s after every `increase_after` seconds without a throttle signal
(429, or a 403 that isn't a private page), and halves immediately on a signal. It therefore settles
near the highest rate Letterboxd tolerates from this IP, and backs off before the block circuit
breaker (which costs minutes) has to trip.
"""
import logging
import time

logger = logging.getLogger(__name__)


class AimdRate:
    def __init__(self, start: float, ceiling: float, floor: float | None = None, step: float = 1.0,
                 increase_after: float = 20.0, decrease_factor: float = 0.5, debounce: float = 10.0,
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
        self._rps = start
        now = clock()
        self._last_change = now
        self._last_throttle = -float("inf")

    @property
    def rps(self) -> float:
        return self._rps

    def on_ok(self) -> None:
        """Called after each successful response."""
        now = self._clock()
        if (self._rps < self.ceiling
                and now - self._last_change >= self.increase_after
                and now - self._last_throttle >= self.increase_after):
            self._rps = min(self.ceiling, self._rps + self.step)
            self._last_change = now
            logger.info(f"Auto-rate: no throttling for {self.increase_after:.0f}s, raising to {self._rps:.1f} req/s")

    def on_throttle(self) -> None:
        """Called on a 429 or a (non-private) 403."""
        now = self._clock()
        if now - self._last_throttle < self.debounce:
            return
        before = self._rps
        self._rps = max(self.floor, self._rps * self.decrease_factor)
        self._last_throttle = now
        self._last_change = now
        logger.warning(f"Auto-rate: throttled by Letterboxd, cutting {before:.1f} -> {self._rps:.1f} req/s")
