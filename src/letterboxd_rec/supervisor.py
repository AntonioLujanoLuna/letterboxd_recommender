"""Keep a long-running command alive: restart it when it crashes, back the database up on a schedule,
and refuse to start a second supervisor (two crawlers on one IP just trigger more throttling)."""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from .config import DB_PATH

logger = logging.getLogger(__name__)

LOCK_PATH = DB_PATH.parent / "supervisor.lock"

RESTART_BACKOFF_START = 5.0
RESTART_BACKOFF_MAX = 300.0
HEALTHY_RUN_SECONDS = 600.0   # a child that ran this long before dying resets the backoff and the failure count
POLL_SECONDS = 5.0


class AlreadyRunning(RuntimeError):
    pass


def pid_alive(pid: int) -> bool:
    """Whether a process exists. (os.kill(pid, 0) would *terminate* the process on Windows.)"""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION, STILL_ACTIVE = 0x1000, 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(path: Path = LOCK_PATH, alive: Callable[[int], bool] = pid_alive) -> None:
    """Write our pid into the lock file, unless a live supervisor already holds it."""
    path = Path(path)
    if path.exists():
        try:
            holder = int(path.read_text().strip())
        except (ValueError, OSError):
            holder = 0
        if holder and holder != os.getpid() and alive(holder):
            raise AlreadyRunning(f"A supervisor is already running (pid {holder}; lock file {path}).")
        logger.info(f"Replacing stale lock from pid {holder or '?'}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(os.getpid()))


def release_lock(path: Path = LOCK_PATH) -> None:
    path = Path(path)
    try:
        if path.exists() and path.read_text().strip() == str(os.getpid()):
            path.unlink()
    except OSError:
        pass


def run_supervised(
    cmd: list[str],
    *,
    backup_fn: Callable[[], object] | None = None,
    backup_every: float = 3600.0,
    max_failures: int = 20,
    popen: Callable[..., "subprocess.Popen"] = subprocess.Popen,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Run `cmd` until it exits 0. A crash restarts it after a growing pause.

    Gives up (returns 1) after `max_failures` crashes in a row, where a run that lasted at least
    HEALTHY_RUN_SECONDS does not count as part of the streak. Returns 0 on a clean finish and 130 if
    interrupted (the child is terminated first).
    """
    def safe_backup(reason: str) -> None:
        if backup_fn is None:
            return
        try:
            backup_fn()
        except Exception as e:  # a failed backup must never take the crawl down with it
            logger.warning(f"Backup ({reason}) failed: {e}")

    failures, backoff = 0, RESTART_BACKOFF_START
    safe_backup("start")
    next_backup = clock() + backup_every
    child = None
    try:
        while True:
            logger.info(f"Starting: {' '.join(cmd)}")
            started = clock()
            child = popen(cmd)
            while child.poll() is None:
                sleep(POLL_SECONDS)
                if backup_fn is not None and clock() >= next_backup:
                    safe_backup("scheduled")
                    next_backup = clock() + backup_every
            code = child.returncode
            safe_backup("child exited")
            if code == 0:
                logger.info("Command finished cleanly.")
                return 0
            if clock() - started >= HEALTHY_RUN_SECONDS:
                failures, backoff = 0, RESTART_BACKOFF_START
            failures += 1
            if failures >= max_failures:
                logger.error(f"Command failed {failures} times in a row (last exit code {code}); giving up.")
                return 1
            logger.warning(f"Command exited with code {code}; restarting in {backoff:.0f}s "
                           f"(failure {failures}/{max_failures}).")
            sleep(backoff)
            backoff = min(backoff * 2, RESTART_BACKOFF_MAX)
    except KeyboardInterrupt:
        logger.info("Interrupted; stopping the command.")
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except Exception:
                child.kill()
        return 130
