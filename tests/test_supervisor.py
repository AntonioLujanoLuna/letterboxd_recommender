import os

import pytest

from letterboxd_rec import supervisor
from letterboxd_rec.supervisor import AlreadyRunning, acquire_lock, release_lock, run_supervised


class FakeChild:
    def __init__(self, code, polls_before_exit=1):
        self.returncode, self._left = code, polls_before_exit

    def poll(self):
        if self._left > 0:
            self._left -= 1
            return None
        return self.returncode


class Harness:
    def __init__(self, codes):
        self.codes, self.started, self.sleeps, self.t = list(codes), 0, [], 0.0

    def popen(self, cmd):
        self.started += 1
        return FakeChild(self.codes.pop(0))

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s

    def clock(self):
        return self.t


def run(h, **kw):
    return run_supervised(["x"], popen=h.popen, sleep=h.sleep, clock=h.clock, **kw)


def test_clean_exit_is_not_restarted():
    h = Harness([0])
    assert run(h) == 0 and h.started == 1


def test_crash_restarts_with_growing_backoff_then_finishes(monkeypatch):
    monkeypatch.setattr(supervisor, "POLL_SECONDS", 1.0)  # keep poll sleeps distinct from backoffs
    h = Harness([1, 1, 1, 0])
    assert run(h) == 0
    assert h.started == 4
    backoffs = [s for s in h.sleeps if s != supervisor.POLL_SECONDS]
    assert backoffs == [5.0, 10.0, 20.0]


def test_gives_up_after_max_consecutive_failures():
    h = Harness([1] * 10)
    assert run(h, max_failures=3) == 1 and h.started == 3


def test_backups_run_at_start_on_schedule_and_after_exit_without_crashing_the_run():
    calls = []
    h = Harness([0])

    def long_child(cmd):
        h.started += 1
        return FakeChild(0, polls_before_exit=10)  # 10 polls x 5s = 50s of run time

    def flaky_backup():
        calls.append(h.t)
        if len(calls) == 2:
            raise OSError("disk full")

    code = run_supervised(["x"], popen=long_child, sleep=h.sleep, clock=h.clock,
                          backup_fn=flaky_backup, backup_every=20.0)
    assert code == 0
    assert calls[0] == 0.0               # at start
    assert len(calls) >= 4               # start + scheduled (t=20, 40) + after exit; the failure didn't stop it


def test_lock_blocks_a_second_live_supervisor_but_replaces_a_stale_one(tmp_path):
    lock = tmp_path / "sup.lock"
    lock.write_text("424242")
    with pytest.raises(AlreadyRunning):
        acquire_lock(lock, alive=lambda pid: True)
    acquire_lock(lock, alive=lambda pid: False)       # stale holder
    assert lock.read_text() == str(os.getpid())
    release_lock(lock)
    assert not lock.exists()


def test_pid_alive_sees_this_process_and_a_dead_pid():
    assert supervisor.pid_alive(os.getpid())
    assert not supervisor.pid_alive(0)
