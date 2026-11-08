import sqlite3
from datetime import datetime, timedelta

import pytest

from letterboxd_rec.backup import backup_database, list_backups


def _make_db(path, rows=3):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(rows)])
    conn.commit()
    return conn  # left open: a live writer


def test_backup_is_consistent_while_the_database_is_open_for_writing(tmp_path):
    live = _make_db(tmp_path / "live.db", rows=5)
    out = backup_database(tmp_path / "bk", keep=3, db_path=tmp_path / "live.db")
    live.close()
    copy = sqlite3.connect(out)
    assert copy.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 5
    copy.close()
    assert not list((tmp_path / "bk").glob("*.partial"))


def test_backup_prunes_to_the_newest_keep(tmp_path):
    _make_db(tmp_path / "live.db").close()
    base = datetime(2026, 1, 1, 12, 0, 0)
    for i in range(5):
        backup_database(tmp_path / "bk", keep=2, db_path=tmp_path / "live.db", now=base + timedelta(minutes=i))
    kept = [p.name for p in list_backups(tmp_path / "bk")]
    assert kept == ["letterboxd-20260101-120400.db", "letterboxd-20260101-120300.db"]


def test_backup_rejects_missing_database_and_bad_keep(tmp_path):
    with pytest.raises(FileNotFoundError):
        backup_database(tmp_path / "bk", db_path=tmp_path / "nope.db")
    _make_db(tmp_path / "live.db").close()
    with pytest.raises(ValueError):
        backup_database(tmp_path / "bk", keep=0, db_path=tmp_path / "live.db")
