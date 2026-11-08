"""Database health checks: one place to see what is missing, orphaned or stuck."""

import sqlite3
from dataclasses import dataclass

from .database.discovery import MAX_QUEUE_ATTEMPTS

OK, INFO, WARN, ERROR = "ok", "info", "warn", "error"
THIN_USER_FILMS = 20


@dataclass
class Finding:
    level: str
    check: str
    detail: str
    fix: str = ""


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0] or 0)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def run_checks(conn: sqlite3.Connection) -> list[Finding]:
    """Read-only checks over an open database connection. Never modifies data."""
    out: list[Finding] = []

    check = conn.execute("PRAGMA quick_check").fetchone()[0]
    out.append(Finding(OK if check == "ok" else ERROR, "integrity",
                       "SQLite quick_check passed" if check == "ok" else f"quick_check: {check}",
                       "" if check == "ok" else "restore the newest file from data/backups (python main.py backup)"))

    users = _scalar(conn, "SELECT COUNT(DISTINCT username) FROM user_films")
    rows = _scalar(conn, "SELECT COUNT(*) FROM user_films")
    films = _scalar(conn, "SELECT COUNT(*) FROM films")
    out.append(Finding(INFO, "size", f"{users:,} users, {rows:,} interactions, {films:,} films"))

    # Interactions pointing at films we know nothing about: recommendations can never use them.
    orphan = _scalar(conn, "SELECT COUNT(DISTINCT film_slug) FROM user_films "
                           "WHERE film_slug NOT IN (SELECT slug FROM films)")
    out.append(Finding(OK if not orphan else WARN, "films without a row",
                       f"{orphan:,} watched films have no entry in the films table" if orphan else "none",
                       "python main.py refresh-metadata" if orphan else ""))

    bare = _scalar(conn, "SELECT COUNT(*) FROM films WHERE year IS NULL AND slug IN "
                         "(SELECT DISTINCT film_slug FROM user_films)")
    out.append(Finding(OK if not bare else WARN, "films without metadata",
                       f"{bare:,} watched films have a row but no year/metadata" if bare else "none",
                       "python main.py refresh-metadata" if bare else ""))

    with_meta = _scalar(conn, "SELECT COUNT(*) FROM films WHERE year IS NOT NULL")
    no_id = _scalar(conn, "SELECT COUNT(*) FROM films WHERE year IS NOT NULL "
                          "AND tmdb_id IS NULL AND (imdb_id IS NULL OR imdb_id = '')")
    if with_meta:
        share = no_id / with_meta
        out.append(Finding(OK if share < 0.02 else INFO, "films without a TMDB/IMDb id",
                           f"{no_id:,} of {with_meta:,} ({share:.1%}); these can only match MovieLens by title+year"))

    for table, col in (("film_genres", "film_slug"), ("film_directors", "film_slug")):
        if _table_exists(conn, table):
            n = _scalar(conn, f"SELECT COUNT(DISTINCT {col}) FROM {table} WHERE {col} NOT IN (SELECT slug FROM films)")
            if n:
                out.append(Finding(WARN, f"orphaned {table}", f"{n:,} films in {table} are not in films"))

    thin = _scalar(conn, f"SELECT COUNT(*) FROM (SELECT username FROM user_films GROUP BY username "
                         f"HAVING COUNT(*) < {THIN_USER_FILMS})")
    if users:
        out.append(Finding(INFO, "thin users",
                           f"{thin:,} users have fewer than {THIN_USER_FILMS} interactions "
                           f"(EASE ignores films with too few interactions; they add little)"))

    if _table_exists(conn, "pending_users"):
        pending = _scalar(conn, "SELECT COUNT(*) FROM pending_users")
        parked = _scalar(conn, "SELECT COUNT(*) FROM pending_users WHERE attempts >= ?", (MAX_QUEUE_ATTEMPTS,))
        done = _scalar(conn, "SELECT COUNT(*) FROM pending_users WHERE username IN "
                             "(SELECT DISTINCT username FROM user_films)")
        out.append(Finding(INFO, "queue", f"{pending:,} pending"))
        out.append(Finding(OK if not parked else WARN, "parked users",
                           f"{parked:,} users failed {MAX_QUEUE_ATTEMPTS}+ times and are parked" if parked else "none",
                           "python main.py queue-clear (or wait: usually throttling, not a bad account)" if parked else ""))
        if done:
            out.append(Finding(WARN, "queued but already scraped",
                               f"{done:,} queue entries are users already in the database",
                               "python main.py prune-pending"))

    if _table_exists(conn, "user_follows"):
        scraped = _scalar(conn, "SELECT COUNT(DISTINCT username) FROM user_films")
        has_edges = _scalar(conn, "SELECT COUNT(*) FROM (SELECT follower AS u FROM user_follows "
                                  "UNION SELECT followee FROM user_follows) WHERE u IN "
                                  "(SELECT DISTINCT username FROM user_films)")
        missing = scraped - has_edges
        out.append(Finding(OK if missing <= 0 else INFO, "users without social edges",
                           f"{max(missing, 0):,} of {scraped:,} scraped users have no follow edges" if missing > 0 else "none",
                           "python main.py backfill-social" if missing > 0 else ""))
    return out


def format_report(findings: list[Finding]) -> str:
    mark = {OK: "ok  ", INFO: "info", WARN: "WARN", ERROR: "FAIL"}
    lines = []
    for f in findings:
        lines.append(f"[{mark[f.level]}] {f.check}: {f.detail}")
        if f.fix:
            lines.append(f"       fix: {f.fix}")
    return "\n".join(lines)


def worst_level(findings: list[Finding]) -> str:
    order = [OK, INFO, WARN, ERROR]
    return max((f.level for f in findings), key=order.index, default=OK)
