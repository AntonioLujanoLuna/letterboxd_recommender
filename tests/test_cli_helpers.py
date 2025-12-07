import json
import logging
import sqlite3
from types import SimpleNamespace

from letterboxd_rec import cli


def test_parse_weights_sanitizes_and_ignores_invalid(caplog):
    caplog.set_level(logging.WARNING)
    weights = ["User-One:2.5", "missingcolon", "bob:not-a-number", "alice:1"]

    result = cli._parse_weights(weights)

    assert result == {"user-one": 2.5, "alice": 1.0}
    assert "expected user:weight" in caplog.text
    assert "invalid number" in caplog.text


def test_load_films_with_filters():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE films (
            slug TEXT PRIMARY KEY,
            title TEXT,
            year INTEGER,
            avg_rating REAL,
            directors TEXT,
            genres TEXT,
            cast TEXT,
            themes TEXT,
            runtime INTEGER,
            rating_count INTEGER,
            countries TEXT,
            languages TEXT,
            writers TEXT,
            cinematographers TEXT,
            composers TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO films (slug, title, year, avg_rating) VALUES (?, ?, ?, ?)",
        [
            ("early", "Early Film", 1985, 4.0),
            ("target", "Target Film", 2001, 4.5),
            ("late", "Late Film", 2015, 3.0),
        ],
    )

    filtered = cli._load_films_with_filters(
        conn, min_year=1990, max_year=2005, min_rating=4.0
    )

    assert set(filtered.keys()) == {"target"}
    assert filtered["target"]["title"] == "Target Film"


def test_warn_missing_metadata(caplog):
    caplog.set_level(logging.WARNING)
    user_films = [{"slug": "known"}, {"slug": "missing"}]
    all_films = {"known": {"slug": "known"}}

    cli._warn_missing_metadata(user_films, all_films, "alice")

    assert "Missing metadata for 1 films" in caplog.text


def test_output_recommendations_json(caplog):
    caplog.set_level(logging.INFO)
    recs = [
        SimpleNamespace(
            slug="film-a",
            title="Film A",
            year=2020,
            score=4.2,
            reasons=["Because we think you'll like it"],
        )
    ]
    all_films = {
        "film-a": {
            "slug": "film-a",
            "title": "Film A",
            "year": 2020,
            "directors": json.dumps(["Director"]),
            "genres": json.dumps(["Drama"]),
            "cast": json.dumps(["Actor"]),
            "themes": json.dumps(["Theme"]),
            "countries": json.dumps(["USA"]),
            "avg_rating": 4.1,
            "rating_count": 1234,
        }
    }
    args = SimpleNamespace(format="json", explain=False, diversity_report=False)

    cli._output_recommendations(recs, all_films, args, "alice", "metadata")

    log_text = "\n".join(record.message for record in caplog.records)
    assert '"slug": "film-a"' in log_text
    assert "https://letterboxd.com/film/film-a/" in log_text

