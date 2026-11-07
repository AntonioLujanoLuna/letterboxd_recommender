import argparse
import random

import numpy as np
import pytest

from letterboxd_rec.movielens import (
    FilmMatcher,
    MovieLensModel,
    build_model,
    normalize_title,
    split_title_year,
)


@pytest.mark.parametrize("raw,expected", [
    ("Matrix, The (1999)", ("The Matrix", 1999)),
    ("Toy Story (1995)", ("Toy Story", 1995)),
    ("Léon: The Professional (a.k.a. The Professional) (Léon) (1994)", ("Léon: The Professional", 1994)),
    ("City of God (Cidade de Deus) (2002)", ("City of God", 2002)),
    ("Babylon 5", ("Babylon 5", None)),
    ("Haine, La (1995)", ("La Haine", 1995)),
])
def test_split_title_year(raw, expected):
    assert split_title_year(raw) == expected


def test_normalize_title():
    assert normalize_title("The Matrix") == normalize_title("Matrix") == "matrix"
    assert normalize_title("Amélie") == "amelie"
    assert normalize_title("Fast & Furious") == "fast and furious"


def _write_ml(tmp_path, n_users=120, seed=0):
    """Two taste groups over 20 films: group A loves films 1-10, group B loves 11-20."""
    rng = random.Random(seed)
    d = tmp_path / "ml"
    d.mkdir()
    with open(d / "movies.csv", "w", encoding="utf-8") as fh:
        fh.write("movieId,title,genres\n")
        for m in range(1, 21):
            fh.write(f'{m},"Film {m}, The ({1990 + m})",Drama\n')
    with open(d / "links.csv", "w", encoding="utf-8") as fh:
        fh.write("movieId,imdbId,tmdbId\n")
        for m in range(1, 21):
            fh.write(f"{m},{1000 + m},{5000 + m}\n")
    with open(d / "ratings.csv", "w", encoding="utf-8") as fh:
        fh.write("userId,movieId,rating,timestamp\n")
        for u in range(1, n_users + 1):
            own = range(1, 11) if u % 2 else range(11, 21)
            other = range(11, 21) if u % 2 else range(1, 11)
            for m in rng.sample(list(own), 7):
                fh.write(f"{u},{m},{rng.choice([4.0, 4.5, 5.0])},0\n")
            for m in rng.sample(list(other), 2):
                fh.write(f"{u},{m},{rng.choice([1.0, 1.5])},0\n")
    return d


def test_build_model_beats_popularity_and_roundtrips(tmp_path):
    ml_dir = _write_ml(tmp_path)
    out = tmp_path / "model.npz"
    model, report = build_model(ml_dir=ml_dir, max_items=20, lambdas=(1.0, 10.0), eval_users=40, eval_min_positives=5, k=5, out_path=out)

    assert report.n_items == 20 and report.n_eval_users == 40
    pop = report.results[0]
    best = max(report.results[1:], key=lambda r: r["ndcg@5"])
    assert pop["model"] == "popularity"
    assert best["ndcg@5"] > pop["ndcg@5"]
    assert report.best_lambda == best["lambda"]
    assert (ml_dir / "ratings.npz").exists()  # cached for the next build

    loaded = MovieLensModel.load(out)
    assert np.array_equal(loaded.B, model.B)
    assert loaded.titles[0] == "The Film 1" and loaded.years[0] == 1991
    assert loaded.tmdb_ids[0] == 5001 and loaded.imdb_ids[0] == "tt0001001"


def test_matcher_and_recommendation_for_letterboxd_history(tmp_path):
    model, _ = build_model(ml_dir=_write_ml(tmp_path), max_items=20, lambdas=(10.0,), eval_users=10, eval_min_positives=5, k=5, out_path=None)
    matcher = FilmMatcher(model)

    films = {
        "film-1": {"slug": "film-1", "title": "Whatever", "tmdb_id": 5001},              # by TMDB id
        "film-2": {"slug": "film-2", "title": "Other", "imdb_id": "tt0001002"},          # by IMDb id
        "the-film-3": {"slug": "the-film-3", "title": "The Film 3", "year": 1994},        # title, year off by one
        "new-2025": {"slug": "new-2025", "title": "Brand New", "year": 2025},            # not in MovieLens
    }
    history = [
        {"slug": "film-1", "rating": 5.0},
        {"slug": "film-2", "rating": 4.5},
        {"slug": "the-film-3", "liked": 1},
        {"slug": "new-2025", "rating": 5.0},
    ]
    weighted, seen, unmatched = matcher.map_history(history, films)
    assert unmatched == 1
    assert {model.col_of_movie[m] for m in (1, 2, 3)} == set(weighted) == seen

    recs = model.recommend_columns(weighted, seen, n=5)
    rec_movies = {int(model.movie_ids[c]) for c, _ in recs}
    assert rec_movies <= set(range(4, 11))  # rest of group A, nothing already seen
    assert matcher.letterboxd_slugs(films)[model.col_of_movie[1]] == "film-1"


def test_cli_movielens_strategy_uses_tmdb_links_for_unknown_films(tmp_path, monkeypatch):
    from letterboxd_rec import cli

    model, _ = build_model(ml_dir=_write_ml(tmp_path), max_items=20, lambdas=(10.0,), eval_users=10, eval_min_positives=5, k=5, out_path=None)
    monkeypatch.setattr("letterboxd_rec.cli.utils.strategies._MOVIELENS_MODEL", model)
    films = {"film-1": {"slug": "film-1", "title": "Film 1", "year": 1991, "tmdb_id": 5001, "genres": "[]"}}
    args = argparse.Namespace(limit=3, genres=None, exclude_genres=None, min_rating=None, min_year=None, max_year=None)
    recs = cli._run_movielens_strategy([{"slug": "film-1", "rating": 5.0}], films, args, "me", None)
    assert len(recs) == 3
    assert all(r.slug.startswith("tmdb:") for r in recs)
    assert cli.film_url(recs[0].slug).startswith("https://letterboxd.com/tmdb/")
    assert all(r.reasons == ["Because you liked The Film 1"] for r in recs)
