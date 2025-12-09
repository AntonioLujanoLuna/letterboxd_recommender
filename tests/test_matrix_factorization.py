import pytest

from letterboxd_rec.matrix_factorization import SVDRecommender


def test_svd_fit_predict_and_recommend(tmp_path):
    all_user_films = {
        "alice": [
            {"slug": "film-a", "rating": 4.0},
            {"slug": "film-b", "rating": 2.0},
        ],
        "bob": [
            {"slug": "film-a", "rating": 5.0},
            {"slug": "film-b", "rating": 1.0},
            {"slug": "film-c", "rating": 5.0},
        ],
        "carol": [
            {"slug": "film-a", "rating": 4.5},
            {"slug": "film-c", "rating": 4.0},
        ],
    }

    svd = SVDRecommender(n_factors=2)
    svd.fit(all_user_films)

    # Predict for an unseen film (for Alice)
    prediction = svd.predict("alice", "film-c")
    assert prediction is not None
    assert 0.5 <= prediction <= 5.0

    recs = svd.recommend("alice", seen_slugs={"film-a", "film-b"}, n=1)
    assert recs
    assert recs[0][0] == "film-c"

    # Persist and reload the model using a temp cache
    cache_path = tmp_path / "svd_model.npz"
    svd.save(cache_path)
    fingerprint = SVDRecommender.compute_fingerprint(all_user_films)
    loaded = SVDRecommender.load(cache_path, expected_fingerprint=fingerprint)
    assert loaded is not None
    assert loaded.is_fitted


def test_svd_load_rejects_mismatched_fingerprint(tmp_path):
    all_user_films = {
        "alice": [{"slug": "film-a", "rating": 4.0}, {"slug": "film-b", "rating": 3.0}],
        "bob": [{"slug": "film-a", "rating": 3.0}],
    }

    svd = SVDRecommender(n_factors=1)
    svd.fit(all_user_films)

    cache_path = tmp_path / "svd_model.npz"
    svd.save(cache_path)

    wrong_fp = {"n_users": 99, "n_items": 1, "n_ratings": 2}
    assert SVDRecommender.load(cache_path, expected_fingerprint=wrong_fp) is None


def test_svd_recommend_handles_unfitted_and_missing_cache(tmp_path):
    svd = SVDRecommender(n_factors=1)

    # Unfitted recommend should return empty without raising
    recs = svd.recommend("alice", seen_slugs=set(), n=3)
    assert recs == []

    # load gracefully returns None when cache missing
    missing_path = tmp_path / "does_not_exist.npz"
    assert SVDRecommender.load(missing_path) is None


def test_svd_compute_fingerprint_includes_hyperparams():
    films = {"alice": [{"slug": "film-a", "rating": 4.0}]}
    fp = SVDRecommender.compute_fingerprint(films, hyperparams={"k": 10})
    assert fp["hyperparams"] == {"k": 10}


def test_svd_fit_uses_implicit_feedback_only():
    all_user_films = {
        "alice": [{"slug": "film-a", "watchlisted": True}],
        "bob": [{"slug": "film-b", "liked": True}],
    }

    svd = SVDRecommender(n_factors=1)
    svd.fit(all_user_films)

    assert svd.is_fitted
    assert set(svd.item_index.keys()) == {"film-a", "film-b"}


def test_svd_fit_requires_signals_when_implicit_disabled():
    svd = SVDRecommender(n_factors=1, use_implicit=False)
    with pytest.raises(ValueError):
        svd.fit({"alice": [{"slug": "film-a"}]})


def test_svd_predict_handles_unfitted_and_unknown_user():
    svd = SVDRecommender(n_factors=1)
    assert svd.predict("alice", "film-a") is None

    data = {
        "alice": [{"slug": "film-a", "rating": 4.0}],
        "bob": [{"slug": "film-a", "rating": 3.0}, {"slug": "film-b", "rating": 4.0}],
    }
    svd.fit(data)

    assert svd.predict("carol", "film-a") is None


def test_svd_recommend_returns_empty_for_unknown_user():
    data = {
        "alice": [{"slug": "film-a", "rating": 4.0}],
        "bob": [{"slug": "film-a", "rating": 3.0}, {"slug": "film-b", "rating": 4.0}],
    }
    svd = SVDRecommender(n_factors=1).fit(data)

    assert svd.recommend("carol", seen_slugs=set(), n=5) == []


def test_svd_load_handles_corrupt_file(tmp_path):
    bad_path = tmp_path / "corrupt.npz"
    bad_path.write_text("not-an-npz")

    assert SVDRecommender.load(bad_path) is None


def test_svd_save_skips_when_unfitted(tmp_path):
    path = tmp_path / "unfitted.npz"
    svd = SVDRecommender()
    svd.save(path)

    assert not path.exists()