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

