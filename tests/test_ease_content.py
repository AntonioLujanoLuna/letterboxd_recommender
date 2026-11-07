import json

import numpy as np
import pytest

from letterboxd_rec.content import ContentRecommender
from letterboxd_rec.ease import EASERecommender, interaction_weight, load_or_fit_ease


def _cooccurrence_data():
    """Group A watches a1..a4 together, group B watches b1..b4 together."""
    data = {}
    for u in range(8):
        data[f"a{u}"] = [{"slug": f"a{i}", "rating": 4.5} for i in range(1, 5) if i != (u % 4) + 1]
        data[f"b{u}"] = [{"slug": f"b{i}", "liked": 1} for i in range(1, 5) if i != (u % 4) + 1]
    return data


def test_interaction_weight():
    assert interaction_weight({"rating": 4.0}) == 1.0
    assert interaction_weight({"rating": 3.0}) == 0.5
    assert interaction_weight({"rating": 1.0}) == 0.0
    assert interaction_weight({"rating": 1.0, "liked": 1}) == 1.0
    assert interaction_weight({"watched": 1}) == 0.5
    assert interaction_weight({"watchlisted": 1}) == 0.25
    assert interaction_weight({}) == 0.0


def test_ease_closed_form_properties():
    model = EASERecommender(lam=10, min_interactions=2).fit(_cooccurrence_data())
    B = model.B
    assert B.shape == (8, 8)
    assert np.allclose(np.diag(B), 0)
    # Within-group weights are positive; cross-group weights are ~0 or negative
    a1, a2, b1 = (model.item_index[s] for s in ("a1", "a2", "b1"))
    assert B[a1, a2] > 0
    assert B[a1, b1] <= 1e-6


def test_ease_recommends_missing_group_film_and_skips_seen():
    model = EASERecommender(lam=10, min_interactions=2).fit(_cooccurrence_data())
    # A new user (not in training) who watched a1..a3 should get a4 first
    recs = model.recommend([{"slug": s, "rating": 5.0} for s in ("a1", "a2", "a3")], n=3)
    assert recs[0][0] == "a4"
    assert not {"a1", "a2", "a3"} & {s for s, _ in recs}
    assert model.recommend([{"slug": "a1", "rating": 5.0}], n=3, exclude={"a2"})[0][0] != "a2"


def test_ease_requires_enough_items():
    with pytest.raises(ValueError):
        EASERecommender(min_interactions=5).fit({"u": [{"slug": "x", "rating": 5.0}]})


def test_ease_cache_roundtrip_and_invalidation(tmp_path):
    data = _cooccurrence_data()
    path = tmp_path / "ease.npz"
    fitted = load_or_fit_ease(data, cache_path=path)
    assert path.exists()
    cached = load_or_fit_ease(data, cache_path=path)
    assert np.array_equal(cached.B, fitted.B) and cached.item_slugs == fitted.item_slugs

    data["a0"][0]["rating"] = 1.0  # same counts, different content
    assert EASERecommender.load(path, expected_fingerprint=EASERecommender().fingerprint(data)) is None


def _film(slug, directors=(), genres=(), cast=(), year=2000):
    return {
        "slug": slug, "title": slug, "year": year,
        "directors": json.dumps(list(directors)), "genres": json.dumps(list(genres)),
        "cast": json.dumps(list(cast)), "themes": json.dumps([]),
    }


def _content_films():
    return {
        "lynch-1": _film("lynch-1", ["David Lynch"], ["mystery"], ["Kyle"]),
        "lynch-2": _film("lynch-2", ["David Lynch"], ["mystery"], ["Laura"]),
        "lynch-3": _film("lynch-3", ["David Lynch"], ["drama"], ["Kyle"]),
        "comedy-1": _film("comedy-1", ["Someone"], ["comedy"], ["Jim"], year=1990),
        "comedy-2": _film("comedy-2", ["Someone"], ["comedy"], ["Jim"], year=1990),
        "drama-x": _film("drama-x", ["Other"], ["drama"], ["Nobody"], year=1970),
    }


def test_content_ranks_shared_director_first_and_explains():
    model = ContentRecommender(_content_films())
    user = [{"slug": "lynch-1", "rating": 5.0}, {"slug": "comedy-1", "rating": 1.5}]
    recs = model.recommend(user, n=5)
    slugs = [s for s, _ in recs]
    assert slugs[0] in {"lynch-2", "lynch-3"}
    assert "comedy-2" not in slugs  # disliked director/genre pushes it below zero
    assert not {"lynch-1", "comedy-1"} & set(slugs)
    reasons = model.top_shared_features(user, slugs[0])
    assert any(r.startswith("Director: david lynch") for r in reasons)


def test_content_rows_are_normalized_and_singletons_dropped():
    model = ContentRecommender(_content_films())
    norms = np.sqrt(np.asarray(model.F.multiply(model.F).sum(axis=1)).ravel())
    assert np.all((np.abs(norms - 1) < 1e-9) | (norms == 0))
    assert ("cast", "nobody") not in model.feature_names  # appears on one film only


def test_content_empty_history_returns_nothing():
    assert ContentRecommender(_content_films()).recommend([], n=5) == []
