import json
import random

import pytest

from letterboxd_rec.evaluation import (
    format_results_table,
    hit_at_k,
    is_positive,
    make_holdout_split,
    ndcg_at_k,
    recall_at_k,
    run_evaluation,
)


def _clustered_data(n_users_per_group=15, n_films_per_group=30, seed=0):
    """Two taste groups: each rates its own films highly and the other group's films low."""
    rng = random.Random(seed)
    films = {}
    groups = {"a": [f"a-{i}" for i in range(n_films_per_group)], "b": [f"b-{i}" for i in range(n_films_per_group)]}
    for g, slugs in groups.items():
        for i, slug in enumerate(slugs):
            films[slug] = {
                "slug": slug, "title": slug, "year": 2000 + i % 20,
                "genres": json.dumps([f"genre-{g}"]), "directors": json.dumps([f"dir-{g}-{i % 5}"]),
                "cast": json.dumps([]), "themes": json.dumps([]), "countries": json.dumps([]),
                "languages": json.dumps([]), "writers": json.dumps([]),
                "cinematographers": json.dumps([]), "composers": json.dumps([]),
                "avg_rating": 3.5, "rating_count": 1000 + i,
            }
    all_user_films = {}
    for g, own in groups.items():
        other = groups["b" if g == "a" else "a"]
        for u in range(n_users_per_group):
            entries = [
                {"slug": s, "rating": rng.choice([4.0, 4.5, 5.0]), "watched": 1, "liked": 0, "watchlisted": 0}
                for s in rng.sample(own, 20)
            ]
            entries += [
                {"slug": s, "rating": rng.choice([1.0, 1.5, 2.0]), "watched": 1, "liked": 0, "watchlisted": 0}
                for s in rng.sample(other, 5)
            ]
            all_user_films[f"{g}-user{u}"] = entries
    return all_user_films, films


def test_is_positive():
    assert is_positive({"rating": 3.5})
    assert is_positive({"rating": None, "liked": 1})
    assert not is_positive({"rating": 3.0})
    assert not is_positive({"rating": None, "watched": 1})


def test_metrics_known_values():
    relevant = {"x", "y"}
    recs = ["x", "a", "y", "b"]
    assert recall_at_k(recs, relevant, 2) == pytest.approx(0.5)
    assert recall_at_k(recs, relevant, 4) == pytest.approx(1.0)
    assert hit_at_k(["a", "b"], relevant, 2) == 0.0
    assert hit_at_k(recs, relevant, 1) == 1.0
    # DCG = 1/log2(2) + 1/log2(4) = 1.5 ; IDCG = 1 + 1/log2(3)
    assert ndcg_at_k(recs, relevant, 4) == pytest.approx(1.5 / (1 + 1 / 1.5849625007211562))
    assert ndcg_at_k(["x", "y"], relevant, 2) == pytest.approx(1.0)
    assert recall_at_k(recs, set(), 5) == 0.0


def test_holdout_split_hides_rows_and_is_deterministic():
    data, films = _clustered_data()
    split1 = make_holdout_split(data, holdout_frac=0.2, min_positives=10, seed=7)
    split2 = make_holdout_split(data, holdout_frac=0.2, min_positives=10, seed=7)
    assert split1.test == split2.test
    assert len(split1.test) == len(data)

    for user, held in split1.test.items():
        train_slugs = {f["slug"] for f in split1.train[user]}
        assert held and not (held & train_slugs)
        assert all(is_positive(f) for f in data[user] if f["slug"] in held)
        assert len(train_slugs) + len(held) == len(data[user])

    # Original data is not mutated
    assert all(len(v) == 25 for v in data.values())


def test_holdout_split_respects_catalog_min_positives_and_users():
    data, _ = _clustered_data()
    catalog = {f"a-{i}" for i in range(30)}
    split = make_holdout_split(data, min_positives=10, catalog=catalog)
    # b-group users' positives are all outside the catalog
    assert all(u.startswith("a-") for u in split.test)
    assert all(s in catalog for held in split.test.values() for s in held)

    split = make_holdout_split(data, users=["a-user0", "nobody"])
    assert set(split.test) == {"a-user0"}

    split = make_holdout_split(data, max_users=5, min_positives=1)
    assert len(split.test) == 5


def test_run_evaluation_ranks_strategies(fresh_recommender_modules):
    data, films = _clustered_data()
    split, results = run_evaluation(
        data, films, strategies=["popularity", "svd", "collaborative", "metadata", "hybrid"],
        k=10, min_positives=10, seed=1,
    )
    by_name = {r.strategy: r for r in results}
    assert set(by_name) == {"popularity", "svd", "collaborative", "metadata", "hybrid"}
    for r in results:
        assert r.error is None, (r.strategy, r.error)
        assert r.n_users == len(split.test) == 30
        assert 0.0 <= r.recall <= 1.0 and 0.0 <= r.ndcg <= 1.0

        assert r.n_failed_users == 0, (r.strategy, r.first_failure)

    # With clean taste clusters, neighbour-based models must clearly beat global popularity
    assert by_name["collaborative"].ndcg > 2 * by_name["popularity"].ndcg
    assert by_name["hybrid"].ndcg > by_name["popularity"].ndcg

    table = format_results_table(results)
    assert "popularity" in table and "vs pop" in table


def test_run_evaluation_rejects_unknown_strategy():
    data, films = _clustered_data()
    with pytest.raises(ValueError):
        run_evaluation(data, films, strategies=["graph"])


def test_failing_strategy_is_reported_not_raised():
    from letterboxd_rec.evaluation import evaluate_strategy

    data, films = _clustered_data()
    split = make_holdout_split(data)

    def broken(train, films):
        raise RuntimeError("boom")

    result = evaluate_strategy("broken", split, films, builder=broken)
    assert result.error and "boom" in result.error


def test_per_user_failures_are_counted_and_shown():
    from letterboxd_rec.evaluation import evaluate_strategy

    data, films = _clustered_data()
    split = make_holdout_split(data)

    def crashes_per_user(train, films):
        def recommend(username, user_films, k):
            raise ValueError("bad user")
        return recommend

    result = evaluate_strategy("flaky", split, films, builder=crashes_per_user)
    assert result.error is None
    assert result.n_failed_users == result.n_users == len(split.test)
    assert "bad user" in result.first_failure
    assert "raised for" in format_results_table([result])
