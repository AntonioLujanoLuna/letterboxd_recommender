"""Regression tests for the SVD implicit-signal fix, cache keys, item similarity and scraper backoff."""

import random
from collections import defaultdict

import numpy as np
import pytest

from letterboxd_rec.matrix_factorization import SVDRecommender


# --- SVD implicit feedback -------------------------------------------------

def _svd_data():
    return {
        "alice": [
            {"slug": "a", "rating": 4.0},
            {"slug": "b", "rating": 4.0},
            {"slug": "liked-only", "rating": None, "liked": 1, "watched": 1},
        ],
        "bob": [
            {"slug": "a", "rating": 5.0},
            {"slug": "b", "rating": 3.0},
            {"slug": "c", "rating": 4.0},
        ],
        "carol": [
            {"slug": "b", "rating": 4.0},
            {"slug": "c", "rating": 4.0},
            {"slug": "liked-only", "rating": None, "liked": 1},
        ],
    }


def test_svd_biases_ignore_implicit_signals():
    svd = SVDRecommender(n_factors=1).fit(_svd_data())
    explicit = [4.0, 4.0, 5.0, 3.0, 4.0, 4.0, 4.0]
    # Previously an unrated like was stored as 4.5 * 0.3 = 1.35 and dragged the mean down
    assert svd.global_mean == pytest.approx(np.mean(explicit))
    # A film with only implicit signals has no rating bias
    assert svd.item_biases[svd.item_index["liked-only"]] == pytest.approx(0.0)
    assert svd.user_biases[svd.user_index["alice"]] == pytest.approx(4.0 - np.mean(explicit))


def test_svd_unrated_like_is_not_read_as_dislike():
    data = _svd_data()
    svd = SVDRecommender(n_factors=2).fit(data)
    # Reconstruct the residual a liked-only entry was trained on: must be positive
    u, i = svd.user_index["alice"], svd.item_index["liked-only"]
    pred = svd.user_biases[u] + svd.item_biases[i] + svd.global_mean + svd.user_factors[u] @ svd.item_factors[:, i]
    baseline = svd.global_mean + svd.user_biases[u] + svd.item_biases[i]
    assert pred >= baseline - 1e-9


def test_svd_fingerprint_changes_when_rating_changes_with_same_counts():
    data = _svd_data()
    fp1 = SVDRecommender.compute_fingerprint(data)
    data["bob"][1]["rating"] = 1.0
    fp2 = SVDRecommender.compute_fingerprint(data)
    assert (fp1["n_users"], fp1["n_items"], fp1["n_ratings"]) == (fp2["n_users"], fp2["n_items"], fp2["n_ratings"])
    assert fp1 != fp2
    # Flag-only change (like added) also invalidates
    data["bob"][2]["liked"] = 1
    assert SVDRecommender.compute_fingerprint(data) != fp2


def test_svd_fingerprint_is_order_independent():
    data = _svd_data()
    reordered = {u: list(reversed(fs)) for u, fs in reversed(list(data.items()))}
    assert SVDRecommender.compute_fingerprint(data) == SVDRecommender.compute_fingerprint(reordered)


# --- Item-item similarity --------------------------------------------------

def _reference_item_similarities(all_user_films, user_index, min_ratings, top_k):
    """The original pure-Python adjusted-cosine loop, kept as an oracle."""
    item_ratings = defaultdict(list)
    for username, films in all_user_films.items():
        for f in films:
            if f.get("rating"):
                item_ratings[f["slug"]].append((user_index[username], f["rating"]))
    item_ratings = {s: r for s, r in item_ratings.items() if len(r) >= min_ratings}
    user_means = {}
    for username, films in all_user_films.items():
        ratings = [f["rating"] for f in films if f.get("rating")]
        if ratings:
            user_means[user_index[username]] = sum(ratings) / len(ratings)
    slugs = list(item_ratings)
    sim_map = defaultdict(list)
    for i, a in enumerate(slugs):
        ra = dict(item_ratings[a])
        for b in slugs[i + 1:]:
            rb = dict(item_ratings[b])
            common = set(ra) & set(rb)
            if len(common) < 2:
                continue
            dot = na = nb = 0.0
            for u in common:
                xa, xb = ra[u] - user_means[u], rb[u] - user_means[u]
                dot += xa * xb
                na += xa * xa
                nb += xb * xb
            if na > 0 and nb > 0:
                sim = dot / (np.sqrt(na) * np.sqrt(nb))
                if sim > 0.3:
                    sim_map[a].append((b, sim))
                    sim_map[b].append((a, sim))
    return {s: sorted(v, key=lambda x: -x[1])[:top_k] for s, v in sim_map.items()}


@pytest.mark.parametrize("block_cells", [10_000_000, 7])  # one block, and many tiny blocks
def test_vectorized_item_similarity_matches_reference(fresh_recommender_modules, monkeypatch, block_cells):
    recommender, _, _ = fresh_recommender_modules
    import letterboxd_rec.recommenders.collaborative as collab_mod
    monkeypatch.setattr(collab_mod, "ITEM_SIM_BLOCK_CELLS", block_cells)
    rng = random.Random(3)
    films = [f"film-{i}" for i in range(40)]
    data = {
        f"user{u}": [
            {"slug": s, "rating": rng.choice([1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])}
            for s in rng.sample(films, rng.randint(5, 25))
        ]
        for u in range(30)
    }
    collab = recommender.CollaborativeRecommender(data, {}, use_cache=False)
    collab._item_sim_top_k = 10
    collab._compute_item_similarities()
    expected = _reference_item_similarities(data, collab._user_index, collab_mod.ITEM_SIM_MIN_RATINGS, 10)

    assert set(collab._item_similarities) == set(expected)
    for slug, sims in expected.items():
        got = dict(collab._item_similarities[slug])
        exp = dict(sims)
        # Ties at the top-k boundary may pick different neighbours; scores must agree
        assert sorted(got.values(), reverse=True) == pytest.approx(sorted(exp.values(), reverse=True))
        for nbr in set(got) & set(exp):
            assert got[nbr] == pytest.approx(exp[nbr])


def test_find_neighbors_with_more_users_than_k(fresh_recommender_modules):
    """Regression: sparse @ sparse result was wrapped by np.asarray and crashed once n_users > k."""
    recommender, _, _ = fresh_recommender_modules
    rng = random.Random(5)
    films = [f"film-{i}" for i in range(30)]
    data = {
        f"user{u}": [{"slug": s, "rating": rng.choice([2.0, 3.0, 4.0, 5.0])} for s in rng.sample(films, 15)]
        for u in range(25)
    }
    collab = recommender.CollaborativeRecommender(data, {}, use_cache=False)
    neighbors = collab._find_neighbors("user0", data["user0"], k=5)
    assert 0 < len(neighbors) <= 5
    assert all(name != "user0" and sim > 0 for name, sim in neighbors)
    assert [s for _, s in neighbors] == sorted((s for _, s in neighbors), reverse=True)
    assert collab.recommend("user0", n=5)


def test_collaborative_use_cache_false_does_not_write(fresh_recommender_modules, tmp_path, monkeypatch):
    recommender, _, _ = fresh_recommender_modules
    import letterboxd_rec.recommenders.collaborative as collab_mod
    cache_path = tmp_path / "real_item_sim_cache.json"
    monkeypatch.setattr(collab_mod, "ITEM_SIM_CACHE_PATH", cache_path)  # the path the code really uses
    data = {
        "u1": [{"slug": "a", "rating": 4.0}, {"slug": "b", "rating": 5.0}],
        "u2": [{"slug": "a", "rating": 2.0}, {"slug": "b", "rating": 3.0}],
    }
    collab = recommender.CollaborativeRecommender(data, {}, use_cache=False)
    collab._compute_item_similarities()
    assert not cache_path.exists()


# --- Graph cache data version ----------------------------------------------

def test_graph_cache_rejects_different_data_version(tmp_path):
    from letterboxd_rec.graph_config import GraphConfig
    from letterboxd_rec.graph_recommender import HeterogeneousGraph, NodeType

    config = GraphConfig()
    graph = HeterogeneousGraph(config)
    a = graph.add_node(NodeType.USER, "alice")
    b = graph.add_node(NodeType.FILM, "film-a")
    graph.add_edge(a, b, "liked", 1.0)
    graph.add_edge(b, a, "liked", 1.0)
    graph.build_transition_matrix()

    path = tmp_path / "graph.npz"
    graph.save(path, data_version="v-one")
    assert HeterogeneousGraph.load(path, config, data_version="v-one") is not None
    assert HeterogeneousGraph.load(path, config, data_version="v-two") is None


def test_data_version_changes_on_rating_update(fresh_db):
    fresh_db.init_db()
    with fresh_db.get_db() as conn:
        conn.execute(
            "INSERT INTO user_films (username, film_slug, rating, watched, scraped_at) "
            "VALUES ('alice', 'film-a', 4.0, 1, '2025-01-01')"
        )
    with fresh_db.get_db(read_only=True) as conn:
        v1 = fresh_db.get_data_version(conn)
        assert fresh_db.get_data_version(conn) == v1
    with fresh_db.get_db() as conn:
        conn.execute("UPDATE user_films SET rating = 2.0 WHERE username = 'alice'")
    with fresh_db.get_db(read_only=True) as conn:
        assert fresh_db.get_data_version(conn) != v1


# --- Async scraper backoff -------------------------------------------------

def test_async_scraper_delay_recovers_after_backoff():
    from letterboxd_rec.scraper import AsyncLetterboxdScraper

    scraper = AsyncLetterboxdScraper(delay=0.2)
    scraper.delay = 5.0  # as if several 429s were hit
    for _ in range(40):
        scraper._decay_delay()
    assert scraper.delay == pytest.approx(0.2, abs=0.01)
    assert scraper.delay >= 0.2


def test_svd_item_bias_shrinkage_demotes_single_rating_films():
    # "classic" is rated 4.5 by ten users; "obscure" got one 5-star rating
    data = {f"u{i}": [{"slug": "classic", "rating": 4.5}, {"slug": "filler", "rating": 3.0}] for i in range(10)}
    data["u0"].append({"slug": "obscure", "rating": 5.0})
    data["target"] = [{"slug": "filler", "rating": 3.0}]

    undamped = SVDRecommender(n_factors=1, item_bias_reg=0).fit(data)
    damped = SVDRecommender(n_factors=1, item_bias_reg=25).fit(data)
    idx = damped.item_index
    assert undamped.item_biases[idx["obscure"]] > undamped.item_biases[idx["classic"]]
    assert damped.item_biases[idx["classic"]] > damped.item_biases[idx["obscure"]]
    assert damped.recommend("target", {"filler"}, n=1)[0][0] == "classic"


# --- Blocked profile pages -------------------------------------------------

def test_scrape_user_raises_when_films_listing_is_blocked(monkeypatch):
    from letterboxd_rec.scraper import LetterboxdScraper, ScrapeIncompleteError

    scraper = LetterboxdScraper(delay=0)
    monkeypatch.setattr(scraper, "_get_with_soft_block_recovery", lambda url: None)  # e.g. 403
    with pytest.raises(ScrapeIncompleteError):
        scraper.scrape_user("someone")
    with pytest.raises(ScrapeIncompleteError):
        scraper.scrape_user_smart("someone")
    scraper.close()


def test_async_scrape_user_raises_when_films_listing_is_blocked(monkeypatch):
    import asyncio
    from letterboxd_rec.scraper import AsyncLetterboxdScraper, ScrapeIncompleteError

    scraper = AsyncLetterboxdScraper(delay=0)

    async def blocked(url):
        return None

    monkeypatch.setattr(scraper, "_get_with_soft_block_recovery", blocked)
    with pytest.raises(ScrapeIncompleteError):
        asyncio.run(scraper.scrape_user("someone"))
    with pytest.raises(ScrapeIncompleteError):
        asyncio.run(scraper.scrape_user_smart("someone"))
