def _film(
    slug: str,
    title: str,
    genres=None,
    directors=None,
    cast=None,
    themes=None,
    year=2020,
    rating_count=1000,
):
    return {
        "slug": slug,
        "title": title,
        "genres": genres or [],
        "directors": directors or [],
        "cast": cast or [],
        "themes": themes or [],
        "languages": [],
        "countries": [],
        "writers": [],
        "cinematographers": [],
        "composers": [],
        "avg_rating": 4.0,
        "rating_count": rating_count,
        "year": year,
    }


def test_metadata_recommender_ranks_stronger_matches_first(fresh_recommender_modules):
    rec, _, _ = fresh_recommender_modules

    films = [
        _film("film-a", "Film A", genres=["horror"], directors=["DirA"]),
        _film("film-b", "Film B", genres=["horror"], directors=["DirA"]),  # same director/genre as seen film
        _film("film-c", "Film C", genres=["horror"], directors=["DirB"]),  # different director, same genre
    ]
    user_films = [{"slug": "film-a", "rating": 4.5, "watched": True}]

    metadata_rec = rec.MetadataRecommender(films)
    recs = metadata_rec.recommend(user_films, n=2, diversity=False)

    assert recs
    assert recs[0].slug == "film-b"


def test_metadata_recommender_diversity_limits_director_concentration(fresh_recommender_modules):
    rec, _, _ = fresh_recommender_modules

    films = [
        _film("film-a", "Film A", genres=["horror"], directors=["DirA"]),
        _film("film-b", "Film B", genres=["horror"], directors=["DirA"]),
        _film("film-c", "Film C", genres=["horror"], directors=["DirB"]),
    ]
    user_films = [{"slug": "film-a", "rating": 5.0, "watched": True}]

    metadata_rec = rec.MetadataRecommender(films)
    diverse = metadata_rec.recommend(user_films, n=2, diversity=True, max_per_director=1)

    assert diverse
    # At most one recommendation per director when diversity is enabled
    directors = [
        metadata_rec.films[r.slug]["directors"][0]
        for r in diverse
        if metadata_rec.films.get(r.slug)
    ]
    assert len(set(directors)) == len(directors)


def test_collaborative_recommender_surfaces_neighbor_favorite(fresh_recommender_modules):
    rec, _, _ = fresh_recommender_modules

    all_user_films = {
        "alice": [
            {"slug": "film-a", "rating": 4.0, "watched": True},
        ],
        "bob": [
            {"slug": "film-a", "rating": 4.0, "watched": True},
            {"slug": "film-b", "rating": 5.0, "watched": True},
        ],
        "carol": [
            {"slug": "film-a", "rating": 4.0, "watched": True},
            {"slug": "film-b", "rating": 4.5, "watched": True},
        ],
    }

    films = {
        "film-a": _film("film-a", "Film A", genres=["drama"], directors=["DirA"]),
        "film-b": _film("film-b", "Film B", genres=["drama"], directors=["DirB"]),
    }

    collab = rec.CollaborativeRecommender(all_user_films, films)
    recs = collab.recommend("alice", n=1, min_neighbors=1)

    assert recs
    assert recs[0].slug == "film-b"


def test_similar_to_finds_item_matches(fresh_recommender_modules):
    rec, _, _ = fresh_recommender_modules

    films = [
        _film("anchor", "Anchor", genres=["horror"], directors=["DirX"], cast=["Lead"], themes=["ghosts"]),
        _film("neighbor", "Neighbor", genres=["horror"], directors=["DirX"], cast=["Lead"]),
        _film("distant", "Distant", genres=["comedy"], directors=["DirY"], year=1980),
    ]

    metadata_rec = rec.MetadataRecommender(films)
    sims = metadata_rec.similar_to("anchor", n=2)

    slugs = [r.slug for r in sims]
    assert "neighbor" in slugs
    assert "distant" not in slugs


def test_collaborative_recommender_item_fallback_when_neighbors_sparse(fresh_recommender_modules):
    rec, _, _ = fresh_recommender_modules

    all_user_films = {
        "alice": [
            {"slug": "film-a", "rating": 5.0, "watched": True},
        ],
        "bob": [
            {"slug": "film-a", "rating": 4.0, "watched": True},
            {"slug": "film-b", "rating": 4.5, "watched": True},
        ],
        "carol": [
            {"slug": "film-a", "rating": 4.0, "watched": True},
            {"slug": "film-b", "rating": 5.0, "watched": True},
        ],
    }

    films = {
        "film-a": _film("film-a", "Film A", genres=["drama"], directors=["DirA"]),
        "film-b": _film("film-b", "Film B", genres=["drama"], directors=["DirB"]),
    }

    # Require too many neighbors to force item-similarity fallback
    collab = rec.CollaborativeRecommender(all_user_films, films)
    recs = collab.recommend("alice", n=1, min_neighbors=5)

    assert recs
    assert recs[0].slug == "film-b"

