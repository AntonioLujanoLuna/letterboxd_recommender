from letterboxd_rec import recommender as rec_module


def test_recommend_from_candidates_cold_start_uses_tfidf(fresh_recommender_modules):
    rec, _, _ = fresh_recommender_modules

    films = [
        {
            "slug": "watched-film",
            "title": "Watched Film",
            "genres": ["drama"],
            "directors": ["DirA"],
            "cast": [],
            "themes": [],
            "languages": [],
            "countries": [],
            "writers": [],
            "cinematographers": [],
            "composers": [],
            "avg_rating": 4.0,
            "rating_count": 100,
            "year": 2020,
        },
        {
            "slug": "candidate-one",
            "title": "Candidate One",
            "genres": ["drama"],
            "directors": ["DirA"],
            "cast": [],
            "themes": [],
            "languages": [],
            "countries": [],
            "writers": [],
            "cinematographers": [],
            "composers": [],
            "avg_rating": 4.1,
            "rating_count": 50,
            "year": 2021,
        },
    ]

    metadata_rec = rec.MetadataRecommender(films)
    user_films = [{"slug": "watched-film", "watched": True}]

    recs = metadata_rec.recommend_from_candidates(user_films, ["candidate-one"], n=1)

    assert recs
    assert recs[0].slug == "candidate-one"


def test_recommend_from_candidates_watchlist_triggers_cold_start(fresh_recommender_modules):
    rec, _, _ = fresh_recommender_modules

    films = [
        {
            "slug": "watchlisted-film",
            "title": "Watchlisted Film",
            "genres": ["thriller"],
            "directors": ["DirA"],
            "cast": [],
            "themes": [],
            "languages": [],
            "countries": [],
            "writers": [],
            "cinematographers": [],
            "composers": [],
            "avg_rating": 3.9,
            "rating_count": 80,
            "year": 2019,
        },
        {
            "slug": "candidate-one",
            "title": "Candidate One",
            "genres": ["thriller"],
            "directors": ["DirA"],
            "cast": [],
            "themes": [],
            "languages": [],
            "countries": [],
            "writers": [],
            "cinematographers": [],
            "composers": [],
            "avg_rating": 4.1,
            "rating_count": 120,
            "year": 2020,
        },
    ]

    metadata_rec = rec.MetadataRecommender(films)
    user_films = [{"slug": "watchlisted-film", "watchlisted": True}]

    recs = metadata_rec.recommend_from_candidates(user_films, ["candidate-one"], n=1)

    assert recs
    assert recs[0].slug == "candidate-one"
