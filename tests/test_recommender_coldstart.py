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

