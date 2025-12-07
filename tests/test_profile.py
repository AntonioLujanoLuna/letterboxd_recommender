from datetime import datetime, timedelta


def test_profile_weights_lists_and_interactions(fresh_recommender_modules):
    _, profile, _ = fresh_recommender_modules

    film_metadata = {
        "fav-film": {
            "slug": "fav-film",
            "title": "Fav Film",
            "genres": ["drama"],
            "directors": ["Director One"],
            "cast": ["Lead Actor"],
            "themes": [],
            "languages": ["english"],
            "writers": ["Writer One"],
            "cinematographers": [],
            "composers": [],
            "countries": ["USA"],
            "year": 2022,
        },
        "recent-horror": {
            "slug": "recent-horror",
            "title": "Recent Horror",
            "genres": ["horror"],
            "directors": ["Director Two"],
            "cast": [],
            "themes": [],
            "languages": ["english"],
            "writers": [],
            "cinematographers": [],
            "composers": [],
            "countries": ["Canada"],
            "year": 2021,
        },
    }

    user_films = [
        {"slug": "fav-film", "rating": 4.0, "watched": True, "liked": True, "scraped_at": "2023-01-01T00:00:00"},
        {"slug": "recent-horror", "rating": 3.0, "watched": True, "liked": False, "scraped_at": "2023-01-02T00:00:00"},
    ]

    user_lists = [
        {
            "username": "tester",
            "list_slug": "profile-favorites",
            "list_name": "Profile Favorites",
            "is_ranked": 0,
            "is_favorites": 1,
            "position": None,
            "film_slug": "fav-film",
        }
    ]

    profile_obj = profile.build_profile(
        user_films,
        film_metadata,
        user_lists=user_lists,
        use_temporal_decay=False,
        weighting_mode="absolute",
    )

    # Favorites multiplier should make drama preference stronger than horror
    assert profile_obj.genres["drama"] > profile_obj.genres["horror"]
    assert profile_obj.directors["Director One"] > 0
    assert profile_obj.n_films == 2
    assert profile_obj.n_liked == 1


def test_temporal_decay_reduces_old_weights(fresh_recommender_modules):
    _, profile, _ = fresh_recommender_modules
    ref_time = datetime(2024, 1, 1)

    film_metadata = {
        "old-classic": {
            "slug": "old-classic",
            "title": "Old Classic",
            "genres": ["noir"],
            "directors": ["Veteran"],
            "cast": [],
            "themes": [],
            "languages": [],
            "writers": [],
            "cinematographers": [],
            "composers": [],
            "countries": [],
            "year": 1960,
        },
        "new-hit": {
            "slug": "new-hit",
            "title": "New Hit",
            "genres": ["sci-fi"],
            "directors": ["Modern"],
            "cast": [],
            "themes": [],
            "languages": [],
            "writers": [],
            "cinematographers": [],
            "composers": [],
            "countries": [],
            "year": 2023,
        },
    }

    user_films = [
        {"slug": "old-classic", "rating": 4.5, "watched": True, "scraped_at": "2010-01-01T00:00:00"},
        {"slug": "new-hit", "rating": 4.5, "watched": True, "scraped_at": ref_time.isoformat()},
    ]

    profile_obj = profile.build_profile(
        user_films,
        film_metadata,
        reference_time=ref_time,
        use_temporal_decay=True,
    )

    assert profile_obj.genres["sci-fi"] > profile_obj.genres["noir"]


def test_weighting_modes_normalized_and_blended(fresh_recommender_modules):
    _, profile, _ = fresh_recommender_modules

    film_metadata = {
        "top": {
            "slug": "top",
            "title": "Top",
            "genres": ["top-genre"],
            "directors": [],
            "cast": [],
            "themes": [],
            "languages": [],
            "writers": [],
            "cinematographers": [],
            "composers": [],
            "countries": [],
            "year": 2022,
        },
        "low": {
            "slug": "low",
            "title": "Low",
            "genres": ["low-genre"],
            "directors": [],
            "cast": [],
            "themes": [],
            "languages": [],
            "writers": [],
            "cinematographers": [],
            "composers": [],
            "countries": [],
            "year": 2022,
        },
    }

    user_films = [
        {"slug": "top", "rating": 5.0, "watched": True},
        {"slug": "low", "rating": 2.0, "watched": True},
    ]

    norm_profile = profile.build_profile(
        user_films, film_metadata, use_temporal_decay=False, weighting_mode="normalized"
    )
    assert norm_profile.genres["top-genre"] > 0
    assert norm_profile.genres["low-genre"] < 0

    blended_profile = profile.build_profile(
        user_films, film_metadata, use_temporal_decay=False, weighting_mode="blended"
    )
    assert blended_profile.genres["top-genre"] > blended_profile.genres["low-genre"]


def test_preference_momentum_detects_recent_shift(fresh_recommender_modules):
    _, profile, _ = fresh_recommender_modules

    now = datetime.now()
    recent = now - timedelta(days=10)
    older = now - timedelta(days=400)
    film_metadata = {
        "old1": {"slug": "old1", "genres": ["horror"], "directors": [], "cast": [], "themes": [], "languages": [], "writers": [], "cinematographers": [], "composers": [], "countries": [], "year": 2010},
        "old2": {"slug": "old2", "genres": ["horror"], "directors": [], "cast": [], "themes": [], "languages": [], "writers": [], "cinematographers": [], "composers": [], "countries": [], "year": 2011},
        "new1": {"slug": "new1", "genres": ["horror"], "directors": [], "cast": [], "themes": [], "languages": [], "writers": [], "cinematographers": [], "composers": [], "countries": [], "year": 2024},
        "new2": {"slug": "new2", "genres": ["horror"], "directors": [], "cast": [], "themes": [], "languages": [], "writers": [], "cinematographers": [], "composers": [], "countries": [], "year": 2024},
    }

    user_films = [
        {"slug": "old1", "rating": 2.0, "watched": True, "scraped_at": older.isoformat()},
        {"slug": "old2", "rating": 2.5, "watched": True, "scraped_at": (older + timedelta(days=20)).isoformat()},
        {"slug": "new1", "rating": 5.0, "watched": True, "scraped_at": recent.isoformat()},
        {"slug": "new2", "rating": 4.5, "watched": True, "scraped_at": (recent + timedelta(days=1)).isoformat()},
    ]

    profile_obj = profile.build_profile(
        user_films,
        film_metadata,
        use_temporal_decay=False,
    )

    momentum = profile_obj.preference_momentum.get("genre:horror")
    assert momentum is not None
    assert momentum >= 0  # recent ratings should not decrease momentum


def test_genre_interactions_capture_combo_penalty(fresh_recommender_modules):
    _, profile, _ = fresh_recommender_modules

    film_metadata = {
        "pair": {"slug": "pair", "genres": ["horror", "comedy"], "directors": [], "cast": [], "themes": [], "languages": [], "writers": [], "cinematographers": [], "composers": [], "countries": [], "year": 2020},
        "pair2": {"slug": "pair2", "genres": ["horror", "comedy"], "directors": [], "cast": [], "themes": [], "languages": [], "writers": [], "cinematographers": [], "composers": [], "countries": [], "year": 2020},
        "pair3": {"slug": "pair3", "genres": ["horror", "comedy"], "directors": [], "cast": [], "themes": [], "languages": [], "writers": [], "cinematographers": [], "composers": [], "countries": [], "year": 2020},
        "horror": {"slug": "horror", "genres": ["horror"], "directors": [], "cast": [], "themes": [], "languages": [], "writers": [], "cinematographers": [], "composers": [], "countries": [], "year": 2020},
        "comedy": {"slug": "comedy", "genres": ["comedy"], "directors": [], "cast": [], "themes": [], "languages": [], "writers": [], "cinematographers": [], "composers": [], "countries": [], "year": 2020},
    }

    user_films = [
        {"slug": "pair", "rating": 2.0, "watched": True},
        {"slug": "pair2", "rating": 2.5, "watched": True},
        {"slug": "pair3", "rating": 1.5, "watched": True},
        {"slug": "horror", "rating": 4.0, "watched": True},
        {"slug": "comedy", "rating": 4.0, "watched": True},
    ]

    profile_obj = profile.build_profile(
        user_films,
        film_metadata,
        use_temporal_decay=False,
    )

    interactions = profile_obj.genre_interactions
    combo_key = "comedy|horror"
    assert combo_key in interactions
    assert interactions[combo_key] < 0  # disliked combo compared to singles
