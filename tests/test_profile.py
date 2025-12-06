from datetime import datetime


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

