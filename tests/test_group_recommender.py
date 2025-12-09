import pytest
from contextlib import contextmanager

from letterboxd_rec import group_recommender
from letterboxd_rec.group_recommender import (
    AggregationStrategy,
    GroupMember,
    GroupProfile,
    GroupRecommendation,
    GroupRecommender,
    recommend_for_group,
)
from letterboxd_rec.profile import UserProfile


@pytest.fixture(autouse=True)
def stub_metadata_recommender(monkeypatch):
    """Avoid hitting the real recommender/DB when constructing GroupRecommender."""

    class _StubMetadataRecommender:
        def __init__(self, films, **_kwargs):
            if isinstance(films, list):
                self.films = {f.get("slug", f"film-{idx}"): f for idx, f in enumerate(films)}
            else:
                self.films = {}

        def _score_film(self, film, profile):
            return 0.0, [], []

    monkeypatch.setattr(group_recommender, "MetadataRecommender", _StubMetadataRecommender)


def _profile(genres=None, directors=None, decades=None):
    prof = UserProfile()
    prof.genres = genres or {}
    prof.directors = directors or {}
    prof.decades = decades or {}
    return prof


def test_group_recommendation_flags_detect_unanimity_and_controversy():
    unanimous = GroupRecommendation(
        slug="a",
        title="A",
        year=2020,
        group_score=1.0,
        user_scores={"u1": 1.0, "u2": 2.0},
        consensus_reasons=[],
        user_reasons={},
        warnings=[],
        agreement_score=1.0,
        min_user_score=1.0,
        max_user_score=2.0,
    )
    assert unanimous.is_unanimous
    assert not unanimous.controversial

    split = GroupRecommendation(
        slug="b",
        title="B",
        year=2020,
        group_score=-0.5,
        user_scores={"u1": 5.0, "u2": -1.0},
        consensus_reasons=[],
        user_reasons={},
        warnings=[],
        agreement_score=0.0,
        min_user_score=-1.0,
        max_user_score=5.0,
    )
    assert not split.is_unanimous
    assert split.controversial


def test_aggregate_scores_cover_strategies():
    rec = GroupRecommender(all_films={}, strategy=AggregationStrategy.AVERAGE)
    scores = [4.0, 2.0]

    assert rec._aggregate_scores(scores, AggregationStrategy.LEAST_MISERY) == 2.0
    assert rec._aggregate_scores(scores, AggregationStrategy.MOST_PLEASURE) == 4.0
    assert rec._aggregate_scores(scores, AggregationStrategy.AVERAGE) == pytest.approx(3.0)
    assert rec._aggregate_scores(scores, AggregationStrategy.FAIRNESS) == pytest.approx(2.5)
    assert rec._aggregate_scores([2.0, 2.0], AggregationStrategy.MULTIPLICATIVE) == pytest.approx(2.0)
    assert rec._aggregate_scores([2.0, -1.0], AggregationStrategy.MULTIPLICATIVE) == 0.0

    approval = rec._aggregate_scores([2.5, 1.5], AggregationStrategy.APPROVAL)
    assert approval == pytest.approx(1.2)


def test_find_consensus_reasons_labels_and_limit():
    rec = GroupRecommender(all_films={})
    members = [
        GroupMember("alice", UserProfile(), []),
        GroupMember("bob", UserProfile(), []),
    ]

    user_reasons = {
        "alice": ["Director: Nolan", "Genre: horror", "Extra: keep"],
        "bob": ["Genre: horror: detail", "Director: Scott", "Another"],
    }

    consensus = rec._find_consensus_reasons(user_reasons, members)

    assert any("Genre (all)" in reason for reason in consensus)
    assert any("(1/2)" in reason for reason in consensus)
    assert len(consensus) == 3  # truncated to top 3


def test_compute_compatibility_blends_components():
    rec = GroupRecommender(all_films={})
    p1 = _profile(genres={"horror": 1.0, "comedy": -1.0}, directors={"nolan": 1.0, "spielberg": -1.0}, decades={1990: 1.0})
    p2 = _profile(genres={"horror": 1.0, "comedy": 1.0}, directors={"nolan": 1.0, "spielberg": 1.0}, decades={1990: 1.0})

    compat = rec._compute_compatibility(p1, p2)
    assert compat == pytest.approx(2 / 3, rel=1e-2)

    assert rec._compute_compatibility(_profile(), _profile()) == 0.5


def test_build_group_profile_sets_consensus_and_divisive_and_watchlist():
    rec = GroupRecommender(all_films={})

    m1 = GroupMember(
        "alice",
        _profile(genres={"horror": 2.0, "comedy": 2.5}, directors={"dirA": 1.5, "dirC": 3.1}, decades={1990: 1.0}),
        films=[{"slug": "shared-film", "watchlisted": True}, {"slug": "seen", "watched": True}],
    )
    m2 = GroupMember(
        "bob",
        _profile(genres={"horror": 1.5, "comedy": -1.2}, directors={"dirA": 1.0, "dirC": -1.3}, decades={1990: 0.5}),
        films=[{"slug": "shared-film", "watchlisted": True}],
    )

    group = rec._build_group_profile([m1, m2])

    assert group.consensus_genres["horror"] == pytest.approx(1.5)
    assert "comedy" in group.divisive_genres
    assert group.consensus_directors["dirA"] == pytest.approx(1.0)
    assert "dirC" in group.divisive_directors
    assert group.shared_watchlist["shared-film"] == 2
    assert group.pairwise_compatibility[(m1.username, m2.username)] > 0
    assert group.overall_compatibility > 0


def test_recommend_filters_and_boosts_shared_watchlist():
    all_films = {
        "seen": {"slug": "seen", "title": "Seen", "year": 2018, "genres": ["drama"]},
        "shared": {"slug": "shared", "title": "Shared", "year": 2016, "genres": ["drama"]},
        "divisive": {"slug": "divisive", "title": "Divisive", "year": 2017, "genres": ["comedy"]},
        "old": {"slug": "old", "title": "Old", "year": 1990, "genres": ["drama"]},
        "future": {"slug": "future", "title": "Future", "year": 2030, "genres": ["drama"]},
        "genre_filtered": {"slug": "genre_filtered", "title": "Genre", "year": 2015, "genres": ["romance"]},
        "negative": {"slug": "negative", "title": "Negative", "year": 2012, "genres": ["drama"]},
        "ok": {"slug": "ok", "title": "OK", "year": 2015, "genres": ["drama"]},
    }

    base_scores = {"shared": 2.0, "ok": 1.0, "negative": -1.0}

    class StubRecommender:
        def __init__(self, scores):
            self.scores = scores

        def _score_film(self, film, profile):
            slug = film["slug"]
            score = self.scores.get(slug, 1.0)
            return score, [f"reason-{slug}"], []

    rec = GroupRecommender(all_films=all_films, strategy=AggregationStrategy.AVERAGE)
    rec._recommender = StubRecommender(base_scores)

    members = [
        GroupMember("alice", UserProfile(), films=[{"slug": "seen", "watched": True}, {"slug": "shared", "watchlisted": True}]),
        GroupMember("bob", UserProfile(), films=[{"slug": "shared", "watchlisted": True}]),
    ]
    group = GroupProfile(
        members=members,
        divisive_genres={"comedy"},
        divisive_directors=set(),
        shared_watchlist={"shared": 2},
    )

    results = rec.recommend(
        group,
        n=5,
        min_year=2000,
        max_year=2025,
        genres=["drama"],
        exclude_divisive=True,
        prioritize_shared_watchlist=True,
    )

    slugs = [r.slug for r in results]
    assert slugs == ["shared", "ok", "negative"]
    assert results[0].group_score == pytest.approx(2.8)  # boosted by shared watchlist
    assert results[-1].warnings  # negative score triggers warnings per user


def test_triage_watchlists_calls_recommend(monkeypatch):
    rec = GroupRecommender(all_films={})
    called = {}

    def fake_recommend(group, n=20, prioritize_shared_watchlist=False, **kwargs):
        called["n"] = n
        called["prioritize"] = prioritize_shared_watchlist
        return ["ok"]

    monkeypatch.setattr(rec, "recommend", fake_recommend)

    empty_group = GroupProfile(members=[])
    assert rec.triage_watchlists(empty_group) == []

    group_with_list = GroupProfile(members=[], shared_watchlist={"x": 2})
    assert rec.triage_watchlists(group_with_list, n=3) == ["ok"]
    assert called["n"] == 3
    assert called["prioritize"] is True


def test_group_recommendation_hint_branches_and_explain_group():
    rec = GroupRecommender(all_films={})

    group = GroupProfile(
        members=[
            GroupMember("a", UserProfile(), []),
            GroupMember("b", UserProfile(), []),
            GroupMember("c", UserProfile(), []),
        ],
        consensus_genres={"drama": 1.0},
        divisive_genres={"horror"},
        shared_watchlist={"x": 2},
    )
    group.pairwise_compatibility = {("a", "b"): 0.9, ("a", "c"): 0.3}
    group.overall_compatibility = 0.65

    info = rec.explain_group(group)
    assert info["compatibility_label"] == "Good compatibility"
    assert info["best_pair"].startswith("a & b")
    assert info["challenging_pair"].startswith("a & c")
    assert "Consider avoiding" in info["recommendation"]

    adventurous = GroupProfile(members=[], overall_compatibility=0.8)
    assert "adventurous" in rec._group_recommendation_hint(adventurous)

    shared_hint = GroupProfile(members=[], divisive_genres=set(), shared_watchlist={"x": 1}, overall_compatibility=0.5)
    assert "shared watchlist" in rec._group_recommendation_hint(shared_hint)

    fallback = GroupProfile(members=[], divisive_genres=set(), shared_watchlist={}, overall_compatibility=0.5)
    assert "crowd-pleasers" in rec._group_recommendation_hint(fallback)


def test_recommend_for_group_uses_injected_dependencies(monkeypatch):
    rows = [
        {
            "slug": "film-a",
            "title": "Film A",
            "year": 2020,
            "genres": [],
            "directors": [],
            "cast": [],
            "themes": [],
            "avg_rating": 4.0,
            "rating_count": 10,
        }
    ]

    class FakeConn:
        def execute(self, *_args, **_kwargs):
            return rows

    @contextmanager
    def fake_get_db(read_only=True):
        yield FakeConn()

    monkeypatch.setattr(group_recommender, "get_db", fake_get_db)

    info_stub = {"info": True}
    recs_stub = ["rec"]

    def fake_create_group(self, usernames, weights=None):
        members = [GroupMember(u, UserProfile(), []) for u in usernames]
        return GroupProfile(members=members)

    def fake_recommend(self, group, n=20, **filters):
        return recs_stub

    def fake_explain_group(self, group):
        return info_stub

    monkeypatch.setattr(GroupRecommender, "create_group", fake_create_group, raising=False)
    monkeypatch.setattr(GroupRecommender, "recommend", fake_recommend, raising=False)
    monkeypatch.setattr(GroupRecommender, "explain_group", fake_explain_group, raising=False)

    recs, info = recommend_for_group(["alice", "bob"], n=1, strategy="average")
    assert recs == recs_stub
    assert info == info_stub


def test_create_group_requires_two_users(monkeypatch):
    @contextmanager
    def fake_get_db(read_only=True):
        class FakeConn:
            def execute(self, *_args, **_kwargs):
                return []

        yield FakeConn()

    monkeypatch.setattr(group_recommender, "get_db", fake_get_db)

    rec = GroupRecommender(all_films={"dummy": {}})
    with pytest.raises(ValueError):
        rec.create_group(["only_one_user"])

