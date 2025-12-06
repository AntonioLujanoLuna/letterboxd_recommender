import httpx
from selectolax.parser import HTMLParser

from letterboxd_rec import scraper


def test_check_user_activity_parses_profile(monkeypatch):
    html = """
    <html>
      <a class="thousands" href="/alice/films/"><span>1,234</span><span>Films</span></a>
      <a href="/alice/films/ratings/rated/1/">ratings</a>
      <a href="/alice/film/some-movie/">diary link</a>
    </html>
    """

    lb = scraper.LetterboxdScraper(delay=0.0)
    monkeypatch.setattr(lb, "_get", lambda url: HTMLParser(html))

    activity = lb.check_user_activity("alice")
    assert activity["film_count"] == 1234
    assert activity["has_ratings"] is True
    assert activity["recent_activity"] is True
    lb.close()


def test_scrape_list_films_position_fallback(monkeypatch):
    html = """
    <html>
      <li class="poster-container">
        <div class="react-component" data-film-slug="film-a"></div>
        <span class="list-number">1.</span>
      </li>
      <li class="poster-container">
        <div class="react-component" data-film-slug="film-b"></div>
        <!-- missing list-number, should fallback to cumulative -->
      </li>
    </html>
    """

    lb = scraper.LetterboxdScraper(delay=0.0)
    # Return content on first page, then empty to break the loop
    call_count = {"n": 0}

    def fake_get(url):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return HTMLParser(html)
        return HTMLParser("<html></html>")

    monkeypatch.setattr(lb, "_get", fake_get)

    films = lb.scrape_list_films("user", "some-list")
    lb.close()

    assert films[0]["film_slug"] == "film-a"
    assert films[0]["position"] == 1
    assert films[1]["film_slug"] == "film-b"
    assert films[1]["position"] == 2


def test_get_returns_none_on_404(monkeypatch):
    lb = scraper.LetterboxdScraper(delay=0.0)

    class DummyResponse:
        def __init__(self, status):
            self.status_code = status
            self.headers = {}
            self.text = ""

        def raise_for_status(self):
            import httpx

            raise httpx.HTTPStatusError("error", request=None, response=None)

    class DummyClient:
        def get(self, url):
            return DummyResponse(404)
        def close(self):
            return None

    lb.client = DummyClient()
    result = lb._get("https://example.com")
    lb.close()
    assert result is None


def _html_with_names(names):
    return "<html>" + "".join(f"<a class='name' href='/{n}/'>{n}</a>" for n in names) + "</html>"


def test_scrape_followers_and_following_limit_and_dedupe(monkeypatch):
    lb = scraper.LetterboxdScraper(delay=0.0)

    def fake_get(url):
        if "page/1" in url:
            return HTMLParser(_html_with_names(["u1", "u2"]))
        if "page/2" in url:
            return HTMLParser(_html_with_names(["u2", "u3"]))
        return HTMLParser("<html></html>")

    monkeypatch.setattr(lb, "_get", fake_get)

    followers = lb.scrape_followers("alice", limit=2)
    following = lb.scrape_following("alice", limit=3)
    lb.close()

    assert followers == ["u1", "u2"]
    assert following == ["u1", "u2", "u3"]


def test_scrape_popular_members_paginates(monkeypatch):
    lb = scraper.LetterboxdScraper(delay=0.0)

    def fake_get(url):
        if "page/1" in url:
            return HTMLParser(_html_with_names(["p1", "p2"]))
        if "page/2" in url:
            return HTMLParser(_html_with_names(["p3"]))
        return HTMLParser("<html></html>")

    monkeypatch.setattr(lb, "_get", fake_get)

    users = lb.scrape_popular_members(limit=3)
    lb.close()

    assert users == ["p1", "p2", "p3"]


def test_scrape_film_reviewers_parses_fields(monkeypatch):
    lb = scraper.LetterboxdScraper(delay=0.0)
    html = """
    <html>
      <li class="film-detail">
        <a class="context" href="/u1/"></a>
        <span class="rating"></span>
        <span class="_nobr"><a>Jan 1</a></span>
      </li>
      <li class="film-detail">
        <a class="context" href="/u2/"></a>
      </li>
    </html>
    """

    def fake_get(url):
        if "page/1" in url:
            return HTMLParser(html)
        return HTMLParser("<html></html>")

    monkeypatch.setattr(lb, "_get", fake_get)

    reviewers = lb.scrape_film_reviewers("film-x", limit=5)
    lb.close()

    assert reviewers == [
        {"username": "u1", "has_rating": True, "review_date": "Jan 1"},
        {"username": "u2", "has_rating": False, "review_date": None},
    ]


def test_scrape_favorites_supports_fallbacks(monkeypatch):
    lb = scraper.LetterboxdScraper(delay=0.0)
    html = """
    <html>
      <section class="profile-favorites">
        <li class="poster-container">
          <div class="react-component" data-film-slug="film-a"></div>
        </li>
        <li class="poster-container">
          <div data-film-slug="film-b"></div>
        </li>
      </section>
    </html>
    """
    monkeypatch.setattr(lb, "_get", lambda url: HTMLParser(html))

    favorites = lb.scrape_favorites("alice")
    lb.close()

    assert favorites == ["film-a", "film-b"]


def test_scrape_user_lists_marks_ranked(monkeypatch):
    lb = scraper.LetterboxdScraper(delay=0.0)
    html_top = """
    <html>
      <section class="list-summary">
        <h2><a href="/user/list/top-10/">Top 10</a></h2>
        <span class="icon-numbered"></span>
      </section>
    </html>
    """
    html_other = """
    <html>
      <section class="film-list-summary">
        <h3><a href="/user/list/random-list/">Random</a></h3>
      </section>
    </html>
    """

    def fake_get(url):
        if "page/1" in url:
            return HTMLParser(html_top)
        if "page/2" in url:
            return HTMLParser(html_other)
        return HTMLParser("<html></html>")

    monkeypatch.setattr(lb, "_get", fake_get)

    lists = lb.scrape_user_lists("alice", limit=5)
    lb.close()

    assert lists == [
        {"list_slug": "top-10", "list_name": "Top 10", "is_ranked": True},
        {"list_slug": "random-list", "list_name": "Random", "is_ranked": False},
    ]