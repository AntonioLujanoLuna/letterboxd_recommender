import httpx
import pytest
from selectolax.parser import HTMLParser

from letterboxd_rec import scraper


def test_validate_slug_and_rating_count():
    assert scraper.validate_slug("the-matrix") == "the-matrix"
    assert scraper.validate_slug("film:482919") == "film:482919"
    assert scraper.validate_slug("Film:482919") == "film:482919"
    assert scraper.validate_slug("The Matrix") is None

    assert scraper._parse_rating_count("1.5M") == 1_500_000
    assert scraper._parse_rating_count("12,345") == 12345
    assert scraper._parse_rating_count("bad") is None
    assert scraper.validate_slug("x" * 300) is None


def test_parse_film_page_extracts_metadata():
    html = """
    <html>
      <h1 class="headline-1">Test Film</h1>
      <small class="number"><a>2023</a></small>
      <a href="/director/someone/">Director One</a>
      <a href="/films/genre/horror/">Horror</a>
      <a href="/films/genre/short/">Short</a>
      <a href="/actor/lead/">Lead Actor</a>
      <a href="/films/theme/ghosts/">Ghosts</a>
      <p class="text-link text-footer">35 mins</p>
      <meta name="twitter:data2" content="4.2 average rating" />
      <a href="/film/test/ratings/">1,234</a>
      <a href="/films/country/usa/">USA</a>
      <a href="/films/language/english/">English</a>
      <a href="/writer/scribe/">Scribe</a>
      <a href="/cinematography/cine/">Cine</a>
      <a href="/composer/music/">Music</a>
    </html>
    """
    tree = HTMLParser(html)
    meta = scraper.parse_film_page(tree, "test-film")

    assert meta.title == "Test Film"
    assert meta.year == 2023
    assert "Director One" in meta.directors
    assert "horror" in meta.genres
    assert meta.runtime == 35
    assert meta.is_short is True
    assert meta.is_animation is False
    assert meta.avg_rating == 4.2
    assert meta.rating_count == 1234
    assert meta.countries == ["USA"]
    assert meta.languages == ["English"]
    assert meta.writers == ["Scribe"]
    assert meta.cinematographers == ["Cine"]
    assert meta.composers == ["Music"]


def test_parse_film_page_uses_ldjson_for_rating_count():
    html = """
    <html>
      <h1 class="headline-1">LD JSON Film</h1>
      <meta property="og:title" content="LD JSON Film (2024)" />
      <script type="application/ld+json">
      /* <![CDATA[ */
      {
        "aggregateRating": {
          "ratingValue": 4.5,
          "ratingCount": 999
        }
      }
      /* ]]> */
      </script>
    </html>
    """
    tree = HTMLParser(html)
    meta = scraper.parse_film_page(tree, "ld-json-film")

    assert meta.avg_rating == 4.5
    assert meta.rating_count == 999


def test_parse_fan_count_from_html():
    html = """
    <section class="section ratings-histogram-chart">
      <a href="/film/test/ratings/" class="all-link more-link"> 26K fans</a>
    </section>
    """
    fan_count = scraper._parse_fan_count_from_html(html)
    assert fan_count == 26_000


@pytest.mark.asyncio
async def test_async_scraper_batch_uses_provided_client():
    film_html = "<html><h1 class='headline-1'>Mock Film</h1></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=film_html)

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as client:
        async_scraper = scraper.AsyncLetterboxdScraper(delay=0.0, max_concurrent=2)
        results = await async_scraper._scrape_batch_with_client(client, ["mock-film"])

    assert len(results) == 1
    assert results[0].slug == "mock-film"


def test_parse_rating_from_span():
    html = "<span class='rating rated-8'></span>"
    tree = HTMLParser(html)
    span = tree.css_first("span")

    lb_scraper = scraper.LetterboxdScraper(delay=0.0)
    try:
        rating = lb_scraper._parse_rating(span)
        assert rating == 4.0
    finally:
        lb_scraper.close()


def test_parse_rating_handles_outliers_and_bad_formats():
    lb_scraper = scraper.LetterboxdScraper(delay=0.0)
    try:
        # Above 5.0 should be rejected
        too_high = HTMLParser("<span class='rating rated-12'></span>").css_first("span")
        assert lb_scraper._parse_rating(too_high) is None

        # Non-numeric suffix should be rejected
        bad_format = HTMLParser("<span class='rating rated-xx'></span>").css_first("span")
        assert lb_scraper._parse_rating(bad_format) is None

        # No rated-* class should also return None
        no_rating = HTMLParser("<span class='rating other-class'></span>").css_first("span")
        assert lb_scraper._parse_rating(no_rating) is None
    finally:
        lb_scraper.close()


@pytest.mark.asyncio
async def test_async_scraper_batch_summarizes_failures(monkeypatch):
    async_scraper = scraper.AsyncLetterboxdScraper(delay=0.0, max_concurrent=2)

    async def fake_scrape(_client, slug):
        if slug == "ok":
            return scraper.FilmMetadata(
                slug="ok",
                title="Good",
                year=2023,
                directors=["A"],
                genres=["test"],
                cast=[],
                themes=[],
                runtime=None,
                avg_rating=None,
                rating_count=None,
                    fan_count=None,
                countries=[],
                languages=[],
                writers=[],
                cinematographers=[],
                    composers=[],
                    is_short=False,
                    is_animation=False,
            )
        if slug == "missing":
            return None
        raise RuntimeError("boom")

    monkeypatch.setattr(async_scraper, "_scrape_film_async", fake_scrape)

    # Provide a dummy client; the fake coroutine ignores it
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        results = await async_scraper._scrape_batch_with_client(client, ["ok", "missing", "err"])

    assert [film.slug for film in results] == ["ok"]


@pytest.mark.asyncio
async def test_async_scrape_user_parses_films_and_watchlist(monkeypatch):
    films_html = """
    <html>
      <ul>
        <li class="griditem">
          <div class="react-component" data-item-slug="film-a"></div>
          <p class="poster-viewingdata">
            <span class="rating rated-8"></span>
            <span class="like"></span>
          </p>
        </li>
      </ul>
    </html>
    """
    watchlist_html = """
    <html>
      <ul>
        <li class="griditem">
          <div class="react-component" data-item-slug="film-b"></div>
        </li>
      </ul>
    </html>
    """
    empty_html = "<html></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/films/page/1/" in url:
            return httpx.Response(200, text=films_html)
        if "/films/page/2/" in url:
            return httpx.Response(200, text=empty_html)
        if "/watchlist/page/1/" in url:
            return httpx.Response(200, text=watchlist_html)
        if "/watchlist/page/2/" in url:
            return httpx.Response(200, text=empty_html)
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    original_async_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_async_client(transport=transport, **kwargs))

    async with scraper.AsyncLetterboxdScraper(delay=0.0, max_concurrent=3) as async_scraper:
        interactions = await async_scraper.scrape_user("alice")

    by_slug = {i.film_slug: i for i in interactions}

    assert by_slug["film-a"].watched is True
    assert by_slug["film-a"].liked is True
    assert by_slug["film-a"].rating == 4.0

    assert by_slug["film-b"].watchlisted is True
    assert by_slug["film-b"].watched is False
    assert by_slug["film-b"].rating is None


def test_check_user_activity_parses_profile(monkeypatch):
    lb_scraper = scraper.LetterboxdScraper(delay=0.0)
    sample_html = """
    <html>
      <a class="thousands" href="/films/"><span>2,479</span><span>Films</span></a>
      <a href="/films/ratings/rated/1/">ratings</a>
      <a href="/alice/film/example/">recent</a>
    </html>
    """

    monkeypatch.setattr(lb_scraper, "_get", lambda _url: HTMLParser(sample_html))

    try:
        info = lb_scraper.check_user_activity("alice")
    finally:
        lb_scraper.close()

    assert info == {
        "film_count": 2479,
        "has_ratings": True,
        "recent_activity": True,
    }


def test_check_user_activity_handles_missing_page(monkeypatch):
    lb_scraper = scraper.LetterboxdScraper(delay=0.0)
    monkeypatch.setattr(lb_scraper, "_get_with_soft_block_recovery", lambda url: None)
    try:
        assert lb_scraper.check_user_activity("ghost") is None
    finally:
        lb_scraper.close()


def test_scrape_favorites_collects_showcase(monkeypatch):
    lb_scraper = scraper.LetterboxdScraper(delay=0.0)
    sample_html = """
    <html>
      <section class="profile-favorites">
        <li class="poster-container">
          <div class="react-component" data-film-slug="fav-react"></div>
        </li>
        <li class="poster-container">
          <div data-film-slug="fav-data"></div>
        </li>
      </section>
    </html>
    """
    monkeypatch.setattr(lb_scraper, "_get_with_soft_block_recovery", lambda url: HTMLParser(sample_html))

    try:
        favs = lb_scraper.scrape_favorites("alice")
    finally:
        lb_scraper.close()

    assert favs == ["fav-react", "fav-data"]


def test_scrape_user_lists_parses_ranked_and_unranked(monkeypatch):
    lb_scraper = scraper.LetterboxdScraper(delay=0.0)

    page_html = """
    <html>
      <section class="list-summary">
        <h2><a href="/alice/list/best-of-2020/">Best of 2020</a></h2>
        <span class="icon-numbered"></span>
      </section>
      <section class="list-summary">
        <h3><a href="/alice/list/watchlist/">Watchlist</a></h3>
      </section>
    </html>
    """

    def fake_get(url):
        if "/lists/page/1/" in url:
            return HTMLParser(page_html)
        return None

    monkeypatch.setattr(lb_scraper, "_get_with_soft_block_recovery", fake_get)

    try:
        lists = lb_scraper.scrape_user_lists("alice", limit=5)
    finally:
        lb_scraper.close()

    assert lists[0]["list_slug"] == "best-of-2020"
    assert lists[0]["is_ranked"] is True
    assert lists[1]["list_slug"] == "watchlist"
    assert lists[1]["is_ranked"] is False