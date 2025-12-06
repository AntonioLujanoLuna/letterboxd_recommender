import httpx
import pytest
from selectolax.parser import HTMLParser

from letterboxd_rec import scraper


def test_validate_slug_and_rating_count():
    assert scraper.validate_slug("the-matrix") == "the-matrix"
    assert scraper.validate_slug("The Matrix") is None

    assert scraper._parse_rating_count("1.5M") == 1_500_000
    assert scraper._parse_rating_count("12,345") == 12345


def test_parse_film_page_extracts_metadata():
    html = """
    <html>
      <h1 class="headline-1">Test Film</h1>
      <small class="number"><a>2023</a></small>
      <a href="/director/someone/">Director One</a>
      <a href="/films/genre/horror/">Horror</a>
      <a href="/actor/lead/">Lead Actor</a>
      <a href="/films/theme/ghosts/">Ghosts</a>
      <p class="text-link text-footer">120 mins</p>
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
    assert meta.runtime == 120
    assert meta.avg_rating == 4.2
    assert meta.rating_count == 1234
    assert meta.countries == ["USA"]
    assert meta.languages == ["English"]
    assert meta.writers == ["Scribe"]
    assert meta.cinematographers == ["Cine"]
    assert meta.composers == ["Music"]


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

