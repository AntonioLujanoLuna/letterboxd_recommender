import httpx
import json
import time
from selectolax.parser import HTMLParser
from dataclasses import dataclass
from tqdm import tqdm
import asyncio

@dataclass
class FilmInteraction:
    film_slug: str
    rating: float | None
    watched: bool
    watchlisted: bool
    liked: bool

@dataclass
class FilmMetadata:
    slug: str
    title: str
    year: int | None
    directors: list[str]
    genres: list[str]
    cast: list[str]
    themes: list[str]
    runtime: int | None
    avg_rating: float | None
    rating_count: int | None
    # Phase 1 enhancements
    countries: list[str]
    languages: list[str]
    writers: list[str]
    cinematographers: list[str]
    composers: list[str]

class LetterboxdScraper:
    BASE = "https://letterboxd.com"
    
    def __init__(self, delay: float = 1.0):
        self.client = httpx.Client(
            headers={"User-Agent": "Mozilla/5.0 (compatible; film-rec/0.1)"},
            follow_redirects=True,
            timeout=30.0
        )
        self.delay = delay
    
    def _get(self, url: str, max_retries: int = 3) -> HTMLParser | None:
        time.sleep(self.delay)
        
        for attempt in range(max_retries):
            try:
                resp = self.client.get(url)
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                return HTMLParser(resp.text)
            except httpx.TimeoutException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Timeout on {url}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"Error: {url} - {e} (max retries exceeded)")
                    return None
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get("Retry-After", 60))
                    print(f"Rate limited (429) on {url}, waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue
                print(f"Error: {url} - {e}")
                return None
            except httpx.HTTPError as e:
                print(f"Error: {url} - {e}")
                return None
        
        return None
    
    def scrape_user(self, username: str) -> list[FilmInteraction]:
        """Scrape all film interactions for a user."""
        films = {}
        
        # Get watched films with ratings
        print(f"Scraping {username}'s films...")
        page = 1
        while True:
            tree = self._get(f"{self.BASE}/{username}/films/page/{page}/")
            if not tree:
                break
            
            # Updated selector for grid items
            items = tree.css("li.griditem")
            if not items:
                break
            
            for item in items:
                # Slug is now in the react component
                react_comp = item.css_first("div.react-component")
                if not react_comp:
                    continue
                    
                slug = react_comp.attributes.get("data-item-slug")
                if not slug:
                    continue
                
                rating = None
                liked = False
                
                # Rating and like are in the viewingdata paragraph
                viewing_data = item.css_first("p.poster-viewingdata")
                if viewing_data:
                    rating_span = viewing_data.css_first("span.rating")
                    if rating_span:
                        rating = self._parse_rating(rating_span)
                    
                    liked = viewing_data.css_first("span.like") is not None
                
                films[slug] = FilmInteraction(slug, rating, True, False, liked)
            
            page += 1
            print(f"  Watched page {page-1}: {len(items)} films")
        
        # Get watchlist
        print(f"Scraping {username}'s watchlist...")
        page = 1
        while True:
            tree = self._get(f"{self.BASE}/{username}/watchlist/page/{page}/")
            if not tree:
                break
            
            items = tree.css("li.griditem")
            if not items:
                break
            
            for item in items:
                react_comp = item.css_first("div.react-component")
                if not react_comp:
                    continue
                    
                slug = react_comp.attributes.get("data-item-slug")
                if not slug:
                    continue
                
                if slug in films:
                    films[slug].watchlisted = True
                else:
                    films[slug] = FilmInteraction(slug, None, False, True, False)
            
            page += 1
        
        n_rated = sum(1 for f in films.values() if f.rating)
        n_liked = sum(1 for f in films.values() if f.liked)
        print(f"Total: {len(films)} films ({n_rated} rated, {n_liked} liked)")
        return list(films.values())
    
    def scrape_film(self, slug: str) -> FilmMetadata | None:
        """Scrape metadata for a single film."""
        tree = self._get(f"{self.BASE}/film/{slug}/")
        if not tree:
            return None
        
        # Title
        title_el = tree.css_first("h1.headline-1")
        title = title_el.text(strip=True) if title_el else slug
        
        # Year
        year = None
        year_el = tree.css_first("small.number a, div.releaseyear a")
        if year_el:
            try:
                year = int(year_el.text(strip=True))
            except ValueError:
                pass
        
        # Directors
        directors = []
        for a in tree.css("a[href*='/director/']"):
            name = a.text(strip=True)
            if name and name not in directors:
                directors.append(name)
        
        # Genres
        genres = []
        for a in tree.css("a[href*='/films/genre/']"):
            g = a.text(strip=True)
            if g and g not in genres:
                genres.append(g)
        
        # Cast (top billed)
        cast = []
        for a in tree.css("a[href*='/actor/']")[:10]:
            name = a.text(strip=True)
            if name and name not in cast:
                cast.append(name)
        
        # Themes/tags
        themes = []
        for a in tree.css("a[href*='/films/theme/'], a[href*='/films/mini-theme/']"):
            t = a.text(strip=True)
            if t and t not in themes:
                themes.append(t)
        
        # Runtime
        runtime = None
        runtime_el = tree.css_first("p.text-link.text-footer")
        if runtime_el:
            text = runtime_el.text()
            if "mins" in text:
                try:
                    runtime = int(text.split()[0])
                except (ValueError, IndexError):
                    pass
        
        # Average rating
        avg_rating = None
        meta = tree.css_first("meta[name='twitter:data2']")
        if meta:
            content = meta.attributes.get("content", "")
            try:
                avg_rating = float(content.split()[0])
            except (ValueError, IndexError):
                pass
        
        # Rating count
        rating_count = None
        ratings_el = tree.css_first("a[href*='/ratings/']")
        if ratings_el:
            text = ratings_el.text(strip=True).replace(",", "").replace("K", "000")
            try:
                rating_count = int(float(text))
            except (ValueError, IndexError):
                pass
        
        # Phase 1: Countries
        countries = []
        for a in tree.css("a[href*='/films/country/']"):
            country = a.text(strip=True)
            if country and country not in countries:
                countries.append(country)
        
        # Phase 1: Languages  
        languages = []
        for a in tree.css("a[href*='/films/language/']"):
            lang = a.text(strip=True)
            if lang and lang not in languages:
                languages.append(lang)
        
        # Phase 1: Writers (look for screenplay, writer, story credits)
        writers = []
        for a in tree.css("a[href*='/writer/']"):
            name = a.text(strip=True)
            if name and name not in writers:
                writers.append(name)
        
        # Phase 1: Cinematographers
        cinematographers = []
        for a in tree.css("a[href*='/cinematography/']"):
            name = a.text(strip=True)
            if name and name not in cinematographers:
                cinematographers.append(name)
        
        # Phase 1: Composers
        composers = []
        for a in tree.css("a[href*='/composer/']"):
            name = a.text(strip=True)
            if name and name not in composers:
                composers.append(name)
        
        return FilmMetadata(
            slug=slug, title=title, year=year, directors=directors,
            genres=genres, cast=cast, themes=themes, runtime=runtime,
            avg_rating=avg_rating, rating_count=rating_count,
            countries=countries, languages=languages, writers=writers,
            cinematographers=cinematographers, composers=composers
        )
    
    def _parse_rating(self, span) -> float | None:
        classes = span.attributes.get("class", "")
        for cls in classes.split():
            if cls.startswith("rated-"):
                try:
                    return int(cls.replace("rated-", "")) / 2
                except ValueError:
                    pass
        return None
    
    def scrape_following(self, username: str, limit: int = 100) -> list[str]:
        """Scrape usernames that the target user follows."""
        usernames = []
        page = 1
        
        print(f"Scraping {username}'s following...")
        while len(usernames) < limit:
            tree = self._get(f"{self.BASE}/{username}/following/page/{page}/")
            if not tree:
                break
            
            links = tree.css("a.name")
            if not links:
                break
            
            for link in links:
                href = link.attributes.get("href", "")
                if href.startswith("/") and href.endswith("/"):
                    user = href.strip("/")
                    if user and user not in usernames:
                        usernames.append(user)
                        if len(usernames) >= limit:
                            break
            
            page += 1
        
        print(f"  Found {len(usernames)} following")
        return usernames
    
    def scrape_followers(self, username: str, limit: int = 100) -> list[str]:
        """Scrape usernames that follow the target user."""
        usernames = []
        page = 1
        
        print(f"Scraping {username}'s followers...")
        while len(usernames) < limit:
            tree = self._get(f"{self.BASE}/{username}/followers/page/{page}/")
            if not tree:
                break
            
            links = tree.css("a.name")
            if not links:
                break
            
            for link in links:
                href = link.attributes.get("href", "")
                if href.startswith("/") and href.endswith("/"):
                    user = href.strip("/")
                    if user and user not in usernames:
                        usernames.append(user)
                        if len(usernames) >= limit:
                            break
            
            page += 1
        
        print(f"  Found {len(usernames)} followers")
        return usernames
    
    def scrape_popular_members(self, limit: int = 50) -> list[str]:
        """Scrape popular members from Letterboxd."""
        usernames = []
        page = 1
        
        print("Scraping popular members...")
        while len(usernames) < limit:
            tree = self._get(f"{self.BASE}/members/popular/this/week/page/{page}/")
            if not tree:
                break
            
            links = tree.css("a.name")
            if not links:
                break
            
            for link in links:
                href = link.attributes.get("href", "")
                if href.startswith("/") and href.endswith("/"):
                    user = href.strip("/")
                    if user and user not in usernames:
                        usernames.append(user)
                        if len(usernames) >= limit:
                            break
            
            page += 1
        
        print(f"  Found {len(usernames)} popular members")
        return usernames
    
    def scrape_film_fans(self, slug: str, limit: int = 50) -> list[str]:
        """Scrape users who are fans of a specific film."""
        usernames = []
        page = 1
        
        print(f"Scraping fans of {slug}...")
        while len(usernames) < limit:
            tree = self._get(f"{self.BASE}/film/{slug}/fans/page/{page}/")
            if not tree:
                break
            
            links = tree.css("a.name")
            if not links:
                break
            
            for link in links:
                href = link.attributes.get("href", "")
                if href.startswith("/") and href.endswith("/"):
                    user = href.strip("/")
                    if user and user not in usernames:
                        usernames.append(user)
                        if len(usernames) >= limit:
                            break
            
            page += 1
        
        print(f"  Found {len(usernames)} fans of {slug}")
        return usernames
    
    def scrape_favorites(self, username: str) -> list[str]:
        """Scrape user's favorite films (4-film profile showcase)."""
        tree = self._get(f"{self.BASE}/{username}/")
        if not tree:
            return []
        
        favorites = []
        
        # Favorites are in the profile showcase section
        # Look for poster containers in the favorites section
        showcase = tree.css("section.profile-favorites li.poster-container, section#favourites li.poster-container")
        for item in showcase[:4]:  # Max 4 favorites
            # Try data attribute first
            react_comp = item.css_first("div.react-component")
            if react_comp:
                slug = react_comp.attributes.get("data-film-slug")
                if slug:
                    favorites.append(slug)
                    continue
            
            # Fallback: extract from link
            link = item.css_first("div[data-film-slug]")
            if link:
                slug = link.attributes.get("data-film-slug")
                if slug:
                    favorites.append(slug)
        
        return favorites
    
    def scrape_user_lists(self, username: str, limit: int = 50) -> list[dict]:
        """
        Scrape all lists for a user.
        
        Returns:
            List of dicts with keys: list_slug, list_name, is_ranked
        """
        lists = []
        page = 1
        
        print(f"Scraping {username}'s lists...")
        while len(lists) < limit:
            tree = self._get(f"{self.BASE}/{username}/lists/page/{page}/")
            if not tree:
                break
            
            # Try different selectors for list items
            list_items = tree.css("section.list-summary")
            if not list_items:
                # Fallback selector
                list_items = tree.css("section.film-list-summary")
            
            if not list_items:
                break
            
            for item in list_items:
                # Extract list URL and name
                link = item.css_first("h2 a, h3 a")
                if not link:
                    continue
                
                href = link.attributes.get("href", "")
                list_name = link.text(strip=True)
                
                # Extract slug from /username/list/list-slug/
                parts = href.strip("/").split("/")
                if len(parts) >= 3 and parts[1] == "list":
                    list_slug = parts[2]
                else:
                    continue
                
                # Detect if ranked (look for numbered list indicators)
                is_ranked = (
                    item.css_first(".icon-numbered") is not None or
                    "numbered"in item.attributes.get("class", "").lower()
                )
                
                lists.append({
                    "list_slug": list_slug,
                    "list_name": list_name,
                    "is_ranked": is_ranked
                })
                
                if len(lists) >= limit:
                    break
            
            page += 1
        
        print(f"  Found {len(lists)} lists")
        return lists
    
    def scrape_list_films(self, username: str, list_slug: str) -> list[dict]:
        """
        Scrape films from a specific list.
        
        Returns:
            List of dicts with keys: film_slug, position (if ranked)
        """
        films = []
        page = 1
        
        while True:
            tree = self._get(f"{self.BASE}/{username}/list/{list_slug}/page/{page}/")
            if not tree:
                break
            
            # Check if this is a ranked list by looking for position indicators
            has_positions = tree.css_first(".list-number, .position") is not None
            
            # Get film items
            items = tree.css("li.poster-container")
            if not items:
                break
            
            for idx, item in enumerate(items):
                # Get film slug
                react_comp = item.css_first("div.react-component")
                film_slug = None
                
                if react_comp:
                    film_slug = react_comp.attributes.get("data-film-slug")
                
                if not film_slug:
                    # Fallback
                    link = item.css_first("div[data-film-slug]")
                    if link:
                        film_slug = link.attributes.get("data-film-slug")
                
                if not film_slug:
                    continue
                
                # Determine position
                position = None
                if has_positions:
                    # Look for explicit position number
                    pos_el = item.css_first(".list-number, .position")
                    if pos_el:
                        try:
                            pos_text = pos_el.text(strip=True).rstrip(".")
                            position = int(pos_text)
                        except (ValueError, AttributeError):
                            # Fallback to sequential position
                            position = (page - 1) * len(items) + idx + 1
                    else:
                        # Sequential position as fallback
                        position = (page - 1) * len(items) + idx + 1
                
                films.append({
                    "film_slug": film_slug,
                    "position": position
                })
            
            page += 1
        
        return films
    
    def close(self):
        self.client.close()



class AsyncLetterboxdScraper:
    """Async scraper for parallel metadata fetching."""
    
    BASE = "https://letterboxd.com"
    
    def __init__(self, delay: float = 0.2, max_concurrent: int = 5):
        self.delay = delay
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def scrape_films_batch(self, slugs: list[str]) -> list:
        """Scrape multiple films concurrently."""
        async with httpx.AsyncClient(
            headers={"User-Agent": "Mozilla/5.0 (compatible; letterboxd-rec/1.0)"},
            follow_redirects=True,
            timeout=30.0
        ) as client:
            tasks = [self._scrape_film_async(client, slug) for slug in slugs]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [r for r in results if r is not None and not isinstance(r, Exception)]
    
    async def _scrape_film_async(self, client: httpx.AsyncClient, slug: str):
        """Scrape a single film asynchronously."""
        async with self.semaphore:
            await asyncio.sleep(self.delay)
            try:
                resp = await client.get(f"{self.BASE}/film/{slug}/")
                if resp.status_code == 404:
                    return None
                
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 60))
                    print(f"Rate limited (429) on {slug}, waiting {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    # Retry once
                    resp = await client.get(f"{self.BASE}/film/{slug}/")
                    if resp.status_code == 429:
                        return None
                
                resp.raise_for_status()
                return self._parse_film_page(resp.text, slug)
            except Exception as e:
                return None
    
    def _parse_film_page(self, html: str, slug: str):
        """Parse film page HTML."""
        tree = HTMLParser(html)
        
        title_el = tree.css_first("h1.headline-1")
        title = title_el.text(strip=True) if title_el else slug
        
        year = None
        year_el = tree.css_first("small.number a, div.releaseyear a")
        if year_el:
            try:
                year = int(year_el.text(strip=True))
            except ValueError:
                pass
        
        directors = list(dict.fromkeys([a.text(strip=True) for a in tree.css("a[href*='/director/']") if a.text(strip=True)]))
        genres = list(dict.fromkeys([a.text(strip=True) for a in tree.css("a[href*='/films/genre/']") if a.text(strip=True)]))
        cast = list(dict.fromkeys([a.text(strip=True) for a in tree.css("a[href*='/actor/']")[:10] if a.text(strip=True)]))
        themes = list(dict.fromkeys([a.text(strip=True) for a in tree.css("a[href*='/films/theme/'], a[href*='/films/mini-theme/']") if a.text(strip=True)]))
        
        runtime = None
        runtime_el = tree.css_first("p.text-link.text-footer")
        if runtime_el and "mins" in runtime_el.text():
            try:
                runtime = int(runtime_el.text().split()[0])
            except (ValueError, IndexError):
                pass
        
        avg_rating = None
        meta = tree.css_first("meta[name='twitter:data2']")
        if meta:
            try:
                avg_rating = float(meta.attributes.get("content", "").split()[0])
            except (ValueError, IndexError):
                pass
        
        rating_count = None
        ratings_el = tree.css_first("a[href*='/ratings/']")
        if ratings_el:
            try:
                rating_count = int(float(ratings_el.text(strip=True).replace(",", "").replace("K", "000")))
            except (ValueError, IndexError):
                pass
        
        # Phase 1: New fields
        countries = list(dict.fromkeys([a.text(strip=True) for a in tree.css("a[href*='/films/country/']") if a.text(strip=True)]))
        languages = list(dict.fromkeys([a.text(strip=True) for a in tree.css("a[href*='/films/language/']") if a.text(strip=True)]))
        writers = list(dict.fromkeys([a.text(strip=True) for a in tree.css("a[href*='/writer/']") if a.text(strip=True)]))
        cinematographers = list(dict.fromkeys([a.text(strip=True) for a in tree.css("a[href*='/cinematography/']") if a.text(strip=True)]))
        composers = list(dict.fromkeys([a.text(strip=True) for a in tree.css("a[href*='/composer/']") if a.text(strip=True)]))
        
        return FilmMetadata(
            slug=slug, title=title, year=year, directors=directors,
            genres=genres, cast=cast, themes=themes,
            runtime=runtime, avg_rating=avg_rating, rating_count=rating_count,
            countries=countries, languages=languages, writers=writers,
            cinematographers=cinematographers, composers=composers
        )
