import httpx
import json
import time
from selectolax.parser import HTMLParser
from dataclasses import dataclass
from tqdm import tqdm

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
        
        return FilmMetadata(
            slug=slug, title=title, year=year, directors=directors,
            genres=genres, cast=cast, themes=themes, runtime=runtime,
            avg_rating=avg_rating, rating_count=rating_count
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
    
    def close(self):
        self.client.close()
