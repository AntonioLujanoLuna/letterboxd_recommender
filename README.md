# Letterboxd Recommender

A sophisticated command-line film recommendation engine that scrapes your Letterboxd profile and generates personalized recommendations using metadata matching, collaborative filtering, and advanced preference modeling.

## Features

### Core Capabilities
- **Scrape** your Letterboxd watched films, ratings, likes, watchlists, and curated lists
- **Discover** similar users from followers, following, popular members, or fans of specific films
- **Three recommendation strategies**: metadata-based, collaborative filtering, and hybrid
- **Advanced scoring** with IDF weighting, temporal decay, confidence adjustments, and genre pair modeling
- **Profile analysis** showing your genre, director, actor, decade, and crew preferences
- **Film similarity search** to find films like ones you already love
- **Watchlist triage** to rank your watchlist by predicted enjoyment
- **Filmography gaps** to discover essential unseen films from favorite directors
- **Diversity mode** to avoid over-concentration on single directors
- **Export/import** your database as JSON for backup or sharing

### Advanced Features
- **Temporal decay**: Recent interactions weighted more heavily (configurable half-life)
- **IDF (Inverse Document Frequency)**: Rare preferences weighted more than common ones
- **List integration**: Favorites and ranked lists boost preference signals
- **Confidence weighting**: Scores adjusted based on sample size
- **Negative matching**: Surface warnings for disliked attributes
- **Activity pre-filtering**: Validate users before scraping (film count, rating activity)
- **Incremental scraping**: Fast updates by stopping at already-scraped films
- **Pending queue**: Efficient user discovery with priority-based scraping
- **Profile caching**: Fast repeated recommendations with automatic cache invalidation
- **Async scraping**: Parallel metadata fetching with coordinated rate limiting

## Installation

```bash
git clone https://github.com/AntonioLujanoLuna/letterboxd-recommender.git
cd letterboxd-recommender
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Scrape your Letterboxd profile
python main.py scrape your_username

# 2. Compute IDF scores for better recommendations (first time only)
python main.py rebuild-idf

# 3. Get personalized recommendations
python main.py recommend your_username

# 4. View your preference profile
python main.py profile your_username
```

## Commands

### `scrape` — Fetch user data from Letterboxd

```bash
python main.py scrape USERNAME [options]
```

**Options:**
- `--refresh DAYS`: Only re-scrape if data is older than N days
- `--include-lists`: Include user lists (favorites, ranked lists) — **enabled by default**
- `--no-include-lists`: Skip scraping user lists
- `--max-lists N`: Maximum number of lists to scrape per user (default: 50)
- `--incremental`: Stop pagination when hitting already-scraped films (faster for updates)

**Examples:**
```bash
# Full scrape with lists
python main.py scrape your_username

# Only scrape if older than 7 days
python main.py scrape your_username --refresh 7

# Quick incremental update (faster)
python main.py scrape your_username --incremental

# Skip lists for minimal scrape
python main.py scrape your_username --no-include-lists
```

### `discover` — Find and scrape other users

```bash
python main.py discover SOURCE [options]
```

**Sources:**
- `following`: Users that USERNAME follows
- `followers`: Users following USERNAME  
- `popular`: This week's popular members
- `film`: Fans of a specific film (requires `--film-slug`)
- `film_reviews`: Reviewers of a specific film (highest quality, requires `--film-slug`)

**Options:**
- `--username USER`: Username (required for `following`/`followers`)
- `--film-slug SLUG`: Film slug (required for `film`/`film_reviews`, e.g., 'perfect-blue')
- `--limit N`: Number of users to scrape (default: 50)
- `--min-films N`: Minimum film count for activity pre-filtering (default: 50)
- `--dry-run`: Show which users would be scraped without actually scraping
- `--continue`: Continue scraping from pending user queue without discovering new users
- `--source-refresh-days N`: Days before re-crawling a source from page 1 (default: 7)

**Examples:**
```bash
# Discover and scrape 20 users who follow you
python main.py discover followers --username your_username --limit 20

# Discover fans of Perfect Blue (highest engagement)
python main.py discover film_reviews --film-slug perfect-blue --limit 30

# Discover popular members who are active (100+ films)
python main.py discover popular --limit 50 --min-films 100

# Dry run to see who would be scraped
python main.py discover following --username your_username --limit 10 --dry-run

# Continue scraping from pending queue
python main.py discover --continue --limit 20
```

**Discovery Queue System:**

The `discover` command uses a two-stage process:
1. **Discovery**: Find users from a source and add them to a pending queue
2. **Scraping**: Process users from the queue with priority ordering

The pending queue allows you to:
- Discover from multiple sources before scraping
- Resume interrupted scraping sessions
- Process users by priority (reviewers > followers > popular > film fans)

### `recommend` — Generate recommendations

```bash
python main.py recommend USERNAME [options]
```

**Options:**
- `--strategy {metadata,collaborative,hybrid}`: Algorithm to use (default: metadata)
- `--limit N`: Number of recommendations (default: 20)
- `--min-year YEAR`: Filter by minimum release year
- `--max-year YEAR`: Filter by maximum release year
- `--genres GENRE [GENRE ...]`: Only include these genres
- `--exclude-genres GENRE [GENRE ...]`: Exclude these genres
- `--min-rating N`: Minimum Letterboxd community rating
- `--diversity`: Limit films per director (avoids 5 Kurosawa films in a row)
- `--max-per-director N`: Max films per director in diversity mode (default: 2)
- `--no-temporal-decay`: Disable temporal decay (treat old and new ratings equally)
- `--format {text,json,markdown,csv}`: Output format

**Examples:**
```bash
# Classic anime recommendations from the 80s-90s
python main.py recommend your_username --genres Animation --min-year 1980 --max-year 1999

# Highly-rated horror, diverse directors, as JSON
python main.py recommend your_username --genres Horror --min-rating 3.5 --diversity --format json

# Collaborative recommendations (needs other users scraped)
python main.py recommend your_username --strategy collaborative

# Hybrid approach (best of both worlds)
python main.py recommend your_username --strategy hybrid --limit 30

# Without temporal decay (treat all ratings equally)
python main.py recommend your_username --no-temporal-decay
```

### `profile` — View your taste profile

```bash
python main.py profile USERNAME
```

Shows your top genres, directors, actors, decades, and other attributes based on your ratings and likes. Includes observation counts for each preference.

### `similar` — Find films similar to a specific film

```bash
python main.py similar FILM_SLUG [--limit N]
```

**Example:**
```bash
python main.py similar perfect-blue --limit 15
```

### `triage` — Rank your watchlist by predicted enjoyment

```bash
python main.py triage USERNAME [--limit N]
```

Scores every film in your watchlist and ranks them by how much you're likely to enjoy them. Perfect for deciding what to watch next.

**Example:**
```bash
python main.py triage your_username --limit 30
```

### `gaps` — Find essential missing films from favorite directors

```bash
python main.py gaps USERNAME [options]
```

Discovers highly-rated films from directors you love that you haven't seen yet.

**Options:**
- `--min-score N`: Minimum director affinity score (default: 2.0)
- `--limit N`: Max films per director (default: 3)
- `--min-year YEAR`: Minimum release year
- `--max-year YEAR`: Maximum release year

**Example:**
```bash
python main.py gaps your_username --min-score 3.0 --limit 5
```

### `rebuild-idf` — Rebuild IDF scores for attribute rarity weighting

```bash
python main.py rebuild-idf
```

Computes Inverse Document Frequency scores for all film attributes (genres, directors, actors, themes, countries, languages). This helps the recommender identify and prioritize your distinctive preferences (e.g., loving Iranian cinema vs. loving drama).

Run this:
- After your first scrape
- Periodically as your database grows
- After adding many new users

### `stats` — Database statistics

```bash
python main.py stats [--verbose]
```

**Options:**
- `--verbose`: Show missing metadata, stalest users, genre distribution

### `export` / `import` — Backup your data

```bash
# Export to JSON
python main.py export backup.json

# Import from JSON
python main.py import backup.json
```

## Recommendation Strategies

### Metadata (default)
Builds a preference profile from your ratings/likes and scores unseen films by matching:
- Genres (with pair modeling for combinations like "horror+comedy")
- Directors, writers, cinematographers, composers
- Cast members
- Countries and languages
- Themes and decades
- Community ratings similar to your average

**Strengths:**
- Works immediately with just your data
- Transparent reasoning (shows why each film is recommended)
- Handles cold start well
- Configurable with filters

**Advanced features:**
- **IDF weighting**: Rare preferences (e.g., Turkish cinema) weighted more than common ones (e.g., drama)
- **Temporal decay**: Recent ratings weighted more than old ones (taste evolution)
- **Confidence weighting**: Preferences with more observations weighted higher
- **List integration**: Films in your favorites/ranked lists boost related preferences
- **Negative matching**: Warns about disliked attributes

### Collaborative
Finds users with similar taste (based on rating correlation) and recommends films they loved.

**Strengths:**
- Discovers unexpected films outside your usual patterns
- Benefits from wisdom of the crowd
- Gets better as you add more users

**Requirements:**
- You have rated films (not just watched/liked)
- Other users are scraped (`discover` command)
- Sufficient rating overlap with other users (5+ common films)

**Technical details:**
- Uses sparse matrices for efficient similarity computation
- Adjusted cosine similarity (mean-centered ratings)
- Confidence weighting based on overlap size
- Fast precomputed similarity components

### Hybrid
Combines both strategies using normalized score merging. Provides balanced recommendations that leverage both your explicit preferences and community wisdom.

**Best when:**
- You have 50+ other users scraped
- You want diverse recommendations
- You have both strong preferences and openness to discovery

## How Scoring Works

### Interaction Weights

Films you've interacted with are weighted:
- ★★★★★ (5.0): +2.0 (loved)
- ★★★★½ (4.5): +2.0 (loved)
- ★★★★ (4.0): +1.0 (liked)
- ★★★½ (3.5): +1.0 (liked)
- ★★★ (3.0): +0.3 (neutral-positive)
- ★★½ (2.5): -0.5 (disliked)
- ★★ (2.0): -0.5 (disliked)
- ★½ (1.5): -1.5 (hated)
- ★ (1.0): -1.5 (hated)
- ½ (0.5): -1.5 (hated)
- ❤️ (liked, no rating): +1.5
- Watched only: +0.4
- Watchlisted: +0.2

### List Multipliers

Films in your lists get preference boosts:
- **Favorites** (profile showcase): 3.0× (strongest signal)
- **Ranked lists, position 1-10**: 2.0×
- **Ranked lists, position 11-30**: 1.5×
- **Ranked lists, position 31+**: 1.2×
- **Curated lists** (non-ranked): 1.3×

These weights accumulate across genres, directors, etc. to build your profile.

### Temporal Decay

By default, recent interactions are weighted more heavily to account for evolving taste:
- Half-life: 2 years (weight halves every 2 years)
- Minimum weight: 0.1 (old favorites don't vanish completely)
- Applied to both profile building and average rating computation

Disable with `--no-temporal-decay` flag if you want equal weighting across time.

## Configuration

### Environment Variables

Advanced users can override defaults via environment variables:

```bash
# Database location
export LETTERBOXD_DB="data/letterboxd.db"

# Scraper delays (in seconds)
export LETTERBOXD_SCRAPER_DELAY=1.0
export LETTERBOXD_ASYNC_DELAY=0.2
export LETTERBOXD_MAX_CONCURRENT=5

# Temporal decay settings
export LETTERBOXD_TEMPORAL_DECAY_ENABLED=true
export LETTERBOXD_TEMPORAL_DECAY_HALF_LIFE_DAYS=730  # 2 years
```

### Advanced Configuration

For deeper customization, edit `src/letterboxd_rec/config.py`:
- Attribute weights (genres, directors, etc.)
- Match thresholds for explanations
- Confidence parameters
- Normalization exponents
- IDF settings

## Tips

1. **Rate more films** on Letterboxd for better collaborative filtering
2. **Scrape fans of films you love** (`discover film --film-slug`) for higher-quality neighbors
3. **Use diversity mode** if recommendations cluster too heavily on one director
4. **Rebuild IDF** periodically as your database grows
5. **Export regularly** to backup your scraped data
6. **Try hybrid strategy** once you have 50+ other users for balanced recommendations
7. **Use triage** to prioritize your ever-growing watchlist
8. **Check gaps** to complete filmographies of directors you love

## Performance Notes

- **Profile caching**: Profiles are automatically cached for 7 days and invalidated when data changes
- **SQL-side filtering**: Year and rating filters applied at database level for efficiency
- **Connection pooling**: Thread-safe connection management with automatic cleanup
- **Sparse matrices**: Collaborative filtering uses scipy sparse matrices for large datasets
- **Async scraping**: Metadata fetching parallelized with coordinated rate limiting
- **Incremental scraping**: `--incremental` flag stops pagination early for faster updates

## Data Storage

All data is stored in `data/letterboxd.db` (SQLite). Database includes:
- **films**: Film metadata (title, year, genres, cast, crew, ratings)
- **user_films**: User interactions (ratings, watches, likes, watchlist)
- **user_lists**: User list entries (favorites, ranked lists)
- **user_profiles**: Cached preference profiles
- **discovery_sources**: Discovery source pagination state
- **pending_users**: Pending user scraping queue
- **attribute_idf**: IDF scores for rarity weighting
- **Normalized tables**: Fast lookup tables for genres, directors, cast, themes

Delete `data/letterboxd.db` to start fresh.

## Requirements

- Python 3.10+
- httpx (HTTP client)
- selectolax (HTML parsing)
- scipy (sparse matrices for collaborative filtering)
- tqdm (progress bars)

## Troubleshooting

### "Only found N neighbors (min: M)"
You need more users scraped for collaborative filtering. Run `discover` commands to add users.

### "Missing metadata for X films"
Some films in user histories lack metadata. Re-run `python main.py scrape username` to fetch missing data.

### "Profile cache invalidated"
Normal behavior when your films/lists are updated. Profile will be rebuilt automatically.

### Rate limiting (429 errors)
The scraper handles this automatically with exponential backoff. If persistent, increase `LETTERBOXD_SCRAPER_DELAY`.

### Connection pool exhausted
If running many concurrent operations, you may hit the connection pool limit (50). This is rare in normal CLI usage but can occur with heavy automation. The pool automatically cleans up dead thread connections.

## License

MIT

## Contributing

Issues and pull requests welcome! Areas for contribution:
- Additional recommendation strategies
- More sophisticated collaborative filtering (matrix factorization, neural CF)
- Web interface
- Integration with other film databases (TMDb, IMDb)
- Enhanced list support (comments, tags)
- Social features (compare profiles, shared favorites)

## Acknowledgments

Built with love for film and data. Letterboxd is a trademark of Letterboxd Limited. This project is not affiliated with or endorsed by Letterboxd.