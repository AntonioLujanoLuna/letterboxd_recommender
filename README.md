# Letterboxd Recommender

A sophisticated command-line film recommendation engine that scrapes your Letterboxd profile and generates personalized recommendations using metadata matching, collaborative filtering, and advanced preference modeling.

## Features

### Core Capabilities
- **Scrape** your Letterboxd watched films, ratings, likes, watchlists, and curated lists
- **Discover** similar users from followers, following, popular members, or fans of specific films
- **Three recommendation strategies**: metadata-based, collaborative filtering, and EASE (with a MovieLens-trained blend)
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

### `--backend letterboxdpy` — Scrape through letterboxdpy

If plain scraping gets 403s, pass the global `--backend letterboxdpy` flag (or set
`LETTERBOXD_BACKEND=letterboxdpy`). Only the HTTP layer changes: requests go through
[letterboxdpy](https://github.com/nmcassa/letterboxdpy)'s browser-impersonating fetcher, and all
parsing, discovery, async batching and the `scrape-daemon` / `crawl` commands work as usual.

```bash
python main.py --backend letterboxdpy scrape YOUR_USERNAME
```

### `crawl` — Unattended large-scale scraping

Seeds the pending queue, drains it with the async daemon, refills it, and repeats until `--users`
new users are scraped or no candidates remain. If the discovery sources run dry it falls back to
unscraped users from the social graph (most-followed first).

```bash
python main.py --backend letterboxdpy crawl   --seed-film perfect-blue --seed-film parasite --popular   --users 2000 --min-films 100   --max-concurrent-users 2 --max-concurrent-requests 4 --async-delay 0.5
```

Useful options: `--seed-user USER` (their followers), `--sources-file`, `--refill-below N`,
`--queue-target N`, `--chunk N` (users per drain), `--no-social`, `--no-expand-graph`,
`--max-stalls N` (abort after N consecutive chunks that scraped nobody). It resumes safely: the
queue lives in the database.

**Resilience.** Queued users are *claimed*, not deleted, when a worker picks them up; they leave
the queue only after a successful scrape. If the process is killed, claims older than
`--claim-ttl-minutes` (default 30; `0` = all) are put back on the next start. A user whose scrape
fails is re-queued and retried on later runs; after 3 failures they are *parked* (kept in the table,
never claimed again; see the "Parked" line in `queue-status`, drop them with `queue-clear`). Each
user gets one attempt per run. With the letterboxdpy backend, repeated 403 "blocked" responses
trip a circuit breaker that pauses all workers (5 minutes, doubling up to 30) instead of
hammering a blocked IP; private profiles don't count. Film metadata for a user is always fetched in
full, in chunks of `--batch`.

**Speed and integrity.** The async scrapers (daemon, `crawl`, `discover --parallel-users`) start
requests at a steady rate capped by `--max-rps` (default `$LETTERBOXD_MAX_RPS` or 6). Throughput is
that cap, not `concurrency / latency`, so raise `--max-concurrent-requests` (about `max-rps x
latency`, e.g. 6-8) before raising the cap. A 429 still slows everything via the adaptive delay. After
page 1, a user's listing pages are fetched 4 at a time. A listing page that can't be fetched is
retried and then fails the user (queued again for later) instead of silently keeping a truncated
profile, because Letterboxd answers pages past the end with an empty grid, never a 404. The daemon
logs a throughput line about once a minute (requests/s, p50/p95 latency, 403/429/error counts) and a
summary at the end; use it to tune `--max-rps`. Concurrent users never fetch the same film twice, and
films whose fetch failed are retried on later runs.

**Going faster.** `--auto-rps CEILING` (daemon, `crawl`, `discover --parallel-users`) starts at
`--max-rps` and probes upward by 1 req/s after every 20 clean seconds, halving at once on a 429 or a
blocking 403, so it settles near what Letterboxd tolerates instead of you guessing:

```bash
python main.py --backend letterboxdpy crawl --seed-film parasite --users 2000   --max-rps 6 --auto-rps 12 --max-concurrent-requests 10
```

Keep `--max-concurrent-requests` above `rps x ~0.5s` (latency), e.g. 10 for a 12 req/s ceiling, or
concurrency becomes the limit. Watch the once-a-minute throughput line for `403`/`429` counts. In a
75-second test it climbed from 6 to 12 req/s with no throttling, but sustained runs weren't
measured, so treat the ceiling as something to verify on your own connection. Discovery's activity
pre-filter (one profile request per candidate) now runs 8 at a time, and users are claimed only up
to `--users`, so no in-flight user is cancelled at the end of a chunk. The letterboxdpy transport
retries a blocking 403 twice with short backoff and returns private-profile 403s immediately.

### Authenticated scraping (`LETTERBOXD_COOKIE`)

Optional. Put your logged-in session in `.env` (git-ignored) as one line:

```
LETTERBOXD_COOKIE="name1=value1; name2=value2; ..."
```

To get it: sign in at letterboxd.com in Chrome, open DevTools (F12) → **Network**, reload, click the
first `letterboxd.com` document request → **Headers** → **Request Headers** → copy the entire
`Cookie:` value. Use this rather than `document.cookie` or the Application tab, which hide
HttpOnly cookies, including the session cookie. Treat it like a password and re-copy it if it
expires. Set `LETTERBOXD_ANONYMOUS=1` to ignore the cookie for a run (no login is sent). Lines that python-dotenv can't parse are still read by a fallback loader.

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
- `--strategy {metadata,collaborative,svd,graph,ease,content,movielens,blend}`: Algorithm to use (default: `blend` when the MovieLens model is built and the database holds 20+ users, `ease` without the model, `metadata` with too few users)
- `--limit N`: Number of recommendations (default: 20)
- `--decade 1990s`: A decade (shortcut for `--min-year 1990 --max-year 1999`; `1990` works too)
- `--min-year YEAR`: Filter by minimum release year
- `--max-year YEAR`: Filter by maximum release year
- `--genres GENRE [GENRE ...]`: Only include these genres
- `--exclude-genres GENRE [GENRE ...]`: Exclude these genres
- `--min-rating N`: Minimum Letterboxd community rating
- `--min-runtime MIN` / `--max-runtime MIN`: Runtime range in minutes
- `--exclude-shorts` / `--exclude-animation`: Leave out short or animated films
- `--countries C [C ...]` / `--languages L [L ...]`: Only films from these countries / in these languages
- `--actors NAME [NAME ...]` / `--directors NAME [NAME ...]`: Only films starring any of these actors (top-billed
  cast, about ten names per film) or directed by any of these directors; names are case-insensitive

The year, genre, rating, runtime, short/animation, country and language filters all apply to the
`ease`, `movielens`, `blend` and `svd` strategies (the other strategies ignore the newer ones and say so).
A film missing a field is never excluded by a filter on that field, except `--actors` and `--directors`,
which mean "must star / must be directed by" and so drop films whose cast or director is unknown.

For `ease`, `blend` and `movielens` the filters restrict the ranking *before* the top results are
taken, so a narrow combination (one actor, or 1950s Japan without animation) still returns a full list
whenever enough matching films exist. If fewer than `--limit` come back, the command says so.

```bash
python main.py recommend you --decade 1990s --countries Japan
python main.py recommend you --actors "Tom Hanks" --limit 10
python main.py recommend you --directors "Christopher Nolan" "Denis Villeneuve"
```

**Short histories:** with fewer than about 50 films in a user's history, `ease` and `blend` mix
popular films into the list (by rank; 75% popularity up to 20 films, fading linearly to 0 at 50),
because below ~20 films plain popularity beat every personalized model offline (recall@20 about
0.10 vs 0.06-0.10), and the mix beat both at every size from 3 to 40 films. The command says so when
it applies; scrape more of the user's activity and it fades out on its own.
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

# EASE item-item model (strongest in offline evaluation; needs other users scraped)
python main.py recommend your_username --strategy ease

# Local EASE + MovieLens EASE, fused by reciprocal rank (needs `python main.py movielens-build`).
# Offline (418 users, paired bootstrap) it beat local EASE alone by ~1.4 points of NDCG@20 (95% CI
# +0.3 to +2.6) and ~0.9 of recall@20 (not distinguishable from zero), and beat MovieLens alone on
# both. About 3% of its picks are films outside the 1000 most-watched, vs 0.2% for local EASE.
# Weight on the local model: LETTERBOXD_BLEND_EASE_WEIGHT (default 0.2).
python main.py recommend your_username --strategy blend

# Less mainstream picks: divides scores by popularity**N (0 = off); applies to ease, movielens and
# blend. Default blend, ~500 held-out users, recall@20 / picks outside the 1000 most-watched:
#   0 -> 0.431 / 4.9%   0.2 -> 0.424 / 11.3%   0.3 -> 0.389 / 17.2%   0.5 -> 0.269 / 34%
# so 0.2 more than doubles niche picks for under a point of recall. Opt-in: it trades accuracy.
python main.py recommend your_username --novelty 0.2

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

### `evaluate` — Measure recommendation quality

```bash
python main.py evaluate [--strategies popularity svd collaborative metadata ease]
                        [--k 20] [--holdout 0.2] [--min-positives 10] [--max-users 200]
                        [--users NAME ...] [--seed 42] [--json [--per-user]]
```

Hides a random 20% of each sampled user's positive films (rated ≥3.5★ or liked), fits each
strategy on what's left, and reports how many hidden films land in the top k:
recall@k, NDCG@k, hit-rate@k and catalog coverage. A **popularity** baseline (most-liked films
the user hasn't seen) is always included; a strategy that doesn't beat it isn't earning its
complexity. Run it before and after changing weights in `config.py` and keep only changes that
improve NDCG. Use the same `--seed` when comparing runs.

The `graph` strategy isn't evaluated yet: it reads the live database, so hidden films would leak in.

### `movielens-build` — Train on MovieLens (lots of real users, no scraping)

```bash
python main.py movielens-build --download   # first time: fetches ML-32M (~240 MB)
python main.py movielens-build --lambdas 500 1000 2000 --max-items 12000
python main.py recommend YOUR_USERNAME --strategy movielens
```

Fits EASE on [MovieLens 32M](https://grouplens.org/datasets/movielens/) (32M ratings from 200k
people), tries each `--lambdas` value on 2,000 held-out MovieLens users, prints recall/NDCG@20
against a popularity baseline, and saves the best model to `data/movielens_ease.npz`.
Your Letterboxd films are matched to MovieLens by TMDB/IMDb id (captured on every film
scrape), falling back to title + year.

**Limits:** MovieLens ratings stop in October 2023, so newer films can't be recommended by
this strategy or used as input (`content` covers them). `--max-items` films are modelled; memory
grows with its square (10k ≈ 3 GB peak). GroupLens terms: personal/research use only, no
redistribution.

### `stats` — Database statistics

```bash
python main.py stats [--verbose]
```

**Options:**
- `--verbose`: Show missing metadata, stalest users, genre distribution

### `backup` — Safe database snapshots

```bash
python main.py backup [--dir data/backups] [--keep 5]
```

Uses SQLite's online backup, so it is consistent even while a crawl is writing, checks the copy
before keeping it, and prunes to the newest `--keep` files. The database is a few hundred MB, so
`--keep 5` uses a couple of GB.

### `supervise` — Keep a crawl alive and backed up

```bash
python main.py supervise [--backup-every 60] [--keep 5] [--max-failures 20] --     --backend letterboxdpy crawl --users 10000 --max-rps 2 --auto-rps 5
```

Runs the command after `--`, restarts it with a growing pause if it crashes (giving up after
`--max-failures` crashes in a row; a run that lasted 10+ minutes resets the count), and backs the
database up every `--backup-every` minutes. A lock file stops a second supervisor from starting, so
two crawlers can't share one IP. A clean exit ends the supervisor. It does not wake a sleeping
laptop; disable sleep for long crawls.

### `doctor` — Check the database

```bash
python main.py doctor
```

Read-only report: integrity check, watched films missing metadata or an id, orphaned rows, parked or
already-scraped queue entries, and users without social edges, each with the command that fixes it.
Some "missing metadata" is normal while a crawl is running, since films are filled in after users.

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

### EASE
A closed-form item-item model (Steck, 2019). It learns, from everyone's watch/rating/like
history, how much each film "predicts" each other film. Fitting is one matrix inverse
(seconds), and the model is cached until the data changes. Explanations name the films you
liked that contributed most.

**Requirements:** other users scraped. Only films with 5+ interactions (`EASE_MIN_INTERACTIONS`)
and at most 8k films (`EASE_MAX_ITEMS`) are modelled. Tune regularization with
`LETTERBOXD_EASE_LAMBDA` and compare runs with `evaluate`.

### Content
A fast, vectorized alternative to the metadata strategy: films become IDF-weighted feature
vectors (directors, genres, themes, cast, crew, countries, decade), your taste is the
rating-weighted sum of the films you've seen, and every film is scored in one sparse
matrix product. Works with only your own data.

### SVD
Truncated SVD on star ratings with damped film biases (`LETTERBOXD_SVD_ITEM_BIAS_REG`).
Predicting star ratings is a weaker objective for picking a top-N list than EASE's;
check `evaluate` before relying on it.

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

# Scraping backend (native | letterboxdpy) and optional logged-in session
# Request-start rate cap for the async scrapers (requests/second)
# LETTERBOXD_MAX_RPS=6

# Block circuit breaker (letterboxdpy backend): threshold, cooldown and max cooldown (seconds)
# LETTERBOXD_BLOCK_THRESHOLD=5  LETTERBOXD_BLOCK_COOLDOWN=300  LETTERBOXD_BLOCK_MAX_COOLDOWN=1800
export LETTERBOXD_BACKEND="letterboxdpy"
# LETTERBOXD_COOKIE is best kept in .env (see Authenticated scraping)

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
6. **Use the default (`blend`/`ease`)** once you have 50+ other users
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