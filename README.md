# Letterboxd Recommender

A command-line film recommendation engine that scrapes your Letterboxd profile and generates personalized recommendations using metadata matching and collaborative filtering.

## Features

- **Scrape** your Letterboxd watched films, ratings, likes, and watchlist
- **Discover** similar users from followers, following, popular members, or fans of specific films
- **Three recommendation strategies**: metadata-based, collaborative filtering, and hybrid
- **Profile analysis** showing your genre, director, actor, and decade preferences
- **Film similarity search** to find films like ones you already love
- **Diversity mode** to avoid over-concentration on single directors
- **Export/import** your database as JSON for backup or sharing

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

# 2. Get recommendations based on your taste
python main.py recommend your_username

# 3. View your preference profile
python main.py profile your_username
```

## Commands

### `scrape` — Fetch user data from Letterboxd

```bash
python main.py scrape USERNAME [--refresh DAYS]
```

- `--refresh DAYS`: Only re-scrape if data is older than N days

### `discover` — Find and scrape other users

```bash
python main.py discover SOURCE [--username USER] [--film-slug SLUG] [--limit N]
```

Sources:
- `following`: Users that USERNAME follows
- `followers`: Users following USERNAME  
- `popular`: This week's popular members
- `film`: Fans of a specific film (requires `--film-slug`)

Examples:
```bash
# Scrape 20 users who follow you
python main.py discover followers --username your_username --limit 20

# Scrape 30 fans of Perfect Blue
python main.py discover film --film-slug perfect-blue --limit 30
```

### `recommend` — Generate recommendations

```bash
python main.py recommend USERNAME [options]
```

Options:
- `--strategy {metadata,collaborative,hybrid}`: Algorithm to use (default: metadata)
- `--limit N`: Number of recommendations (default: 20)
- `--min-year YEAR`: Filter by minimum release year
- `--max-year YEAR`: Filter by maximum release year
- `--genres GENRE [GENRE ...]`: Only include these genres
- `--exclude-genres GENRE [GENRE ...]`: Exclude these genres
- `--min-rating N`: Minimum Letterboxd community rating
- `--diversity`: Limit films per director (avoids 5 Kurosawa films in a row)
- `--max-per-director N`: Max films per director in diversity mode (default: 2)
- `--format {text,json,markdown}`: Output format

Examples:
```bash
# Classic anime recommendations from the 80s-90s
python main.py recommend your_username --genres Animation --min-year 1980 --max-year 1999

# Highly-rated horror, diverse directors, as JSON
python main.py recommend your_username --genres Horror --min-rating 3.5 --diversity --format json

# Collaborative recommendations (needs other users scraped)
python main.py recommend your_username --strategy collaborative
```

### `profile` — View your taste profile

```bash
python main.py profile USERNAME
```

Shows your top genres, directors, actors, and decade preferences based on your ratings and likes.

### `similar` — Find films similar to a specific film

```bash
python main.py similar FILM_SLUG [--limit N]
```

Example:
```bash
python main.py similar perfect-blue --limit 15
```

### `stats` — Database statistics

```bash
python main.py stats [--verbose]
```

- `--verbose`: Show missing metadata, stalest users, genre distribution

### `export` / `import` — Backup your data

```bash
python main.py export backup.json
python main.py import backup.json
```

## Recommendation Strategies

### Metadata (default)
Builds a preference profile from your ratings/likes and scores unseen films by matching genres, directors, actors, themes, and decades. Works well with any amount of data.

### Collaborative
Finds users with similar taste (based on rating correlation) and recommends films they loved. **Requires**:
- You have rated films (not just watched/liked)
- Other users are scraped (`discover` command)
- Sufficient rating overlap with other users

### Hybrid
Combines both strategies using Borda count rank merging. Best of both worlds when you have enough data.

## How Scoring Works

Films you've interacted with are weighted:
- ★★★★★ (5.0): +2.0 (loved)
- ★★★★ (4.0): +1.0 (liked)
- ★★★ (3.0): +0.3 (neutral)
- ★★ (2.0): -0.5 (disliked)
- ★ (1.0): -1.5 (hated)
- ❤️ (liked, no rating): +1.5
- Watched only: +0.4
- Watchlisted: +0.2

These weights accumulate across genres, directors, etc. to build your profile.

## Tips

1. **Rate more films** on Letterboxd for better collaborative filtering
2. **Scrape fans of films you love** (`discover film --film-slug`) for higher-quality neighbors
3. **Use diversity mode** if recommendations cluster too heavily on one director
4. **Export regularly** to backup your scraped data

## Data Storage

All data is stored in `data/letterboxd.db` (SQLite). Delete this file to start fresh.

## Requirements

- Python 3.10+
- httpx
- selectolax
- tqdm

## License

MIT