import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta

from .database import init_db, get_db, load_json, load_user_lists
from .scraper import LetterboxdScraper
from .recommender import MetadataRecommender, CollaborativeRecommender, Recommendation
from .profile import build_profile
from tqdm import tqdm
import asyncio

logger = logging.getLogger(__name__)


def _validate_slug(slug: str) -> str:
    """
    Validate a film slug to prevent injection or invalid characters.
    Raises ValueError if slug contains invalid characters.
    Returns lowercased valid slug.
    """
    import re
    if not re.match(r'^[a-z0-9-]+$', slug.lower()):
        raise ValueError(f"Invalid slug: {slug}")
    return slug.lower()


def _validate_username(username: str) -> str:
    """
    Sanitize a Letterboxd username.
    Returns lowercased alphanumeric + underscores/hyphens only.
    """
    import re
    sanitized = re.sub(r'[^a-z0-9_-]', '', username.lower())
    if sanitized != username.lower():
        logger.warning(f"Username '{username}' sanitized to '{sanitized}'")
    return sanitized


def _load_all_user_films(conn, username_filter: str | None = None) -> dict[str, list[dict]]:
    """
    Load all user films in a single query with optional username filter.

    Args:
        conn: Database connection
        username_filter: If provided, only load films for this user (useful for single-user operations)

    Returns:
        Dict mapping username -> list of film interaction dicts.
    """
    all_user_films = defaultdict(list)

    if username_filter:
        # Optimized single-user query
        rows = conn.execute("""
            SELECT username, film_slug as slug, rating, watched, watchlisted, liked
            FROM user_films
            WHERE username = ?
        """, (username_filter,)).fetchall()
    else:
        # Load all users (for collaborative filtering)
        rows = conn.execute("""
            SELECT username, film_slug as slug, rating, watched, watchlisted, liked
            FROM user_films
        """).fetchall()

    for row in rows:
        row_dict = dict(row)
        username = row_dict.pop('username')
        all_user_films[username].append(row_dict)

    return dict(all_user_films)


def _scrape_film_metadata(scraper: 'LetterboxdScraper', slugs: list[str], max_per_batch: int = 100, use_async: bool = True) -> None:
    """Helper to scrape film metadata for a list of slugs."""
    if not slugs:
        return

    limited_slugs = slugs[:max_per_batch]

    if use_async and len(limited_slugs) > 10:
        from .scraper import AsyncLetterboxdScraper
        logger.info(f"Fetching {len(limited_slugs)} films (async)...")

        async def scrape_batch():
            async with AsyncLetterboxdScraper(delay=0.2, max_concurrent=5) as async_scraper:
                return await async_scraper.scrape_films_batch(limited_slugs)

        metadata_list = asyncio.run(scrape_batch())
    else:
        metadata_list = []
        for slug in tqdm(limited_slugs, desc="Metadata"):
            meta = scraper.scrape_film(slug)
            if meta:
                metadata_list.append(meta)
    
    # Batch insert all metadata
    if metadata_list:
        with get_db() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO films 
                (slug, title, year, directors, genres, cast, themes, runtime, avg_rating, rating_count,
                 countries, languages, writers, cinematographers, composers)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [(
                m.slug, m.title, m.year,
                json.dumps(m.directors), json.dumps(m.genres),
                json.dumps(m.cast), json.dumps(m.themes),
                m.runtime, m.avg_rating, m.rating_count,
                json.dumps(m.countries), json.dumps(m.languages),
                json.dumps(m.writers), json.dumps(m.cinematographers),
                json.dumps(m.composers)
            ) for m in metadata_list])
            
            # Populate normalized tables for fast queries
            from .database import populate_normalized_tables
            for m in metadata_list:
                populate_normalized_tables(conn, m)
        
        logger.info(f"Saved {len(metadata_list)} films")


def cmd_scrape(args: argparse.Namespace) -> None:
    """Scrape a user's Letterboxd data."""
    init_db()
    
    # Validate and sanitize username
    username = _validate_username(args.username)
    
    scraper = LetterboxdScraper(delay=1.0)
    
    try:
        # Check if refresh is needed
        refresh = getattr(args, 'refresh', None)
        if refresh:
            with get_db() as conn:
                result = conn.execute("""
                    SELECT MAX(scraped_at) as last_scrape
                    FROM user_films
                    WHERE username = ?
                """, (username,)).fetchone()
                
                if result and result['last_scrape']:
                    last_scrape = datetime.fromisoformat(result['last_scrape'])
                    age_days = (datetime.now() - last_scrape).days

                    if age_days < args.refresh:
                        print(f"  Skipping {username} (last scraped {age_days} days ago, refresh threshold: {args.refresh} days)")
                        return
                    else:
                        print(f"  Refreshing {username} (last scraped {age_days} days ago)")

        # Check for incremental mode
        incremental = getattr(args, 'incremental', False)
        existing_slugs = set()

        if incremental:
            with get_db() as conn:
                existing_slugs = {
                    r['film_slug'] for r in conn.execute("""
                        SELECT film_slug FROM user_films WHERE username = ?
                    """, (username,))
                }
                logger.info(f"Incremental mode: found {len(existing_slugs)} existing films for {username}")

        interactions = scraper.scrape_user(username, existing_slugs=existing_slugs, stop_on_existing=incremental)

        # Batch insert user films within a transaction
        with get_db() as conn:
            # Start explicit transaction for atomicity
            conn.execute("BEGIN")
            try:
                conn.executemany("""
                    INSERT OR REPLACE INTO user_films
                    (username, film_slug, rating, watched, watchlisted, liked, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                """, [(username, i.film_slug, i.rating, i.watched, i.watchlisted, i.liked)
                      for i in interactions])

                existing = {r['slug'] for r in conn.execute("SELECT slug FROM films")}
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        new_slugs = [i.film_slug for i in interactions if i.film_slug not in existing]
        
        # Scrape and batch insert film metadata
        print("\nFetching film metadata...")
        _scrape_film_metadata(scraper, new_slugs)
        
        # Scrape user lists if enabled
        if args.include_lists:
            print(f"\nScraping {username}'s lists...")

            # Get profile favorites (4-film showcase)
            favorites = scraper.scrape_favorites(username)
            if favorites:
                print(f"  Found {len(favorites)} profile favorites")
                with get_db() as conn:
                    conn.execute("BEGIN")
                    try:
                        for slug in favorites:
                            conn.execute("""
                                INSERT OR REPLACE INTO user_lists
                                (username, list_slug, list_name, is_ranked, is_favorites, position, film_slug, scraped_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                username, "profile-favorites", "Profile Favorites",
                                0, 1, None, slug, datetime.now().isoformat()
                            ))
                        conn.commit()
                    except Exception:
                        conn.rollback()
                        raise
            
            # Get all user lists
            lists = scraper.scrape_user_lists(username, limit=args.max_lists)

            # Scrape films from each list
            for list_info in lists:
                list_slug = list_info['list_slug']
                list_name = list_info['list_name']
                is_ranked = list_info['is_ranked']

                # Detect favorites
                is_favorites = "favorite" in list_name.lower() or list_slug == "favorites"

                print(f"  Scraping list: {list_name}...")
                films = scraper.scrape_list_films(username, list_slug)
                
                if not films:
                    print(f"    (empty list)")
                    continue
                
                # Save to database
                with get_db() as conn:
                    conn.execute("BEGIN")
                    try:
                        for film in films:
                            conn.execute("""
                                INSERT OR REPLACE INTO user_lists
                                (username, list_slug, list_name, is_ranked, is_favorites, position, film_slug, scraped_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                username, list_slug, list_name,
                                is_ranked, is_favorites, film.get('position'),
                                film['film_slug'], datetime.now().isoformat()
                            ))
                        conn.commit()
                    except Exception:
                        conn.rollback()
                        raise
                
                print(f"    {len(films)} films")

        print(f"\nDone! {len(interactions)} films for {username}")
        
    finally:
        scraper.close()


def cmd_discover(args: argparse.Namespace) -> None:
    """Discover and scrape other users."""
    init_db()
    scraper = LetterboxdScraper(delay=1.0)
    
    try:
        # Get existing users
        with get_db() as conn:
            existing_users = {r['username'] for r in conn.execute("SELECT DISTINCT username FROM user_films")}
        
        all_discovered = []
        max_attempts = 5
        attempts = 0
        
        print(f"Target: {args.limit} new users (have {len(existing_users)} existing)")
        
        while len(all_discovered) < args.limit and attempts < max_attempts:
            attempts += 1
            batch_size = args.limit * (attempts + 1)
            
            # Discover users based on source
            if args.source == 'following':
                if not args.username:
                    print("--username is required for 'following' source")
                    return
                usernames = scraper.scrape_following(args.username, limit=batch_size)
            elif args.source == 'followers':
                if not args.username:
                    print("--username is required for 'followers' source")
                    return
                usernames = scraper.scrape_followers(args.username, limit=batch_size)
            elif args.source == 'popular':
                usernames = scraper.scrape_popular_members(limit=batch_size)
            elif args.source == 'film':
                if not args.film_slug:
                    print("--film-slug is required for 'film' source")
                    return
                usernames = scraper.scrape_film_fans(args.film_slug, limit=batch_size)
            else:
                print(f"Unknown source: {args.source}")
                return
            
            # Filter to new users only
            new_usernames = [u for u in usernames if u not in existing_users and u not in all_discovered]
            all_discovered.extend(new_usernames)
            
            print(f"  Attempt {attempts}: found {len(usernames)} users, {len(new_usernames)} new")
            
            if len(usernames) < batch_size:
                break
            
            if len(all_discovered) >= args.limit:
                break
        
        new_usernames = all_discovered[:args.limit]
        
        if not new_usernames:
            print(f"No new users found! All discovered users already in database.")
            return

        # Check for dry-run mode
        dry_run = getattr(args, 'dry_run', False)

        if dry_run:
            print(f"\n[DRY RUN] Would scrape {len(new_usernames)} new users:")
            for i, username in enumerate(new_usernames, 1):
                print(f"  {i}. {username}")
            print(f"\nTo actually scrape these users, run without --dry-run flag")
            return

        print(f"\nScraping {len(new_usernames)} new users...")

        # Scrape each user
        for username in tqdm(new_usernames, desc="Users"):
            try:
                interactions = scraper.scrape_user(username)

                with get_db() as conn:
                    conn.execute("BEGIN")
                    try:
                        conn.executemany("""
                            INSERT OR REPLACE INTO user_films
                            (username, film_slug, rating, watched, watchlisted, liked, scraped_at)
                            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                        """, [(username, i.film_slug, i.rating, i.watched, i.watchlisted, i.liked)
                              for i in interactions])

                        existing_film_slugs = {r['slug'] for r in conn.execute("SELECT slug FROM films")}
                        conn.commit()
                    except Exception:
                        conn.rollback()
                        raise

                existing_users.add(username)
                
                new_slugs = [i.film_slug for i in interactions if i.film_slug not in existing_film_slugs]
                _scrape_film_metadata(scraper, new_slugs, max_per_batch=100)
                
            except Exception as e:
                logger.error(f"Error scraping {username}: {e}")
        
        print(f"\nDone! Scraped {len(new_usernames)} new users.")
        
    finally:
        scraper.close()


def cmd_recommend(args: argparse.Namespace) -> None:
    """Generate recommendations."""
    # Validate username
    username = _validate_username(args.username)
    
    with get_db() as conn:
        user_films = [dict(r) for r in conn.execute("""
            SELECT film_slug as slug, rating, watched, watchlisted, liked
            FROM user_films WHERE username = ?
        """, (username,))]
        
        if not user_films:
            print(f"No data for '{username}'. Run: python main.py scrape {username}")
            return
        
        # Load film metadata (needed by both strategies)
        all_films = {r['slug']: dict(r) for r in conn.execute("SELECT * FROM films")}
        
        # Check for missing metadata
        seen_slugs = {f['slug'] for f in user_films}
        missing_slugs = seen_slugs - set(all_films.keys())
        if missing_slugs:
            logger.warning(f"Missing metadata for {len(missing_slugs)} films in {username}'s history. Consider running 'scrape' again.")
        
        strategy = getattr(args, 'strategy', 'metadata')
        
        if strategy == 'hybrid':
            # Hybrid: combine metadata and collaborative (single query)
            all_user_films = _load_all_user_films(conn)
            
            # Get recommendations from both strategies
            meta_rec = MetadataRecommender(list(all_films.values()))
            collab_rec = CollaborativeRecommender(all_user_films, all_films)
            
            meta_recs = meta_rec.recommend(
                user_films,
                n=args.limit * 2,
                min_year=args.min_year,
                max_year=args.max_year,
                genres=args.genres,
                exclude_genres=args.exclude_genres,
                min_rating=args.min_rating,
                username=username
            )
            collab_recs = collab_rec.recommend(
                username,
                n=args.limit * 2,
                min_year=args.min_year,
                max_year=args.max_year
            )
            
            # Merge by rank (Borda count style)
            scores = {}
            reasons_map = {}
            
            for rank, r in enumerate(meta_recs):
                scores[r.slug] = scores.get(r.slug, 0) + (len(meta_recs) - rank)
                if r.slug not in reasons_map:
                    reasons_map[r.slug] = []
                reasons_map[r.slug].extend(r.reasons)

            for rank, r in enumerate(collab_recs):
                scores[r.slug] = scores.get(r.slug, 0) + (len(collab_recs) - rank)
                if r.slug not in reasons_map:
                    reasons_map[r.slug] = []
                reasons_map[r.slug].extend(r.reasons)
            
            # Build final recommendations
            merged = sorted(scores.items(), key=lambda x: -x[1])[:args.limit]
            recs = []
            for slug, score in merged:
                if slug in all_films:
                    film = all_films[slug]
                    # Deduplicate reasons while preserving order
                    unique_reasons = list(dict.fromkeys(reasons_map.get(slug, [])))
                    recs.append(Recommendation(
                        slug=slug,
                        title=film.get('title', slug),
                        year=film.get('year'),
                        score=score,
                        reasons=unique_reasons[:3] # Top 3 combined reasons
                    ))
                    
        elif strategy == 'collaborative':
            # Load all user data (single query)
            all_user_films = _load_all_user_films(conn)
            
            recommender = CollaborativeRecommender(all_user_films, all_films)
            recs = recommender.recommend(
                username,
                n=args.limit,
                min_year=args.min_year,
                max_year=args.max_year
            )
        else:
            # Metadata-based (default)
            diversity = getattr(args, 'diversity', False)
            max_per_director = getattr(args, 'max_per_director', 2)
            # Load user lists for profile building
            user_lists = load_user_lists(username)
            recommender = MetadataRecommender(list(all_films.values()))
            recs = recommender.recommend(
                user_films,
                n=args.limit,
                min_year=args.min_year,
                max_year=args.max_year,
                genres=args.genres,
                exclude_genres=args.exclude_genres,
                min_rating=args.min_rating,
                diversity=diversity,
                max_per_director=max_per_director,
                username=username,
                user_lists=user_lists
            )
    
    # Format and print results
    output_format = getattr(args, 'format', 'text')
    if output_format == 'json':
        output = []
        for r in recs:
            film = all_films.get(r.slug, {})
            output.append({
                "title": r.title,
                "year": r.year,
                "slug": r.slug,
                "score": round(r.score, 2),
                "reasons": r.reasons,
                "url": f"https://letterboxd.com/film/{r.slug}/",
                "directors": load_json(film.get('directors', [])),
                "genres": load_json(film.get('genres', [])),
                "cast": load_json(film.get('cast', []))[:5],
                "themes": load_json(film.get('themes', [])),
                "countries": load_json(film.get('countries', [])),
                "avg_rating": film.get('avg_rating'),
                "rating_count": film.get('rating_count')
            })
        print(json.dumps(output, indent=2))
    elif output_format == 'csv':
        print("Title,Year,URL,Score,Reasons")
        for r in recs:
            reasons = "; ".join(r.reasons).replace('"', '""')
            print(f'"{r.title}",{r.year},https://letterboxd.com/film/{r.slug}/,{r.score:.2f},"{reasons}"')
    elif output_format == 'markdown':
        print(f"\n# Top {len(recs)} recommendations for {username} ({strategy})\n")
        for i, r in enumerate(recs, 1):
            print(f"## {i}. [{r.title} ({r.year})](https://letterboxd.com/film/{r.slug}/)")
            print(f"**Score**: {r.score:.1f}  ")
            print(f"**Why**: {', '.join(r.reasons)}\n")
    else:
        print(f"\nTop {len(recs)} recommendations for {username} ({strategy}):")
        for i, r in enumerate(recs, 1):
            print(f"{i}. {r.title} ({r.year}) - Score: {r.score:.1f}")
            print(f"   Why: {', '.join(r.reasons)}")


def cmd_stats(args: argparse.Namespace) -> None:
    """Show database statistics."""
    with get_db() as conn:
        user_count = conn.execute("SELECT COUNT(DISTINCT username) FROM user_films").fetchone()[0]
        film_count = conn.execute("SELECT COUNT(*) FROM films").fetchone()[0]
        interaction_count = conn.execute("SELECT COUNT(*) FROM user_films").fetchone()[0]
        rated_count = conn.execute("SELECT COUNT(*) FROM user_films WHERE rating IS NOT NULL").fetchone()[0]
        
        print(f"\nDatabase Statistics:")
        print(f"  Users: {user_count}")
        print(f"  Films: {film_count}")
        print(f"  Total interactions: {interaction_count}")
        print(f"  Rated interactions: {rated_count}")
        
        if user_count > 0:
            top_users = conn.execute("""
                SELECT username, COUNT(*) as film_count 
                FROM user_films 
                GROUP BY username 
                ORDER BY film_count DESC 
                LIMIT 5
            """).fetchall()
            
            print(f"\nTop users by film count:")
            for user, count in top_users:
                print(f"  {user}: {count} films")
        
        verbose = getattr(args, 'verbose', False)
        if verbose:
            missing_metadata = conn.execute("""
                SELECT COUNT(DISTINCT uf.film_slug) 
                FROM user_films uf 
                LEFT JOIN films f ON uf.film_slug = f.slug 
                WHERE f.slug IS NULL
            """).fetchone()[0]
            
            print(f"\n  Films without metadata: {missing_metadata}")
            
            oldest = conn.execute("""
                SELECT username, MIN(scraped_at) as oldest_scrape
                FROM user_films 
                GROUP BY username 
                ORDER BY oldest_scrape 
                LIMIT 5
            """).fetchall()
            
            if oldest:
                print(f"\nOldest scraped users:")
                for user, scrape_time in oldest:
                    print(f"  {user}: {scrape_time}")
            
            film_genres = conn.execute("SELECT genres FROM films WHERE genres IS NOT NULL").fetchall()
            from collections import Counter
            genre_counts = Counter()
            for (genres_json,) in film_genres:
                genres = load_json(genres_json)
                for g in genres:
                    genre_counts[g] += 1
            
            if genre_counts:
                print(f"\nTop genres in database:")
                for genre, count in genre_counts.most_common(10):
                    print(f"  {genre}: {count} films")


def cmd_export(args: argparse.Namespace) -> None:
    """Export database to JSON file."""
    with get_db() as conn:
        user_films = [dict(r) for r in conn.execute("SELECT * FROM user_films")]
        films = [dict(r) for r in conn.execute("SELECT * FROM films")]
    
    data = {
        "user_films": user_films,
        "films": films,
        "exported_at": datetime.now().isoformat()
    }
    
    with open(args.file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported {len(user_films)} user interactions and {len(films)} films to {args.file}")


def cmd_import(args: argparse.Namespace) -> None:
    """Import database from JSON file."""
    with open(args.file, 'r') as f:
        data = json.load(f)
    
    init_db()
    
    with get_db() as conn:
        if 'films' in data:
            for film in data['films']:
                conn.execute("""
                    INSERT OR REPLACE INTO films
                    (slug, title, year, directors, genres, cast, themes, runtime, avg_rating, rating_count,
                     countries, languages, writers, cinematographers, composers)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    film['slug'], film.get('title'), film.get('year'),
                    film.get('directors'), film.get('genres'),
                    film.get('cast'), film.get('themes'),
                    film.get('runtime'), film.get('avg_rating'), film.get('rating_count'),
                    film.get('countries'), film.get('languages'),
                    film.get('writers'), film.get('cinematographers'), film.get('composers')
                ))
            print(f"Imported {len(data['films'])} films")
        
        if 'user_films' in data:
            for uf in data['user_films']:
                conn.execute("""
                    INSERT OR REPLACE INTO user_films 
                    (username, film_slug, rating, watched, watchlisted, liked, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    uf['username'], uf['film_slug'], uf.get('rating'),
                    uf.get('watched'), uf.get('watchlisted'), uf.get('liked'),
                    uf.get('scraped_at')
                ))
            print(f"Imported {len(data['user_films'])} user interactions")
    
    print(f"Import completed from {args.file}")


def cmd_profile(args: argparse.Namespace) -> None:
    """Show user's preference profile."""
    with get_db() as conn:
        user_films = [dict(r) for r in conn.execute("""
            SELECT film_slug as slug, rating, watched, watchlisted, liked
            FROM user_films WHERE username = ?
        """, (args.username,))]
        all_films = {r['slug']: dict(r) for r in conn.execute("SELECT * FROM films")}
    
    if not user_films:
        print(f"No data for '{args.username}'. Run: python main.py scrape {args.username}")
        return
    
    profile = build_profile(user_films, all_films, username=args.username)
    
    print(f"\nProfile for {args.username}")
    print(f"  Films: {profile.n_films} ({profile.n_rated} rated, {profile.n_liked} liked)")
    if profile.avg_liked_rating:
        print(f"  Average rating: {profile.avg_liked_rating:.2f}★")
    
    if profile.genres:
        print("\nTop genres:")
        for g, score in sorted(profile.genres.items(), key=lambda x: -x[1])[:10]:
            print(f"  {g}: {score:+.2f}")
    
    if profile.directors:
        print("\nTop directors:")
        for d, score in sorted(profile.directors.items(), key=lambda x: -x[1])[:10]:
            print(f"  {d}: {score:+.2f}")
    
    if profile.actors:
        print("\nTop actors:")
        for a, score in sorted(profile.actors.items(), key=lambda x: -x[1])[:10]:
            print(f"  {a}: {score:+.2f}")
    
    if profile.decades:
        print("\nDecade preferences:")
        for dec in sorted(profile.decades.keys()):
            score = profile.decades[dec]
            bar_length = int(max(0, score * 2))
            bar = "█" * bar_length
            print(f"  {dec}s: {bar} ({score:+.1f})")


def cmd_similar(args: argparse.Namespace) -> None:
    """Find films similar to a specific film."""
    # Validate slug
    slug = _validate_slug(args.slug)
    
    with get_db() as conn:
        all_films = [dict(r) for r in conn.execute("SELECT * FROM films")]
    
    if not all_films:
        print("No films in database. Run scrape first.")
        return
    
    recommender = MetadataRecommender(all_films)
    recs = recommender.similar_to(slug, n=args.limit)
    
    if not recs:
        print(f"No film found with slug '{args.slug}'")
        return
    
    print(f"\nFilms similar to {args.slug}:")
    for i, r in enumerate(recs, 1):
        print(f"{i}. {r.title} ({r.year}) - Score: {r.score:.1f}")
        print(f"   Why: {', '.join(r.reasons)}")


def cmd_triage(args: argparse.Namespace) -> None:
    """Rank user's watchlist by predicted enjoyment."""
    with get_db() as conn:
        user_films = [dict(r) for r in conn.execute("""
            SELECT film_slug as slug, rating, watched, watchlisted, liked
            FROM user_films WHERE username = ?
        """, (args.username,))]
        
        watchlist = [r['film_slug'] for r in conn.execute("""
            SELECT film_slug FROM user_films 
            WHERE username = ? AND watchlisted = 1
        """, (args.username,))]
        
        all_films = {r['slug']: dict(r) for r in conn.execute("SELECT * FROM films")}
    
    if not user_films:
        print(f"No data for '{args.username}'. Run: python main.py scrape {args.username}")
        return
    
    if not watchlist:
        print(f"No watchlist data for '{args.username}'.")
        return
    
    recommender = MetadataRecommender(list(all_films.values()))
    recs = recommender.recommend_from_candidates(user_films, watchlist, n=args.limit)
    
    print(f"\nWatchlist Triage for {args.username} (Top {len(recs)}):")
    for i, r in enumerate(recs, 1):
        print(f"{i}. {r.title} ({r.year}) - Score: {r.score:.1f}")
        print(f"   Why: {', '.join(r.reasons)}")


def cmd_gaps(args: argparse.Namespace) -> None:
    """Find gaps in filmography of favorite directors."""
    with get_db() as conn:
        user_films = [dict(r) for r in conn.execute("""
            SELECT film_slug as slug, rating, watched, watchlisted, liked
            FROM user_films WHERE username = ?
        """, (args.username,))]
        
        all_films = {r['slug']: dict(r) for r in conn.execute("SELECT * FROM films")}
    
    if not user_films:
        print(f"No data for '{args.username}'. Run: python main.py scrape {args.username}")
        return
    
    recommender = MetadataRecommender(list(all_films.values()))
    gaps = recommender.find_gaps(
        user_films,
        min_director_score=args.min_score,
        limit_per_director=args.limit,
        min_year=args.min_year,
        max_year=args.max_year
    )
    
    if not gaps:
        print(f"No gaps found for {args.username}. Try lowering --min-score.")
        return
    
    print(f"\nFilmography Gaps for {args.username}:")
    for director, recs in sorted(gaps.items(), key=lambda x: -len(x[1])):
        print(f"\n{director}:")
        for r in recs:
            print(f"  - {r.title} ({r.year}) [{r.score:.1f}]")


def main():
    parser = argparse.ArgumentParser(description="Letterboxd Recommender")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Scrape user data")
    scrape_parser.add_argument("username", help="Letterboxd username")
    scrape_parser.add_argument("--refresh", type=int, metavar="DAYS",
                                help="Only scrape if last scraped more than N days ago")
    scrape_parser.add_argument("--include-lists", action="store_true", default=True,
                                help="Include user lists (favorites, ranked lists)")
    scrape_parser.add_argument("--no-include-lists", dest="include_lists", action="store_false",
                                help="Skip scraping user lists")
    scrape_parser.add_argument("--max-lists", type=int, default=50,
                                help="Maximum number of lists to scrape per user (default: 50)")
    scrape_parser.add_argument("--incremental", action="store_true",
                                help="Stop pagination when hitting already-scraped films (faster for updates)")
    scrape_parser.set_defaults(func=cmd_scrape)
    
    # Discover command
    discover_parser = subparsers.add_parser("discover", help="Discover and scrape other users")
    discover_parser.add_argument("source", 
                                  choices=['following', 'followers', 'popular', 'film'], 
                                  help="Source for user discovery")
    discover_parser.add_argument("--username", help="Username (for following/followers)")
    discover_parser.add_argument("--film-slug", help="Film slug (for film source, e.g., 'perfect-blue')")
    discover_parser.add_argument("--limit", type=int, default=50, help="Number of NEW users to scrape")
    discover_parser.add_argument("--dry-run", action="store_true",
                                  help="Show which users would be scraped without actually scraping")
    discover_parser.set_defaults(func=cmd_discover)
    
    # Recommend command
    rec_parser = subparsers.add_parser("recommend", help="Generate recommendations")
    rec_parser.add_argument("username", help="Letterboxd username")
    rec_parser.add_argument("--strategy", choices=['metadata', 'collaborative', 'hybrid'], 
                            default='metadata', help="Recommendation strategy")
    rec_parser.add_argument("--limit", type=int, default=20, help="Number of recommendations")
    rec_parser.add_argument("--min-year", type=int, help="Minimum release year")
    rec_parser.add_argument("--max-year", type=int, help="Maximum release year")
    rec_parser.add_argument("--genres", nargs="+", help="Filter by genres (metadata only)")
    rec_parser.add_argument("--exclude-genres", nargs="+", help="Exclude genres (metadata only)")
    rec_parser.add_argument("--min-rating", type=float, help="Minimum community rating (metadata only)")
    rec_parser.add_argument("--diversity", action="store_true", help="Enable diversity mode (metadata only)")
    rec_parser.add_argument("--max-per-director", type=int, default=2, help="Max films per director (diversity mode)")
    rec_parser.add_argument("--format", choices=['text', 'json', 'markdown', 'csv'], default='text',
                            help="Output format")
    rec_parser.set_defaults(func=cmd_recommend)
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.set_defaults(func=cmd_stats)
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export database to JSON")
    export_parser.add_argument("file", help="Output JSON file path")
    export_parser.set_defaults(func=cmd_export)
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import database from JSON")
    import_parser.add_argument("file", help="Input JSON file path")
    import_parser.set_defaults(func=cmd_import)
    
    # Profile command
    profile_parser = subparsers.add_parser("profile", help="Show user's preference profile")
    profile_parser.add_argument("username", help="Letterboxd username")
    profile_parser.set_defaults(func=cmd_profile)
    
    # Similar command
    similar_parser = subparsers.add_parser("similar", help="Find films similar to a specific film")
    similar_parser.add_argument("slug", help="Film slug (e.g., 'the-matrix')")
    similar_parser.add_argument("--limit", type=int, default=10, help="Number of similar films")
    similar_parser.set_defaults(func=cmd_similar)
    
    # Triage command
    triage_parser = subparsers.add_parser("triage", help="Rank watchlist by predicted enjoyment")
    triage_parser.add_argument("username", help="Letterboxd username")
    triage_parser.add_argument("--limit", type=int, default=20, help="Number of films to show")
    triage_parser.set_defaults(func=cmd_triage)
    
    # Gaps command
    gaps_parser = subparsers.add_parser("gaps", help="Find essential missing films from favorite directors")
    gaps_parser.add_argument("username", help="Letterboxd username")
    gaps_parser.add_argument("--min-score", type=float, default=2.0, help="Minimum director affinity score")
    gaps_parser.add_argument("--limit", type=int, default=3, help="Max films per director")
    gaps_parser.add_argument("--min-year", type=int, help="Minimum release year")
    gaps_parser.add_argument("--max-year", type=int, help="Maximum release year")
    gaps_parser.set_defaults(func=cmd_gaps)
    
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    args.func(args)