import argparse
import json
from .database import init_db, get_db
from .scraper import LetterboxdScraper
from .recommender import MetadataRecommender, CollaborativeRecommender
from tqdm import tqdm

def cmd_scrape(args):
    """Scrape a user's Letterboxd data."""
    init_db()
    scraper = LetterboxdScraper(delay=1.0)
    
    try:
        interactions = scraper.scrape_user(args.username)
        
        with get_db() as conn:
            for i in interactions:
                conn.execute("""
                    INSERT OR REPLACE INTO user_films 
                    (username, film_slug, rating, watched, watchlisted, liked, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                """, (args.username, i.film_slug, i.rating, i.watched, i.watchlisted, i.liked))
        
        # Scrape metadata for new films
        print("\nFetching film metadata...")
        with get_db() as conn:
            existing = {r['slug'] for r in conn.execute("SELECT slug FROM films")}
        
        new_slugs = [i.film_slug for i in interactions if i.film_slug not in existing]
        
        for slug in tqdm(new_slugs, desc="Metadata"):
            meta = scraper.scrape_film(slug)
            if meta:
                with get_db() as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO films 
                        (slug, title, year, directors, genres, cast, themes, runtime, avg_rating, rating_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        meta.slug, meta.title, meta.year,
                        json.dumps(meta.directors), json.dumps(meta.genres),
                        json.dumps(meta.cast), json.dumps(meta.themes),
                        meta.runtime, meta.avg_rating, meta.rating_count
                    ))
        
        print(f"\nDone! {len(interactions)} films for {args.username}")
        
    finally:
        scraper.close()

def cmd_discover(args):
    """Discover and scrape other users."""
    init_db()
    scraper = LetterboxdScraper(delay=1.0)
    
    try:
        # Get existing users
        with get_db() as conn:
            existing_users = {r['username'] for r in conn.execute("SELECT DISTINCT username FROM user_films")}
        
        # Discover users, keep fetching until we have enough NEW users
        all_discovered = []
        fetch_limit = args.limit
        max_attempts = 5  # Don't fetch endlessly
        attempts = 0
        
        print(f"Target: {args.limit} new users")
        
        while len(all_discovered) < args.limit and attempts < max_attempts:
            attempts += 1
            batch_size = fetch_limit * (attempts + 1)  # Increase fetch size each attempt
            
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
            else:
                print(f"Unknown source: {args.source}")
                return
            
            # Filter to new users
            new_usernames = [u for u in usernames if u not in existing_users and u not in all_discovered]
            all_discovered.extend(new_usernames)
            
            if len(new_usernames) == 0:
                print(f"No more new users found after {attempts} attempts")
                break
        
        # Trim to limit
        new_usernames = all_discovered[:args.limit]
        total_found = len(set(usernames) if 'usernames' in locals() else [])
        skipped = total_found - len(new_usernames)
        
        if skipped > 0:
            print(f"Skipped {skipped} already-scraped users.")
        
        if not new_usernames:
            print("No new users to scrape!")
            return
        
        print(f"\nDiscovered {len(new_usernames)} new users. Scraping their data...")
        
        # Scrape each user
        for username in tqdm(new_usernames, desc="Users"):
            try:
                interactions = scraper.scrape_user(username)
                
                with get_db() as conn:
                    for i in interactions:
                        conn.execute("""
                            INSERT OR REPLACE INTO user_films 
                            (username, film_slug, rating, watched, watchlisted, liked, scraped_at)
                            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                        """, (username, i.film_slug, i.rating, i.watched, i.watchlisted, i.liked))
                
                # Add to existing users set
                existing_users.add(username)
                
                # Scrape metadata for new films
                with get_db() as conn:
                    existing = {r['slug'] for r in conn.execute("SELECT slug FROM films")}
                
                new_slugs = [i.film_slug for i in interactions if i.film_slug not in existing]
                
                for slug in new_slugs[:100]:  # Limit metadata scraping per user
                    meta = scraper.scrape_film(slug)
                    if meta:
                        with get_db() as conn:
                            conn.execute("""
                                INSERT OR REPLACE INTO films 
                                (slug, title, year, directors, genres, cast, themes, runtime, avg_rating, rating_count)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                meta.slug, meta.title, meta.year,
                                json.dumps(meta.directors), json.dumps(meta.genres),
                                json.dumps(meta.cast), json.dumps(meta.themes),
                                meta.runtime, meta.avg_rating, meta.rating_count
                            ))
            except Exception as e:
                print(f"Error scraping {username}: {e}")
        
        print(f"\nDone! Scraped {len(new_usernames)} new users.")
        
    finally:
        scraper.close()

def cmd_recommend(args):
    """Generate recommendations."""
    with get_db() as conn:
        user_films = [dict(r) for r in conn.execute("""
            SELECT film_slug as slug, rating, watched, watchlisted, liked
            FROM user_films WHERE username = ?
        """, (args.username,))]
        
        if not user_films:
            print(f"No data for '{args.username}'. Run: python main.py scrape {args.username}")
            return
        
        strategy = args.strategy if hasattr(args, 'strategy') else 'metadata'
        
        if strategy == 'collaborative':
            # Load all user data
            all_user_films = {}
            with get_db() as conn:
                all_users = [r['username'] for r in conn.execute("SELECT DISTINCT username FROM user_films")]
                for user in all_users:
                    all_user_films[user] = [dict(r) for r in conn.execute("""
                        SELECT film_slug as slug, rating, watched, watchlisted, liked
                        FROM user_films WHERE username = ?
                    """, (user,))]
            
            recommender = CollaborativeRecommender(all_user_films)
            recs = recommender.recommend(
                args.username,
                n=args.limit,
                min_year=args.min_year,
                max_year=args.max_year
            )
        else:
            # Metadata-based (default)
            all_films = [dict(r) for r in conn.execute("SELECT * FROM films")]
            recommender = MetadataRecommender(all_films)
            recs = recommender.recommend(
                user_films,
                n=args.limit,
                min_year=args.min_year,
                max_year=args.max_year,
                genres=args.genres,
                exclude_genres=args.exclude_genres,
                min_rating=args.min_rating
            )
    
    print(f"\nTop {len(recs)} recommendations for {args.username} ({strategy}):")
    for i, r in enumerate(recs, 1):
        print(f"{i}. {r.title} ({r.year}) - Score: {r.score:.1f}")
        print(f"   Why: {', '.join(r.reasons)}")

def main():
    parser = argparse.ArgumentParser(description="Letterboxd Recommender")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Scrape user data")
    scrape_parser.add_argument("username", help="Letterboxd username")
    scrape_parser.set_defaults(func=cmd_scrape)
    
    # Discover command
    discover_parser = subparsers.add_parser("discover", help="Discover and scrape other users")
    discover_parser.add_argument("source", 
                                  choices=['following', 'followers', 'popular'], 
                                  help="Source for user discovery")
    discover_parser.add_argument("--username", help="Username (for following/followers)")
    discover_parser.add_argument("--limit", type=int, default=50, help="Number of NEW users to scrape")
    discover_parser.set_defaults(func=cmd_discover)
    
    # Recommend command
    rec_parser = subparsers.add_parser("recommend", help="Generate recommendations")
    rec_parser.add_argument("username", help="Letterboxd username")
    rec_parser.add_argument("--strategy", choices=['metadata', 'collaborative'], 
                            default='metadata', help="Recommendation strategy")
    rec_parser.add_argument("--limit", type=int, default=20, help="Number of recommendations")
    rec_parser.add_argument("--min-year", type=int, help="Minimum release year")
    rec_parser.add_argument("--max-year", type=int, help="Maximum release year")
    rec_parser.add_argument("--genres", nargs="+", help="Filter by genres (metadata only)")
    rec_parser.add_argument("--exclude-genres", nargs="+", help="Exclude genres (metadata only)")
    rec_parser.add_argument("--min-rating", type=float, help="Minimum community rating (metadata only)")
    rec_parser.set_defaults(func=cmd_recommend)
    
    args = parser.parse_args()
    args.func(args)
