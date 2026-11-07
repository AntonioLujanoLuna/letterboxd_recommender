"""Content fingerprints used to invalidate model caches when the data changes."""

import hashlib


def interactions_digest(all_user_films: dict[str, list[dict]]) -> str:
    """
    Stable hash of every user-film interaction.

    Any change to a rating or a liked/watched/watchlisted flag changes the digest,
    unlike row counts, which survive edits and same-size re-scrapes.
    """
    h = hashlib.blake2b(digest_size=16)
    for username in sorted(all_user_films):
        rows = sorted(
            f"{f['slug']}\t{f.get('rating')}\t{int(bool(f.get('liked')))}"
            f"{int(bool(f.get('watched')))}{int(bool(f.get('watchlisted')))}"
            for f in all_user_films[username]
        )
        h.update(username.encode("utf-8"))
        h.update(b"\x00")
        h.update("\n".join(rows).encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()
