"""Collaborative filtering recommender implementation."""

import json
import logging
import math
from collections import defaultdict
from pathlib import Path

from ..database import load_json
from ..config import (
    COLLAB_SHRINKAGE,
    COLLAB_IMPLICIT_WEIGHTS,
    COLLAB_POPULARITY_DEBIAS,
    ITEM_SIM_CACHE_PATH,
    ITEM_SIM_MIN_RATINGS,
    ITEM_SIM_MAX_ITEMS,
    ITEM_SIM_BLOCK_CELLS,
)
from ..data_version import interactions_digest
from .recommendation import Recommendation

logger = logging.getLogger(__name__)


class CollaborativeRecommender:
    """
    Collaborative filtering recommender.
    Finds users with similar taste and recommends films they liked.

    Uses sparse matrices for efficient similarity computation on large datasets.
    """

    def __init__(
        self,
        all_user_films: dict[str, list[dict]],
        film_metadata: dict[str, dict] | None = None,
        use_cache: bool = True,
    ):
        """
        Args:
            all_user_films: Dict mapping username -> list of user_films dicts
            film_metadata: Optional dict mapping slug -> film metadata dict for filtering and display
            use_cache: Read/write the on-disk item-similarity cache. Disable for throwaway
                models (e.g. offline evaluation on a train split) so the real cache is untouched.
        """
        self.all_user_films = all_user_films
        self.films = film_metadata or {}
        self.use_cache = use_cache

        # Precompute user-item matrix for efficient similarity computation
        self._user_matrix = None
        self._user_index = None
        self._film_index = None
        self._normalized_matrix = None  # Cached normalized matrix for fast similarity
        self._overlap_matrix = None      # Cached binary overlap matrix
        self._item_similarities: dict[str, list[tuple[str, float]]] = {}
        self._item_sim_top_k = 50

        self._build_sparse_matrix()
        self._precompute_similarity_components()
        self._fingerprint = self._compute_fingerprint()
        self._maybe_load_item_similarity_cache()

    def _build_sparse_matrix(self):
        """
        Build sparse user-item rating matrix for efficient similarity computation.

        Creates a CSR matrix where rows are users and columns are films.
        Also builds index mappings for fast lookups.
        """
        from scipy.sparse import csr_matrix
        import numpy as np

        # Build user and film indexes
        usernames = list(self.all_user_films.keys())
        self._user_index = {username: idx for idx, username in enumerate(usernames)}

        # Collect all unique films
        all_films_set = set()
        for films in self.all_user_films.values():
            for film in films:
                all_films_set.add(film['slug'])

        all_films_list = list(all_films_set)
        self._film_index = {slug: idx for idx, slug in enumerate(all_films_list)}

        # Build sparse matrix (users × films)
        n_users = len(usernames)
        n_films = len(all_films_list)

        # Use COO format for building, then convert to CSR
        row_indices = []
        col_indices = []
        ratings = []

        for username, user_films in self.all_user_films.items():
            user_idx = self._user_index[username]
            for film in user_films:
                rating = film.get('rating')
                if rating:  # Only include rated films
                    film_idx = self._film_index.get(film['slug'])
                    if film_idx is not None:
                        row_indices.append(user_idx)
                        col_indices.append(film_idx)
                        ratings.append(rating)

        # Create sparse matrix
        if row_indices:
            self._user_matrix = csr_matrix(
                (ratings, (row_indices, col_indices)),
                shape=(n_users, n_films),
                dtype=np.float32
            )
        else:
            # Empty matrix if no ratings
            self._user_matrix = csr_matrix((n_users, n_films), dtype=np.float32)

        logger.debug(f"Built sparse user-item matrix: {n_users} users × {n_films} films, {len(ratings)} ratings")

    def _precompute_similarity_components(self):
        """
        Precompute normalized matrix and overlap matrix for fast similarity computation.

        This method performs the expensive mean-centering and normalization operations once,
        dramatically speeding up similarity computations in _find_neighbors.
        """
        import numpy as np
        from scipy.sparse import diags

        if self._user_matrix is None or self._user_matrix.nnz == 0:
            logger.debug("No ratings to precompute similarity components")
            return

        n_users = self._user_matrix.shape[0]

        # Compute row means efficiently: sum / count
        row_sums = np.array(self._user_matrix.sum(axis=1)).flatten()
        row_nnz = np.array(self._user_matrix.getnnz(axis=1), dtype=np.float64)
        row_nnz[row_nnz == 0] = 1  # Avoid division by zero
        row_means = row_sums / row_nnz

        # Mean-center the matrix: subtract row mean from each non-zero entry
        centered = self._user_matrix.copy().tocsr()
        for i in range(n_users):
            start, end = centered.indptr[i], centered.indptr[i + 1]
            centered.data[start:end] -= row_means[i]

        # Normalize to unit length (for cosine similarity)
        row_norms = np.sqrt(np.array(centered.power(2).sum(axis=1)).flatten())
        row_norms[row_norms == 0] = 1  # Avoid division by zero
        inv_norms = diags(1.0 / row_norms)
        self._normalized_matrix = inv_norms @ centered

        # Binary overlap matrix (which films each user has rated)
        self._overlap_matrix = (self._user_matrix > 0).astype(np.float32)

        logger.debug(f"Precomputed similarity components for {n_users} users")

    def _compute_fingerprint(self) -> dict:
        n_users = len(self.all_user_films)
        n_ratings = int(self._user_matrix.nnz) if self._user_matrix is not None else 0
        n_items = len(self._film_index) if self._film_index else 0
        return {
            "n_users": n_users,
            "n_items": n_items,
            "n_ratings": n_ratings,
            "data_digest": interactions_digest(self.all_user_films),
        }

    def _maybe_load_item_similarity_cache(self):
        if not self.use_cache or not ITEM_SIM_CACHE_PATH.exists():
            return
        try:
            payload = json.loads(ITEM_SIM_CACHE_PATH.read_text())
            if payload.get("fingerprint") != self._fingerprint:
                return
            if payload.get("top_k") != self._item_sim_top_k:
                return
            data = payload.get("items", {})
            self._item_similarities = {
                slug: [(entry["slug"], entry["score"]) for entry in entries]
                for slug, entries in data.items()
            }
            logger.info(f"Loaded item-item similarity cache ({len(self._item_similarities)} items)")
        except Exception as e:
            logger.warning(f"Failed to load item similarity cache: {e}")

    def _save_item_similarity_cache(self):
        if not self.use_cache:
            return
        try:
            ITEM_SIM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "fingerprint": self._fingerprint,
                "top_k": self._item_sim_top_k,
                "items": {
                    slug: [{"slug": s, "score": float(sc)} for s, sc in sims]
                    for slug, sims in self._item_similarities.items()
                },
            }
            ITEM_SIM_CACHE_PATH.write_text(json.dumps(payload))
            logger.info(f"Saved item similarity cache to {ITEM_SIM_CACHE_PATH}")
        except Exception as e:
            logger.warning(f"Could not save item similarity cache: {e}")

    def _compute_item_similarities(self):
        """
        Compute item similarity using rating patterns, not just co-occurrence.

        Films are similar if users who rate one highly also rate the other highly.
        Adjusted cosine: ratings are centered by each user's mean rating, and both norms
        are taken over co-rating users only. Pairs need at least 2 co-raters and a
        similarity above 0.3.

        Vectorized as sparse matrix products over column blocks, so memory stays bounded
        (ITEM_SIM_BLOCK_CELLS). Benchmark: 2k users x 20k films in about a minute, where
        the previous pairwise Python loop took hours.
        """
        import numpy as np

        X = self._user_matrix
        if not self.all_user_films or X is None or X.nnz == 0:
            return

        logger.info("Computing item-item similarity cache...")
        X = X.tocsr().astype(np.float64)

        # Filter to reduce compute: drop very sparse items and cap total
        counts = X.getnnz(axis=0)
        eligible = np.flatnonzero(counts >= ITEM_SIM_MIN_RATINGS)
        if eligible.size == 0:
            logger.info("Item similarity: no items meet minimum rating count")
            return
        if eligible.size > ITEM_SIM_MAX_ITEMS:
            order = np.argsort(-counts[eligible], kind="stable")
            dropped = eligible.size - ITEM_SIM_MAX_ITEMS
            logger.info(
                f"Item similarity: trimming items from {eligible.size} to "
                f"{ITEM_SIM_MAX_ITEMS} (dropped {dropped}) to cap computation"
            )
            eligible = np.sort(eligible[order[:ITEM_SIM_MAX_ITEMS]])

        # User means over all of the user's ratings (not just eligible items)
        row_nnz = X.getnnz(axis=1)
        row_sums = np.asarray(X.sum(axis=1)).ravel()
        user_means = np.divide(row_sums, row_nnz, out=np.zeros_like(row_sums), where=row_nnz > 0)

        Xe = X[:, eligible].tocsr()
        Xe.sort_indices()
        # B (co-rating pattern) and C (centered ratings) share Xe's sparsity structure,
        # so a rating equal to the user's mean still counts as a co-rating.
        B = Xe.copy()
        B.data = np.ones_like(B.data)
        C = Xe.copy()
        C.data = Xe.data - np.repeat(user_means, np.diff(Xe.indptr))
        C2 = C.multiply(C).tocsr()

        Bt, Ct, C2t = B.T.tocsr(), C.T.tocsr(), C2.T.tocsr()
        B_csc, C_csc, C2_csc = B.tocsc(), C.tocsc(), C2.tocsc()

        m = eligible.size
        slugs_by_col = {idx: slug for slug, idx in self._film_index.items()}
        eligible_slugs = [slugs_by_col[int(c)] for c in eligible]
        top_k = self._item_sim_top_k
        block = max(1, ITEM_SIM_BLOCK_CELLS // max(m, 1))

        for j0 in range(0, m, block):
            j1 = min(j0 + block, m)
            dot = (Ct @ C_csc[:, j0:j1]).toarray()
            co = (Bt @ B_csc[:, j0:j1]).toarray()
            norm_a = (C2t @ B_csc[:, j0:j1]).toarray()   # sum of x_a^2 over co-raters
            norm_b = (Bt @ C2_csc[:, j0:j1]).toarray()   # sum of x_b^2 over co-raters

            denom = np.sqrt(norm_a * norm_b)
            sim = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
            valid = (co >= 2) & (norm_a > 0) & (norm_b > 0) & (sim > 0.3)
            cols = np.arange(j1 - j0)
            valid[j0 + cols, cols] = False  # no self-similarity

            for c in range(j1 - j0):
                nbrs = np.flatnonzero(valid[:, c])
                if nbrs.size == 0:
                    continue
                scores = sim[nbrs, c]
                if nbrs.size > top_k:
                    keep = np.argpartition(-scores, top_k - 1)[:top_k]
                    nbrs, scores = nbrs[keep], scores[keep]
                order = np.argsort(-scores, kind="stable")
                self._item_similarities[eligible_slugs[j0 + c]] = [
                    (eligible_slugs[int(nbrs[i])], float(scores[i])) for i in order
                ]

        self._save_item_similarity_cache()

    def _item_recommend_from_anchors(self, anchors: list[str], seen: set[str], top_k: int = 100) -> list[tuple[str, float, list[str]]]:
        if not anchors:
            return []
        if not self._item_similarities:
            self._compute_item_similarities()
        scores = {}
        reasons = {}
        for anchor in anchors:
            neighbors = self._item_similarities.get(anchor, [])
            for slug, sim in neighbors:
                if slug in seen:
                    continue
                scores[slug] = scores.get(slug, 0.0) + sim
                reasons.setdefault(slug, []).append(f"Similar to {anchor}")
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [(slug, score, reasons.get(slug, [])[:2]) for slug, score in ranked[:top_k]]

    def recommend(
        self,
        username: str,
        n: int = 20,
        min_neighbors: int = 3,
        min_year: int | None = None,
        max_year: int | None = None,
        genres: list[str] | None = None,
        exclude_genres: list[str] | None = None,
    ) -> list[Recommendation]:
        """Generate collaborative recommendations."""

        if username not in self.all_user_films:
            return []

        target_films = self.all_user_films[username]

        # Find neighbors (users with similar taste)
        neighbor_k = max(10, min_neighbors * 2)
        influencers, _ = self._find_neighbors_asymmetric(username, k=neighbor_k)
        base_neighbors = self._find_neighbors(username, target_films, k=neighbor_k)

        neighbor_scores: dict[str, float] = {}
        for user, sim in influencers:
            neighbor_scores[user] = sim
        for user, sim in base_neighbors:
            if user not in neighbor_scores or sim > neighbor_scores[user]:
                neighbor_scores[user] = sim

        neighbors = sorted(neighbor_scores.items(), key=lambda x: -x[1])[:neighbor_k]

        if len(neighbors) < min_neighbors:
            logger.warning(f"Warning: Only found {len(neighbors)} neighbors (min: {min_neighbors})")

        # Get films seen by target
        seen = {f['slug'] for f in target_films}

        # Score unseen films based on neighbor ratings
        film_scores = {}
        film_reasons = {}

        for neighbor_user, similarity in neighbors:
            neighbor_films = self.all_user_films[neighbor_user]

            for interaction in neighbor_films:
                slug = interaction['slug']
                if slug in seen:
                    continue

                # Apply filters if metadata available
                if self.films and slug in self.films:
                    film_meta = self.films[slug]
                    year = film_meta.get('year')
                    if min_year and year and year < min_year:
                        continue
                    if max_year and year and year > max_year:
                        continue

                    # Apply genre filters (genres are stored lowercase)
                    film_genres = load_json(film_meta.get('genres'))
                    if genres:
                        genres_lower = [g.lower() for g in genres]
                        if not any(g in film_genres for g in genres_lower):
                            continue
                    if exclude_genres:
                        exclude_genres_lower = [g.lower() for g in exclude_genres]
                        if any(g in film_genres for g in exclude_genres_lower):
                            continue

                rating = interaction.get('rating')  # Use original variable
                liked = interaction.get('liked', False)
                watched = interaction.get('watched', False)
                watchlisted = interaction.get('watchlisted', False)

                # Score based on rating or implicit feedback
                if rating is not None:
                    score = (rating - 2.5) * similarity  # normalize around mid-point
                else:
                    implicit = 0.0
                    if liked:
                        implicit += COLLAB_IMPLICIT_WEIGHTS["liked"]
                    if watched:
                        implicit += COLLAB_IMPLICIT_WEIGHTS["watched"]
                    if watchlisted:
                        implicit += COLLAB_IMPLICIT_WEIGHTS["watchlisted"]
                    if implicit == 0:
                        implicit = 0.05
                    score = implicit * similarity

                # Popularity debias: down-weight very popular films
                if self.films and slug in self.films:
                    rating_count = self.films[slug].get('rating_count') or 0
                else:
                    rating_count = 0
                if rating_count:
                    penalty = COLLAB_POPULARITY_DEBIAS * (math.log1p(rating_count) / math.log1p(100_000))
                    score *= max(0.1, 1 - penalty)

                if slug not in film_scores:
                    film_scores[slug] = 0
                    film_reasons[slug] = []

                film_scores[slug] += score

                # Track who recommended it
                if score > 0.5 and len(film_reasons[slug]) < 3:
                    film_reasons[slug].append(f"Liked by {neighbor_user}")

        # Item-based fallback for sparse neighborhoods
        if len(neighbors) < min_neighbors:
            anchor_slugs = [
                f['slug'] for f in target_films
                if (f.get('rating') and f['rating'] >= 3.5) or f.get('liked')
            ]
            item_based = self._item_recommend_from_anchors(anchor_slugs, seen, top_k=n * 2)
            for slug, score, reasons in item_based:
                film_scores[slug] = film_scores.get(slug, 0.0) + score
                film_reasons.setdefault(slug, []).extend(reasons)

        # Sort by score
        ranked = sorted(film_scores.items(), key=lambda x: -x[1])

        # Build results with film metadata if available
        results = []
        for slug, score in ranked[:n]:
            if self.films and slug in self.films:
                film = self.films[slug]
                title = film.get('title', slug)
                year = film.get('year')
            else:
                title = slug
                year = None

            results.append(Recommendation(
                slug=slug,
                title=title,
                year=year,
                score=score,
                reasons=film_reasons.get(slug, [])[:3]
            ))

        return results

    def _find_neighbors(self, username: str, target_films: list[dict], k: int = 10) -> list[tuple[str, float]]:
        """
        Find k most similar users using precomputed matrices.

        Uses adjusted cosine similarity (mean-centered ratings) which approximates
        Pearson correlation but is much faster for sparse matrices thanks to precomputation.

        Args:
            username: Target username
            target_films: Target user's film interactions (unused, kept for API compatibility)
            k: Number of neighbors to return

        Returns:
            List of (username, similarity_score) tuples, sorted by score descending
        """
        import numpy as np

        if username not in self._user_index or self._normalized_matrix is None:
            return []

        target_idx = self._user_index[username]
        n_users = self._normalized_matrix.shape[0]

        # Similarity = normalized_matrix @ target_row.T (single sparse matrix-vector multiply)
        target_row = self._normalized_matrix[target_idx]
        # sparse @ sparse returns a sparse matrix; np.asarray() would wrap it in a 0-d object
        # array rather than densify it, so convert explicitly.
        similarities = (self._normalized_matrix @ target_row.T).toarray().ravel()

        # Overlap counts for confidence weighting
        target_binary = self._overlap_matrix[target_idx]
        # Force dense array to avoid sparse truthiness/boolean issues downstream
        overlap_vec = self._overlap_matrix @ target_binary.T
        overlaps = np.asarray(overlap_vec.toarray()).ravel()

        # Filter and weight (be lenient on tiny datasets)
        min_overlap = 2 if n_users < 20 or self._overlap_matrix.shape[1] < 50 else 5
        valid = (overlaps >= min_overlap) & (np.arange(n_users) != target_idx)
        confidence = np.minimum(overlaps / 20.0, 1.0)  # Full confidence at 20+ common films
        shrinkage = overlaps / (overlaps + COLLAB_SHRINKAGE)

        # Apply confidence weighting and filter invalid
        weighted = np.where(valid, similarities * confidence * shrinkage, -np.inf)

        # Get top-k using partial sort (more efficient than full sort)
        if k >= len(weighted):
            top_k = np.argsort(weighted)[::-1]
        else:
            top_k = np.argpartition(weighted, -k)[-k:]
            top_k = top_k[np.argsort(weighted[top_k])[::-1]]

        # Filter to positive similarities only
        top_k = [i for i in top_k if weighted[i] > 0]

        # Build result list
        usernames = list(self._user_index.keys())
        return [(usernames[i], weighted[i]) for i in top_k]

    def _find_neighbors_asymmetric(
        self,
        username: str,
        k: int = 10
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """
        Find neighbors with asymmetric similarity.

        Returns:
            influencers: Users whose taste predicts yours (more experienced, you tend to agree)
            followers: Users who tend to agree with your ratings (less experienced, follow your taste)
        """
        if username not in self._user_index:
            return [], []

        target_films = self.all_user_films[username]
        target_ratings = {f['slug']: f.get('rating') for f in target_films if f.get('rating') is not None}

        if not target_ratings:
            return [], []

        influencer_scores: dict[str, float] = {}
        follower_scores: dict[str, float] = {}

        for other_username, other_films in self.all_user_films.items():
            if other_username == username:
                continue

            other_ratings = {f['slug']: f.get('rating') for f in other_films if f.get('rating') is not None}
            common = set(target_ratings.keys()) & set(other_ratings.keys())

            # Permit very small overlaps for sparse datasets
            if len(common) < 1:
                continue

            # Compute agreement score
            agreements = []
            for slug in common:
                diff = abs(target_ratings[slug] - other_ratings[slug])
                agreements.append(1.0 - (diff / 4.5))  # Normalize to 0-1

            agreement_score = sum(agreements) / len(agreements)
            overlap_factor = len(common) / 20  # Scale by overlap size

            # Experience ratio determines influencer vs follower
            target_experience = len(target_ratings)
            other_experience = len(other_ratings)
            experience_ratio = other_experience / max(target_experience, 1)

            if experience_ratio > 1.0:
                # They have more experience -> potential influencer
                influencer_weight = min(experience_ratio, 2.0)
                influencer_scores[other_username] = agreement_score * influencer_weight * overlap_factor
            else:
                # They have less experience -> potential follower
                follower_weight = min(1.0 / max(experience_ratio, 0.1), 2.0)
                follower_scores[other_username] = agreement_score * follower_weight * overlap_factor

        sorted_influencers = sorted(influencer_scores.items(), key=lambda x: -x[1])[:k]
        sorted_followers = sorted(follower_scores.items(), key=lambda x: -x[1])[:k]

        return sorted_influencers, sorted_followers
