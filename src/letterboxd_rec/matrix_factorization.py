import json
import logging
from datetime import datetime
from pathlib import Path
from .config import SVD_CACHE_PATH, SVD_IMPLICIT_RESIDUALS, SVD_ITEM_BIAS_REG
from .data_version import interactions_digest

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

logger = logging.getLogger(__name__)

# Baseline used only when a model is fitted on implicit signals with no star ratings at all
DEFAULT_GLOBAL_MEAN = 3.0

class SVDRecommender:
    """
    Matrix factorization recommender using truncated SVD.
    
    Decomposes user-item matrix R ≈ U @ Σ @ V^T where:
    - U captures user latent factors (taste dimensions)
    - V captures item latent factors (film characteristics)
    - Σ captures factor importance
    """
    
    def __init__(
        self,
        n_factors: int = 50,
        use_implicit: bool = True,
        implicit_weight: float = 1.0,
        item_bias_reg: float = SVD_ITEM_BIAS_REG,
    ):
        self.n_factors = n_factors
        self.item_bias_reg = item_bias_reg
        self.use_implicit = use_implicit
        self.implicit_weight = implicit_weight
        self.user_factors = None
        self.item_factors = None
        self.user_index = None
        self.item_index = None
        self.global_mean = 0.0
        self.user_biases = None
        self.item_biases = None
        self.metadata: dict | None = None
        self.is_fitted = False

    @staticmethod
    def compute_fingerprint(all_user_films: dict[str, list[dict]], hyperparams: dict | None = None) -> dict:
        """Fingerprint for cache validation: counts plus a digest of every interaction."""
        n_users = len(all_user_films)
        rated_slugs = set()
        n_ratings = 0
        for films in all_user_films.values():
            for f in films:
                if f.get('rating') is not None:
                    n_ratings += 1
                    rated_slugs.add(f['slug'])
        fp = {
            "n_users": n_users,
            "n_items": len(rated_slugs),
            "n_ratings": n_ratings,
            "data_digest": interactions_digest(all_user_films),
        }
        if hyperparams:
            fp["hyperparams"] = hyperparams
        return fp

    @staticmethod
    def _implicit_residual(f: dict) -> float:
        """Residual (stars above baseline) for an unrated interaction; strongest signal wins."""
        for key in ("liked", "watchlisted", "watched"):
            if f.get(key):
                return SVD_IMPLICIT_RESIDUALS[key]
        return 0.0

    def fit(self, all_user_films: dict[str, list[dict]]) -> 'SVDRecommender':
        """
        Fit SVD model to user-film rating data.

        Uses bias-corrected SVD: R_predicted = global_mean + user_bias + item_bias + U @ V^T

        Biases come from explicit star ratings only. An unrated like/watch/watchlist entry
        enters the residual matrix as a small positive residual (see SVD_IMPLICIT_RESIDUALS)
        rather than as a pseudo-rating, so it can never read as a dislike after centering.
        """
        # Reset fitted flag in case of re-use
        self.is_fitted = False

        usernames = list(all_user_films.keys())
        self.user_index = {u: i for i, u in enumerate(usernames)}
        self.item_index = {}

        rows, cols, data = [], [], []
        implicit_rows, implicit_cols, implicit_data = [], [], []
        for username, films in all_user_films.items():
            user_idx = self.user_index[username]
            for f in films:
                rating = f.get('rating')
                if rating is not None:
                    film_idx = self.item_index.setdefault(f['slug'], len(self.item_index))
                    rows.append(user_idx)
                    cols.append(film_idx)
                    data.append(rating)
                elif self.use_implicit:
                    residual = self._implicit_residual(f)
                    if residual > 0:
                        film_idx = self.item_index.setdefault(f['slug'], len(self.item_index))
                        implicit_rows.append(user_idx)
                        implicit_cols.append(film_idx)
                        implicit_data.append(residual * self.implicit_weight)

        if not data and not implicit_data:
            logger.warning("No ratings to fit SVD model")
            raise ValueError("Cannot fit SVD model because no ratings were provided.")

        n_users = len(usernames)
        n_items = len(self.item_index)
        rows_e = np.asarray(rows, dtype=np.int64)
        cols_e = np.asarray(cols, dtype=np.int64)
        vals_e = np.asarray(data, dtype=np.float64)

        # Biases from explicit ratings only (implicit signals are not on the star scale)
        self.global_mean = float(vals_e.mean()) if vals_e.size else DEFAULT_GLOBAL_MEAN

        user_counts = np.bincount(rows_e, minlength=n_users).astype(np.float64)
        user_sums = np.bincount(rows_e, weights=vals_e, minlength=n_users)
        self.user_biases = np.divide(
            user_sums, user_counts, out=np.full(n_users, self.global_mean), where=user_counts > 0
        ) - self.global_mean

        # Film biases: what's left after the user's bias, shrunk toward 0 for films with
        # few ratings (one 5-star rating must not outrank a consensus classic)
        item_counts = np.bincount(cols_e, minlength=n_items).astype(np.float64)
        item_resid = np.bincount(
            cols_e, weights=vals_e - self.global_mean - self.user_biases[rows_e], minlength=n_items
        )
        denom = item_counts + self.item_bias_reg
        self.item_biases = np.divide(item_resid, denom, out=np.zeros(n_items), where=denom > 0)

        # Residual matrix: explicit ratings minus baseline, implicit signals as positive residuals
        explicit_resid = vals_e - (self.global_mean + self.user_biases[rows_e] + self.item_biases[cols_e])
        R_centered = csr_matrix(
            (
                np.concatenate([explicit_resid, np.asarray(implicit_data, dtype=np.float64)]),
                (
                    np.concatenate([rows_e, np.asarray(implicit_rows, dtype=np.int64)]),
                    np.concatenate([cols_e, np.asarray(implicit_cols, dtype=np.int64)]),
                ),
            ),
            shape=(n_users, n_items),
            dtype=np.float64,
        )

        # Truncated SVD
        k = min(self.n_factors, min(n_users, n_items) - 1)
        if k < 1:
            raise ValueError("Cannot fit SVD model: need at least 2 users and 2 films.")
        U, sigma, Vt = svds(R_centered.astype(np.float64), k=k)
        
        # Store factors (incorporate sigma into both for symmetric treatment)
        sigma_sqrt = np.sqrt(sigma)
        self.user_factors = U * sigma_sqrt
        self.item_factors = (Vt.T * sigma_sqrt).T  # Shape: (k, n_items)
        self.is_fitted = True  # Mark model as ready for inference

        # Cache metadata for persistence
        self.metadata = {
            "fingerprint": self.compute_fingerprint(all_user_films),
            "n_users": n_users,
            "n_items": n_items,
            "n_ratings": len(data),
            "created_at": datetime.utcnow().isoformat(),
            "n_factors": self.n_factors,
            "use_implicit": self.use_implicit,
            "implicit_weight": self.implicit_weight,
            "item_bias_reg": self.item_bias_reg,
        }
        
        logger.info(f"Fitted SVD with {k} factors on {n_users} users × {n_items} items")
        return self
    
    def predict(self, username: str, film_slug: str) -> float | None:
        """Predict rating for a user-film pair."""
        if not self.is_fitted or self.user_factors is None or self.item_factors is None:
            logger.warning("SVD model is not fitted; cannot generate prediction.")
            return None

        if username not in self.user_index or film_slug not in self.item_index:
            return None
        
        user_idx = self.user_index[username]
        item_idx = self.item_index[film_slug]
        
        prediction = (
            self.global_mean +
            self.user_biases[user_idx] +
            self.item_biases[item_idx] +
            self.user_factors[user_idx] @ self.item_factors[:, item_idx]
        )
        
        # Clip to valid rating range
        return float(np.clip(prediction, 0.5, 5.0))
    
    def recommend(
        self,
        username: str,
        seen_slugs: set[str],
        n: int = 20
    ) -> list[tuple[str, float]]:
        """Generate top-N recommendations for a user."""
        if (
            not self.is_fitted
            or self.user_factors is None
            or self.item_factors is None
            or self.user_biases is None
            or self.item_biases is None
        ):
            logger.warning("SVD model is not fitted; returning no recommendations.")
            return []

        if username not in self.user_index:
            return []
        
        user_idx = self.user_index[username]
        
        # Predict all items at once
        user_vec = self.user_factors[user_idx]
        predictions = (
            self.global_mean +
            self.user_biases[user_idx] +
            self.item_biases +
            user_vec @ self.item_factors
        )
        
        # Mask seen items
        item_slugs = list(self.item_index.keys())
        results = []
        for idx in np.argsort(predictions)[::-1]:
            slug = item_slugs[idx]
            if slug not in seen_slugs:
                results.append((slug, float(predictions[idx])))
                if len(results) >= n:
                    break
        
        return results

    @classmethod
    def load(cls, path: str | Path = SVD_CACHE_PATH, expected_fingerprint: dict | None = None) -> 'SVDRecommender | None':
        """Load a cached SVD model if it matches the expected fingerprint."""
        cache_path = Path(path)
        if not cache_path.exists():
            return None

        try:
            data = np.load(cache_path, allow_pickle=True)
            raw_meta = data["metadata"]
            metadata = json.loads(raw_meta.item() if hasattr(raw_meta, "item") else str(raw_meta))

            if expected_fingerprint and metadata.get("fingerprint") != expected_fingerprint:
                logger.info("Cached SVD fingerprint mismatch; ignoring cache.")
                return None

            n_factors = int(data["n_factors"])
            inst = cls(n_factors=n_factors)
            inst.user_factors = data["user_factors"]
            inst.item_factors = data["item_factors"]
            inst.user_biases = data["user_biases"]
            inst.item_biases = data["item_biases"]
            inst.global_mean = float(data["global_mean"])

            user_index_list = data["user_index"].tolist()
            item_index_list = data["item_index"].tolist()
            inst.user_index = {u: i for i, u in enumerate(user_index_list)}
            inst.item_index = {s: i for i, s in enumerate(item_index_list)}
            inst.metadata = metadata
            inst.is_fitted = True

            logger.info(f"Loaded cached SVD model from {cache_path}")
            return inst
        except Exception as e:
            logger.warning(f"Failed to load cached SVD model: {e}")
            return None

    def save(self, path: str | Path = SVD_CACHE_PATH, metadata: dict | None = None) -> None:
        """Persist the fitted model to disk for reuse."""
        if self.user_factors is None or self.item_factors is None:
            logger.debug("SVD model not fitted; skipping save.")
            return

        cache_path = Path(path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        payload = metadata or self.metadata or {}
        payload.setdefault("created_at", datetime.utcnow().isoformat())
        payload.setdefault("fingerprint", None)
        payload.setdefault("n_factors", self.n_factors)

        np.savez_compressed(
            cache_path,
            user_factors=self.user_factors,
            item_factors=self.item_factors,
            user_biases=self.user_biases,
            item_biases=self.item_biases,
            user_index=np.array(list(self.user_index.keys())),
            item_index=np.array(list(self.item_index.keys())),
            global_mean=self.global_mean,
            n_factors=self.n_factors,
            metadata=json.dumps(payload),
        )
        logger.info(f"Saved SVD model to {cache_path}")