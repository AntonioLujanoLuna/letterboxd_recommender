import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import logging

logger = logging.getLogger(__name__)

class SVDRecommender:
    """
    Matrix factorization recommender using truncated SVD.
    
    Decomposes user-item matrix R ≈ U @ Σ @ V^T where:
    - U captures user latent factors (taste dimensions)
    - V captures item latent factors (film characteristics)
    - Σ captures factor importance
    """
    
    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.user_index = None
        self.item_index = None
        self.global_mean = 0.0
        self.user_biases = None
        self.item_biases = None
    
    def fit(self, all_user_films: dict[str, list[dict]]) -> 'SVDRecommender':
        """
        Fit SVD model to user-film rating data.
        
        Uses bias-corrected SVD: R_predicted = global_mean + user_bias + item_bias + U @ V^T
        """
        # Build sparse matrix
        usernames = list(all_user_films.keys())
        self.user_index = {u: i for i, u in enumerate(usernames)}
        
        all_films = set()
        for films in all_user_films.values():
            for f in films:
                if f.get('rating'):
                    all_films.add(f['slug'])
        
        film_list = list(all_films)
        self.item_index = {slug: i for i, slug in enumerate(film_list)}
        
        n_users = len(usernames)
        n_items = len(film_list)
        
        # Build COO data
        rows, cols, data = [], [], []
        for username, films in all_user_films.items():
            user_idx = self.user_index[username]
            for f in films:
                rating = f.get('rating')
                if rating and f['slug'] in self.item_index:
                    rows.append(user_idx)
                    cols.append(self.item_index[f['slug']])
                    data.append(rating)
        
        if not data:
            logger.warning("No ratings to fit SVD model")
            return self
        
        R = csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
        
        # Compute biases
        self.global_mean = np.mean(data)
        
        # User biases: mean rating per user - global mean
        user_sums = np.array(R.sum(axis=1)).flatten()
        user_counts = np.array(R.getnnz(axis=1), dtype=np.float32)
        user_counts[user_counts == 0] = 1
        self.user_biases = (user_sums / user_counts) - self.global_mean
        
        # Item biases: mean rating per item - global mean
        item_sums = np.array(R.sum(axis=0)).flatten()
        item_counts = np.array(R.getnnz(axis=0), dtype=np.float32)
        item_counts[item_counts == 0] = 1
        self.item_biases = (item_sums / item_counts) - self.global_mean
        
        # Center the matrix
        R_centered = R.copy().tocsr()
        for i in range(n_users):
            start, end = R_centered.indptr[i], R_centered.indptr[i + 1]
            for j in range(start, end):
                col = R_centered.indices[j]
                R_centered.data[j] -= (self.global_mean + self.user_biases[i] + self.item_biases[col])
        
        # Truncated SVD
        k = min(self.n_factors, min(n_users, n_items) - 1)
        U, sigma, Vt = svds(R_centered.astype(np.float64), k=k)
        
        # Store factors (incorporate sigma into both for symmetric treatment)
        sigma_sqrt = np.sqrt(sigma)
        self.user_factors = U * sigma_sqrt
        self.item_factors = (Vt.T * sigma_sqrt).T  # Shape: (k, n_items)
        
        logger.info(f"Fitted SVD with {k} factors on {n_users} users × {n_items} items")
        return self
    
    def predict(self, username: str, film_slug: str) -> float | None:
        """Predict rating for a user-film pair."""
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