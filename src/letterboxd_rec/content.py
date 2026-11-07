"""
Vectorized content-based recommender.

Films become rows of a sparse film x feature matrix F (directors, genres, themes, cast,
writers, countries, languages, crew, decade). Each feature is IDF-weighted, so a shared
rare director counts for far more than a shared "drama", then scaled by a per-field
weight, and each row is L2-normalized.

A user's taste vector is a weighted sum of the rows of the films they've seen:

    u = sum_f w_f * F[f],   w_f = (rating - user's mean rating) (+ a bonus if liked)

and every film is scored at once with F @ u. That takes milliseconds, where the
rule-based MetadataRecommender loops over every film in Python. It is also a linear
model whose field weights could later be fitted rather than hand-set.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.sparse import csr_matrix, diags

from .config import CONTENT_FIELD_WEIGHTS, CONTENT_MAX_CAST
from .database import load_json

logger = logging.getLogger(__name__)

# Weight of unrated interactions in the taste vector
UNRATED_WEIGHTS = {"liked": 1.0, "watched": 0.15, "watchlisted": 0.3}
LIKED_BONUS = 0.5  # added to the centered rating when a rated film is also liked

FIELD_LABELS = {
    "directors": "Director", "genres": "Genre", "themes": "Theme", "cast": "Cast",
    "writers": "Writer", "countries": "Country", "languages": "Language",
    "cinematographers": "Cinematographer", "composers": "Composer", "decade": "Decade",
}


def film_features(film: dict) -> list[tuple[str, str]]:
    """(field, value) pairs describing a film."""
    feats: list[tuple[str, str]] = []
    for field in CONTENT_FIELD_WEIGHTS:
        if field == "decade":
            year = film.get("year")
            if year:
                feats.append(("decade", str(int(year) // 10 * 10)))
            continue
        values = load_json(film.get(field))
        if not isinstance(values, list):
            continue
        if field == "cast":
            values = values[:CONTENT_MAX_CAST]
        for v in values:
            if v:
                feats.append((field, str(v).lower()))
    return feats


class ContentRecommender:
    def __init__(self, films: dict[str, dict], field_weights: dict[str, float] | None = None):
        self.field_weights = field_weights or CONTENT_FIELD_WEIGHTS
        self.slugs = list(films)
        self.index = {s: i for i, s in enumerate(self.slugs)}
        self.films = films
        self.F = self._build_matrix(films)

    def _build_matrix(self, films: dict[str, dict]) -> csr_matrix:
        per_film = [set(film_features(films[s])) for s in self.slugs]
        df: dict[tuple[str, str], int] = {}
        for feats in per_film:
            for feat in feats:
                df[feat] = df.get(feat, 0) + 1

        n = max(len(self.slugs), 1)
        # A feature on a single film can't link two films; drop it
        vocab = {feat: i for i, feat in enumerate(f for f, c in df.items() if c >= 2)}
        self.feature_names = list(vocab)

        rows, cols, vals = [], [], []
        for r, feats in enumerate(per_film):
            for feat in feats:
                c = vocab.get(feat)
                if c is None:
                    continue
                rows.append(r)
                cols.append(c)
                vals.append(self.field_weights.get(feat[0], 0.0) * math.log(n / df[feat]))
        F = csr_matrix((vals, (rows, cols)), shape=(len(self.slugs), len(vocab)), dtype=np.float64)
        norms = np.sqrt(np.asarray(F.multiply(F).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        return (diags(1.0 / norms) @ F).tocsr()

    @staticmethod
    def interaction_weights(user_films: list[dict]) -> dict[str, float]:
        ratings = [f["rating"] for f in user_films if f.get("rating") is not None]
        mean = sum(ratings) / len(ratings) if ratings else 0.0
        weights: dict[str, float] = {}
        for f in user_films:
            rating = f.get("rating")
            if rating is not None:
                w = rating - mean + (LIKED_BONUS if f.get("liked") else 0.0)
            elif f.get("liked"):
                w = UNRATED_WEIGHTS["liked"]
            elif f.get("watched"):
                w = UNRATED_WEIGHTS["watched"]
            elif f.get("watchlisted"):
                w = UNRATED_WEIGHTS["watchlisted"]
            else:
                continue
            if w:
                weights[f["slug"]] = w
        return weights

    def profile(self, user_films: list[dict]) -> np.ndarray:
        weights = self.interaction_weights(user_films)
        idx = [self.index[s] for s in weights if s in self.index]
        if not idx:
            return np.zeros(self.F.shape[1])
        w = np.array([weights[self.slugs[i]] for i in idx])
        return np.asarray(self.F[idx].T @ w).ravel()

    def score(self, user_films: list[dict]) -> np.ndarray:
        return self.F @ self.profile(user_films)

    def recommend(self, user_films: list[dict], n: int = 20, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        if not self.slugs:
            return []
        scores = self.score(user_films)
        for slug in {f["slug"] for f in user_films} | (exclude or set()):
            i = self.index.get(slug)
            if i is not None:
                scores[i] = -np.inf
        k = min(n, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self.slugs[i], float(scores[i])) for i in top if np.isfinite(scores[i]) and scores[i] > 0]

    def top_shared_features(self, user_films: list[dict], slug: str, n: int = 3) -> list[str]:
        """Human-readable reasons: the film's features that contribute most to its score."""
        i = self.index.get(slug)
        if i is None:
            return []
        contrib = self.F[i].multiply(self.profile(user_films)).tocoo()
        order = np.argsort(-contrib.data)[:n]
        reasons = []
        for j in order:
            if contrib.data[j] <= 0:
                break
            field, value = self.feature_names[contrib.col[j]]
            reasons.append(f"{FIELD_LABELS.get(field, field)}: {value}")
        return reasons
