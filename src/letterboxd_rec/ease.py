"""
EASE: Embarrassingly Shallow Autoencoder (Steck, 2019).

A linear item-item model with a closed-form solution:

    G = X^T X + lambda * I
    P = G^-1
    B = -P / diag(P),  diag(B) = 0

and a user's scores are x_u @ B. One matrix inverse, no iterations, and on implicit
feedback benchmarks it is usually as good as or better than kNN and matrix factorization.

Because scoring only needs the user's interaction vector, any user can be scored
without refitting, including users who are not in the training data.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .config import (
    EASE_CACHE_PATH,
    EASE_LAMBDA,
    EASE_MAX_ITEMS,
    EASE_MIN_INTERACTIONS,
    EASE_WEIGHTS,
)
from .data_version import interactions_digest

logger = logging.getLogger(__name__)


def interaction_weight(f: dict) -> float:
    """Strength of a user-film interaction as an EASE input (0 means ignore)."""
    weight = 0.0
    rating = f.get("rating")
    if rating is not None:
        if rating >= 3.5:
            weight = EASE_WEIGHTS["rating_high"]
        elif rating >= 2.5:
            weight = EASE_WEIGHTS["rating_mid"]
        else:
            weight = EASE_WEIGHTS["rating_low"]
    elif f.get("watched"):
        weight = EASE_WEIGHTS["watched"]
    elif f.get("watchlisted"):
        weight = EASE_WEIGHTS["watchlisted"]
    if f.get("liked"):
        weight = max(weight, EASE_WEIGHTS["liked"])
    return weight


class EASERecommender:
    def __init__(
        self,
        lam: float = EASE_LAMBDA,
        min_interactions: int = EASE_MIN_INTERACTIONS,
        max_items: int = EASE_MAX_ITEMS,
    ):
        self.lam = lam
        self.min_interactions = min_interactions
        self.max_items = max_items
        self.B: np.ndarray | None = None
        self.item_slugs: list[str] = []
        self.item_index: dict[str, int] = {}
        self.metadata: dict = {}
        self.item_counts: np.ndarray | None = None  # users who interacted with each film

    @property
    def is_fitted(self) -> bool:
        return self.B is not None

    def hyperparams(self) -> dict:
        return {
            "lambda": self.lam,
            "min_interactions": self.min_interactions,
            "max_items": self.max_items,
            "weights": EASE_WEIGHTS,
            "format": 2,  # v2 caches also hold item_counts (needed for novelty)
        }

    def fingerprint(self, all_user_films: dict[str, list[dict]]) -> dict:
        return {"data_digest": interactions_digest(all_user_films), "hyperparams": self.hyperparams()}

    def fit(self, all_user_films: dict[str, list[dict]]) -> "EASERecommender":
        # Keep films with enough (positively weighted) interactions, most-interacted first
        counts: dict[str, int] = {}
        for films in all_user_films.values():
            for f in films:
                if interaction_weight(f) > 0:
                    counts[f["slug"]] = counts.get(f["slug"], 0) + 1
        eligible = sorted(
            (s for s, c in counts.items() if c >= self.min_interactions),
            key=lambda s: (-counts[s], s),
        )[: self.max_items]
        if len(eligible) < 2:
            raise ValueError(
                f"EASE needs at least 2 films with >= {self.min_interactions} interactions "
                f"(found {len(eligible)}). Add more users or lower EASE_MIN_INTERACTIONS."
            )

        self.item_slugs = eligible
        self.item_index = {s: i for i, s in enumerate(eligible)}
        X = self._user_matrix(all_user_films.values())

        G = (X.T @ X).toarray().astype(np.float64)
        G[np.diag_indices_from(G)] += self.lam
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[np.diag_indices_from(B)] = 0.0
        self.B = B.astype(np.float32)
        self.item_counts = np.asarray(np.diff((X > 0).tocsc().indptr), dtype=np.float32)

        self.metadata = {
            "fingerprint": self.fingerprint(all_user_films),
            "n_users": len(all_user_films),
            "n_items": len(eligible),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(f"Fitted EASE on {len(all_user_films)} users x {len(eligible)} films (lambda={self.lam})")
        return self

    def _user_matrix(self, users_films) -> csr_matrix:
        users_films = list(users_films)
        rows, cols, vals = [], [], []
        for r, films in enumerate(users_films):
            for f in films:
                idx = self.item_index.get(f["slug"])
                if idx is None:
                    continue
                w = interaction_weight(f)
                if w > 0:
                    rows.append(r)
                    cols.append(idx)
                    vals.append(w)
        return csr_matrix(
            (np.asarray(vals, dtype=np.float64), (rows, cols)),
            shape=(len(users_films), len(self.item_slugs)),
        )

    def score(self, user_films: list[dict]) -> np.ndarray:
        """Scores for every model film (seen films are not masked here)."""
        if self.B is None:
            raise RuntimeError("EASE model is not fitted")
        x = np.zeros(len(self.item_slugs), dtype=np.float32)
        for f in user_films:
            idx = self.item_index.get(f["slug"])
            if idx is not None:
                x[idx] = interaction_weight(f)
        return x @ self.B

    def recommend(
        self,
        user_films: list[dict],
        n: int = 20,
        exclude: set[str] | None = None,
        novelty: float = 0.0,
        allowed: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Top-n (slug, score), skipping everything in the user's history and `exclude`.

        `allowed`, when given, restricts the ranking to those slugs *before* the top-n is taken, so a
        narrow filter (one actor, one country and decade) still fills the list instead of being
        applied to a fixed shortlist that may contain none of them.

        `novelty` (0..1) divides each positive score by popularity**novelty, trading accuracy for
        less mainstream picks. Offline (150 users): 0.3 cost ~1 point of recall@20 for ~10x the
        recall on films outside the 1000 most-watched; 0.5 cost ~8 points for 25% niche picks.
        """
        if self.B is None:
            return []
        scores = self.score(user_films)
        if novelty > 0 and self.item_counts is not None:
            damp = np.maximum(self.item_counts, 1.0) ** novelty
            scores = np.where(scores > 0, scores / damp, scores).astype(np.float32)
        blocked = {f["slug"] for f in user_films} | (exclude or set())
        for slug in blocked:
            idx = self.item_index.get(slug)
            if idx is not None:
                scores[idx] = -np.inf
        if allowed is not None:
            keep = np.zeros(len(scores), dtype=bool)
            for slug in allowed:
                idx = self.item_index.get(slug)
                if idx is not None:
                    keep[idx] = True
            scores[~keep] = -np.inf
        k = min(n, len(scores))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self.item_slugs[i], float(scores[i])) for i in top if np.isfinite(scores[i])]

    # --- persistence -------------------------------------------------------

    def save(self, path: str | Path = EASE_CACHE_PATH) -> None:
        if self.B is None:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, B=self.B, item_slugs=np.array(self.item_slugs), metadata=json.dumps(self.metadata),
                 item_counts=self.item_counts)
        logger.info(f"Saved EASE model to {path}")

    @classmethod
    def load(cls, path: str | Path = EASE_CACHE_PATH, expected_fingerprint: dict | None = None) -> "EASERecommender | None":
        path = Path(path)
        if not path.exists():
            return None
        try:
            blob = np.load(path, allow_pickle=False)
            metadata = json.loads(str(blob["metadata"]))
            if expected_fingerprint is not None and metadata.get("fingerprint") != expected_fingerprint:
                logger.info("Cached EASE model is out of date; refitting.")
                return None
            hp = metadata.get("fingerprint", {}).get("hyperparams", {})
            inst = cls(
                lam=hp.get("lambda", EASE_LAMBDA),
                min_interactions=hp.get("min_interactions", EASE_MIN_INTERACTIONS),
                max_items=hp.get("max_items", EASE_MAX_ITEMS),
            )
            inst.B = blob["B"]
            inst.item_counts = blob["item_counts"]
            inst.item_slugs = [str(s) for s in blob["item_slugs"]]
            inst.item_index = {s: i for i, s in enumerate(inst.item_slugs)}
            inst.metadata = metadata
            return inst
        except Exception as exc:
            logger.warning(f"Failed to load EASE cache {path}: {exc}")
            return None


def load_or_fit_ease(all_user_films: dict[str, list[dict]], cache_path: str | Path | None = EASE_CACHE_PATH) -> EASERecommender:
    """Load a cached model if it was fitted on exactly this data, else fit (and cache)."""
    model = EASERecommender()
    if cache_path is not None:
        cached = EASERecommender.load(cache_path, expected_fingerprint=model.fingerprint(all_user_films))
        if cached is not None:
            logger.info("Using cached EASE model.")
            return cached
    model.fit(all_user_films)
    if cache_path is not None:
        model.save(cache_path)
    return model
