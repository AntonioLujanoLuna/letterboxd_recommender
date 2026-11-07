"""
EASE trained on MovieLens, applied to a Letterboxd history.

MovieLens (GroupLens, University of Minnesota) publishes real users' film ratings for
research. ML-32M has 32M ratings from 200k users, collected up to October 2023, so
films released after that are not in it (use the `content` strategy for those).

Pipeline:
  1. build_model(): read ratings.csv once (cached as .npz), keep the most-rated films,
     fit EASE for a few regularization strengths, score each on held-out MovieLens
     users (recall/NDCG@k vs a popularity baseline), save the best.
  2. FilmMatcher: map Letterboxd films to MovieLens films by TMDB id, then IMDb id,
     then normalized title + year.
  3. MovieLensModel.recommend(): score a Letterboxd user's mapped history with EASE.

GroupLens terms: research/personal use, don't redistribute the data, acknowledge
the source (F. Maxwell Harper and Joseph A. Konstan, 2015, ACM TiiS).
"""

from __future__ import annotations

import csv
import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .ease import interaction_weight
from .evaluation import ndcg_at_k, recall_at_k

logger = logging.getLogger(__name__)

ML_DIR = Path("data/movielens/ml-32m")
ML_MODEL_PATH = Path("data/movielens_ease.npz")
ML_DOWNLOAD_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"


def _rating_weights(ratings: np.ndarray) -> np.ndarray:
    """Vectorized interaction_weight() for star ratings (same thresholds as EASE_WEIGHTS)."""
    from .config import EASE_WEIGHTS

    return np.select(
        [ratings >= 3.5, ratings >= 2.5],
        [EASE_WEIGHTS["rating_high"], EASE_WEIGHTS["rating_mid"]],
        default=EASE_WEIGHTS["rating_low"],
    ).astype(np.float32)


# --- Raw data --------------------------------------------------------------

def load_ratings(ml_dir: Path = ML_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(user_ids, movie_ids, ratings) from ratings.csv, cached as ratings.npz after the first read."""
    cache = ml_dir / "ratings.npz"
    if cache.exists():
        blob = np.load(cache)
        return blob["users"], blob["movies"], blob["ratings"]
    src = ml_dir / "ratings.csv"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Download {ML_DOWNLOAD_URL} and unzip it into {ml_dir.parent}, "
            "or run 'python main.py movielens-build --download'."
        )
    logger.info(f"Reading {src} (first run only; this takes a minute)...")
    start = time.perf_counter()
    raw = np.loadtxt(src, delimiter=",", skiprows=1, usecols=(0, 1, 2), dtype=np.float64)
    users = raw[:, 0].astype(np.int32)
    movies = raw[:, 1].astype(np.int32)
    ratings = raw[:, 2].astype(np.float32)
    del raw
    np.savez(cache, users=users, movies=movies, ratings=ratings)
    logger.info(f"Read {len(ratings):,} ratings in {time.perf_counter() - start:.0f}s (cached to {cache})")
    return users, movies, ratings


_YEAR_RE = re.compile(r"\s*\((\d{4})(?:[-–]\d{0,4})?\)\s*$")
_ARTICLE_RE = re.compile(r"^(.*), (The|A|An|Les|Le|La|L'|Il|Das|Der|Die|El|Los|Las)$", re.I)


def split_title_year(raw_title: str) -> tuple[str, int | None]:
    """'Matrix, The (1999)' -> ('The Matrix', 1999). Alternate titles in parentheses are dropped."""
    m = _YEAR_RE.search(raw_title)
    year = int(m.group(1)) if m else None
    title = _YEAR_RE.sub("", raw_title).strip()
    # Drop trailing alternate/original titles: 'Seven (a.k.a. Se7en)', 'City of God (Cidade de Deus)'
    while True:
        stripped = re.sub(r"\s*\([^()]*\)\s*$", "", title).strip()
        if not stripped or stripped == title:
            break
        title = stripped
    m = _ARTICLE_RE.match(title)
    if m:
        sep = "" if m.group(2).endswith("'") else " "
        title = f"{m.group(2)}{sep}{m.group(1)}"
    return title, year


def normalize_title(title: str) -> str:
    title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    title = title.lower().replace("&", "and")
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return re.sub(r"^(the|a|an) ", "", title.strip())


@dataclass
class MovieInfo:
    movie_id: int
    title: str
    year: int | None
    genres: list[str]
    tmdb_id: int | None = None
    imdb_id: str | None = None


def load_movies(ml_dir: Path = ML_DIR) -> dict[int, MovieInfo]:
    movies: dict[int, MovieInfo] = {}
    with open(ml_dir / "movies.csv", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            title, year = split_title_year(row["title"])
            genres = [] if row["genres"] == "(no genres listed)" else row["genres"].lower().split("|")
            movies[int(row["movieId"])] = MovieInfo(int(row["movieId"]), title, year, genres)
    links = ml_dir / "links.csv"
    if links.exists():
        with open(links, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                info = movies.get(int(row["movieId"]))
                if info is None:
                    continue
                if row.get("tmdbId"):
                    info.tmdb_id = int(row["tmdbId"])
                if row.get("imdbId"):
                    info.imdb_id = f"tt{int(row['imdbId']):07d}"
    return movies


# --- Model -----------------------------------------------------------------

@dataclass
class MovieLensModel:
    B: np.ndarray                      # items x items EASE weights (float32)
    movie_ids: np.ndarray              # column -> MovieLens movieId
    titles: list[str]
    years: list[int | None]
    tmdb_ids: list[int | None]
    imdb_ids: list[str | None]
    popularity: np.ndarray             # positive-interaction count per column
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.col_of_movie = {int(m): i for i, m in enumerate(self.movie_ids)}

    @property
    def n_items(self) -> int:
        return len(self.movie_ids)

    def save(self, path: Path = ML_MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            B=self.B,
            movie_ids=self.movie_ids,
            popularity=self.popularity,
            items=json.dumps({
                "titles": self.titles, "years": self.years,
                "tmdb_ids": self.tmdb_ids, "imdb_ids": self.imdb_ids,
            }),
            metadata=json.dumps(self.metadata),
        )
        logger.info(f"Saved MovieLens EASE model to {path} ({path.stat().st_size / 1e6:.0f} MB)")

    @classmethod
    def load(cls, path: Path = ML_MODEL_PATH) -> "MovieLensModel":
        if not path.exists():
            raise FileNotFoundError(
                f"No MovieLens model at {path}. Run 'python main.py movielens-build' first."
            )
        blob = np.load(path, allow_pickle=False)
        items = json.loads(str(blob["items"]))
        return cls(
            B=blob["B"], movie_ids=blob["movie_ids"], popularity=blob["popularity"],
            titles=items["titles"], years=items["years"],
            tmdb_ids=items["tmdb_ids"], imdb_ids=items["imdb_ids"],
            metadata=json.loads(str(blob["metadata"])),
        )

    def score_columns(self, weighted_cols: dict[int, float]) -> np.ndarray:
        x = np.zeros(self.n_items, dtype=np.float32)
        for col, w in weighted_cols.items():
            x[col] = w
        return x @ self.B

    def recommend_columns(self, weighted_cols: dict[int, float], seen_cols: set[int], n: int) -> list[tuple[int, float]]:
        if not weighted_cols:
            return []
        scores = self.score_columns(weighted_cols)
        if seen_cols:
            scores[list(seen_cols)] = -np.inf
        k = min(n, self.n_items)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(c), float(scores[c])) for c in top if np.isfinite(scores[c])]


def _fit_ease_from_gram(G: np.ndarray, lam: float) -> np.ndarray:
    A = G.copy()
    A[np.diag_indices_from(A)] += lam
    P = np.linalg.inv(A)
    B = P / (-np.diag(P))
    B[np.diag_indices_from(B)] = 0.0
    return B.astype(np.float32)


@dataclass
class BuildReport:
    n_ratings: int
    n_users: int
    n_items: int
    n_eval_users: int
    k: int
    results: list[dict]
    best_lambda: float


def build_model(
    ml_dir: Path = ML_DIR,
    max_items: int = 10_000,
    lambdas: tuple[float, ...] = (200.0, 500.0, 1000.0, 3000.0),
    eval_users: int = 2000,
    eval_min_positives: int = 10,
    holdout_frac: float = 0.2,
    k: int = 20,
    seed: int = 42,
    out_path: Path | None = ML_MODEL_PATH,
) -> tuple[MovieLensModel, BuildReport]:
    """
    Fit EASE on MovieLens and pick the regularization that scores best on held-out users.

    Held-out users keep 80% of their positives in training (their hidden 20% are removed),
    then are scored exactly the way a Letterboxd user will be: from their history alone.
    """
    users, movies, ratings = load_ratings(ml_dir)
    weights = _rating_weights(ratings)

    # Most-interacted films (positive interactions only)
    pos = weights > 0
    uniq_movies, pos_counts = np.unique(movies[pos], return_counts=True)
    keep_movies = uniq_movies[np.argsort(-pos_counts, kind="stable")[:max_items]]
    keep_movies.sort()
    col_of = np.full(int(movies.max()) + 1, -1, dtype=np.int64)
    col_of[keep_movies] = np.arange(len(keep_movies))

    mask = pos & (col_of[movies] >= 0)
    u = users[mask]
    c = col_of[movies[mask]]
    w = weights[mask]
    uniq_users, row = np.unique(u, return_inverse=True)
    n_users, n_items = len(uniq_users), len(keep_movies)
    logger.info(f"MovieLens: {len(ratings):,} ratings -> {mask.sum():,} positive interactions, "
                f"{n_users:,} users x {n_items:,} films")

    # Held-out evaluation users (enough positives to hide some)
    rng = np.random.default_rng(seed)
    per_user = np.bincount(row, minlength=n_users)
    candidates = np.flatnonzero(per_user >= eval_min_positives)
    if candidates.size == 0:
        raise ValueError(f"No MovieLens users with >= {eval_min_positives} liked films to evaluate on.")
    chosen = rng.choice(candidates, size=min(eval_users, len(candidates)), replace=False)
    is_eval = np.zeros(n_users, dtype=bool)
    is_eval[chosen] = True

    hidden = np.zeros(len(row), dtype=bool)
    eval_rows = np.flatnonzero(is_eval[row])
    order = np.argsort(row[eval_rows], kind="stable")
    eval_rows = eval_rows[order]
    bounds = np.flatnonzero(np.diff(row[eval_rows])) + 1
    for group in np.split(eval_rows, bounds):
        n_hide = max(1, int(round(len(group) * holdout_frac)))
        hidden[rng.choice(group, size=n_hide, replace=False)] = True

    train = ~hidden
    X = csr_matrix((w[train], (row[train], c[train])), shape=(n_users, n_items), dtype=np.float64)
    popularity = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.int64)

    logger.info("Computing item-item Gram matrix...")
    G = (X.T @ X).toarray()

    # Evaluation data
    eval_users_idx = np.flatnonzero(is_eval)
    X_eval = X[eval_users_idx]
    hidden_sets: dict[int, set[int]] = {}
    for r_, c_ in zip(row[hidden], c[hidden]):
        hidden_sets.setdefault(int(r_), set()).add(int(c_))

    def evaluate(score_fn) -> tuple[float, float]:
        recalls, ndcgs = [], []
        for i, uidx in enumerate(eval_users_idx):
            x = X_eval[i]
            scores = score_fn(x)
            scores[x.indices] = -np.inf
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top], kind="stable")].tolist()
            rel = hidden_sets.get(int(uidx), set())
            recalls.append(recall_at_k(top, rel, k))
            ndcgs.append(ndcg_at_k(top, rel, k))
        return float(np.mean(recalls)), float(np.mean(ndcgs))

    results = []
    pop_scores = popularity.astype(np.float64)
    r, n = evaluate(lambda x: pop_scores.copy())
    results.append({"model": "popularity", "lambda": None, f"recall@{k}": r, f"ndcg@{k}": n, "fit_seconds": 0.0})
    logger.info(f"  popularity: recall@{k}={r:.4f} ndcg@{k}={n:.4f}")

    best_B, best_lam, best_ndcg = None, None, -1.0
    for lam in lambdas:
        start = time.perf_counter()
        B = _fit_ease_from_gram(G, lam)
        fit_s = time.perf_counter() - start
        r, n = evaluate(lambda x: np.asarray(x @ B).ravel().astype(np.float64))
        results.append({"model": "ease", "lambda": lam, f"recall@{k}": r, f"ndcg@{k}": n, "fit_seconds": fit_s})
        logger.info(f"  EASE lambda={lam:g}: recall@{k}={r:.4f} ndcg@{k}={n:.4f} ({fit_s:.0f}s)")
        if n > best_ndcg:
            best_B, best_lam, best_ndcg = B, lam, n
        else:
            del B
    del G

    info = load_movies(ml_dir)
    cols = [info.get(int(m)) for m in keep_movies]
    model = MovieLensModel(
        B=best_B,
        movie_ids=keep_movies.astype(np.int32),
        titles=[ci.title if ci else str(m) for ci, m in zip(cols, keep_movies)],
        years=[ci.year if ci else None for ci in cols],
        tmdb_ids=[ci.tmdb_id if ci else None for ci in cols],
        imdb_ids=[ci.imdb_id if ci else None for ci in cols],
        popularity=popularity,
        metadata={
            "source": str(ml_dir),
            "lambda": best_lam,
            "n_users": int(n_users),
            "n_items": int(n_items),
            "n_ratings": int(len(ratings)),
            "evaluation": results,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    if out_path is not None:
        model.save(out_path)
    report = BuildReport(
        n_ratings=int(len(ratings)), n_users=int(n_users), n_items=int(n_items),
        n_eval_users=int(len(eval_users_idx)), k=k, results=results, best_lambda=float(best_lam),
    )
    return model, report


# --- Letterboxd <-> MovieLens ----------------------------------------------

class FilmMatcher:
    """Map Letterboxd films (rows of the `films` table) to model columns."""

    def __init__(self, model: MovieLensModel):
        self.model = model
        self.by_tmdb = {t: i for i, t in enumerate(model.tmdb_ids) if t}
        self.by_imdb = {t: i for i, t in enumerate(model.imdb_ids) if t}
        self.by_title: dict[tuple[str, int | None], list[int]] = {}
        for i, (title, year) in enumerate(zip(model.titles, model.years)):
            self.by_title.setdefault((normalize_title(title), year), []).append(i)

    def column_for(self, film: dict) -> int | None:
        tmdb = film.get("tmdb_id")
        if tmdb and int(tmdb) in self.by_tmdb:
            return self.by_tmdb[int(tmdb)]
        imdb = film.get("imdb_id")
        if imdb and imdb in self.by_imdb:
            return self.by_imdb[imdb]
        title = film.get("title")
        if not title:
            return None
        key = normalize_title(title)
        year = film.get("year")
        for y in ((year, year - 1, year + 1) if year else (None,)):
            hits = self.by_title.get((key, y))
            if hits and len(hits) == 1:  # ambiguous title+year: don't guess
                return hits[0]
        return None

    def map_history(self, user_films: list[dict], films: dict[str, dict]) -> tuple[dict[int, float], set[int], int]:
        """(weighted columns, all seen columns, number of unmatched films) for a history."""
        weighted: dict[int, float] = {}
        seen: set[int] = set()
        unmatched = 0
        for f in user_films:
            meta = films.get(f["slug"])
            col = self.column_for(meta) if meta else None
            if col is None:
                unmatched += 1
                continue
            seen.add(col)
            weight = interaction_weight(f)
            if weight > 0:
                weighted[col] = max(weighted.get(col, 0.0), weight)
        return weighted, seen, unmatched

    def letterboxd_slugs(self, films: dict[str, dict]) -> dict[int, str]:
        """Model column -> Letterboxd slug for films already in the local database."""
        out: dict[int, str] = {}
        for slug, film in films.items():
            col = self.column_for(film)
            if col is not None:
                out.setdefault(col, slug)
        return out
