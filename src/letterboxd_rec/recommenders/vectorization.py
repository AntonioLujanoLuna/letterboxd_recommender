"""TF-IDF embedder for metadata-based cold-start similarity."""

import math

from ..database import load_json
from ..config import TFIDF_FIELDS, TFIDF_MIN_DF, TFIDF_MAX_FEATURES


class TfidfEmbedder:
    """
    Lightweight TF-IDF embedder for metadata-based cold-start similarity.

    Keeps everything in simple Python dicts to avoid heavy dependencies.
    """

    def __init__(self, films: dict[str, dict]):
        self.films = films
        self.idf: dict[str, float] = {}
        self.vectors: dict[str, dict[str, float]] = {}
        self._build()

    def _tokenize(self, film: dict) -> list[str]:
        tokens = []
        parsed = film.get('_parsed', {})
        for field in TFIDF_FIELDS:
            values = parsed.get(field)
            if values is None:
                values = load_json(film.get(field, []))
            tokens.extend([str(v).lower() for v in (values or [])])
        return tokens

    def _build(self):
        df_counts = {}
        corpus = {}
        for slug, film in self.films.items():
            tokens = self._tokenize(film)
            corpus[slug] = tokens
            unique_tokens = set(tokens)
            for tok in unique_tokens:
                df_counts[tok] = df_counts.get(tok, 0) + 1

        n_docs = max(1, len(corpus))
        # Filter rare and extremely common tokens
        for tok, df in df_counts.items():
            if df < TFIDF_MIN_DF:
                continue
            idf = math.log((n_docs + 1) / (df + 1)) + 1
            self.idf[tok] = idf

        # Limit feature size by top IDF weights to avoid memory blow-up
        if len(self.idf) > TFIDF_MAX_FEATURES:
            # Keep top features
            top_tokens = sorted(self.idf.items(), key=lambda x: -x[1])[:TFIDF_MAX_FEATURES]
            self.idf = dict(top_tokens)

        for slug, tokens in corpus.items():
            tf = {}
            for tok in tokens:
                if tok not in self.idf:
                    continue
                tf[tok] = tf.get(tok, 0) + 1
            if not tf:
                continue
            vec = {tok: (count / len(tokens)) * self.idf[tok] for tok, count in tf.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.vectors[slug] = {tok: val / norm for tok, val in vec.items()}

    def similarity(self, slug_a: str, slug_b: str) -> float:
        va = self.vectors.get(slug_a)
        vb = self.vectors.get(slug_b)
        if not va or not vb:
            return 0.0
        # Cosine on sparse dicts
        if len(va) > len(vb):
            va, vb = vb, va
        return sum(weight * vb.get(tok, 0.0) for tok, weight in va.items())

    def rank_against(self, slug: str, candidates: list[str], top_k: int = 50) -> list[tuple[str, float]]:
        scores = []
        for other in candidates:
            if other == slug:
                continue
            sim = self.similarity(slug, other)
            if sim > 0:
                scores.append((other, sim))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def score_to_centroid(self, anchor_slugs: list[str], candidates: list[str], top_k: int = 50) -> list[tuple[str, float]]:
        if not anchor_slugs:
            return []
        centroid = {}
        for slug in anchor_slugs:
            vec = self.vectors.get(slug)
            if not vec:
                continue
            for tok, weight in vec.items():
                centroid[tok] = centroid.get(tok, 0.0) + weight
        if not centroid:
            return []
        norm = math.sqrt(sum(v * v for v in centroid.values())) or 1.0
        centroid = {k: v / norm for k, v in centroid.items()}

        scores = []
        for slug in candidates:
            vec = self.vectors.get(slug)
            if not vec:
                continue
            if len(vec) > len(centroid):
                vec, centroid = centroid, vec  # swap for faster loop
            sim = sum(weight * centroid.get(tok, 0.0) for tok, weight in vec.items())
            if sim > 0:
                scores.append((slug, sim))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
