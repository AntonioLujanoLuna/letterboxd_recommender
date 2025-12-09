from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
from scipy import sparse

from .database import get_db, load_idf, load_json, parse_timestamp_naive
from .graph_config import GraphConfig
from .recommender import Recommendation

logger = logging.getLogger(__name__)


class NodeType(Enum):
    USER = "user"
    FILM = "film"
    DIRECTOR = "director"
    GENRE = "genre"
    ACTOR = "actor"
    THEME = "theme"
    COUNTRY = "country"
    LANGUAGE = "language"
    WRITER = "writer"
    CINEMATOGRAPHER = "cinematographer"
    COMPOSER = "composer"
    DECADE = "decade"


@dataclass
class GraphEdge:
    src: int
    dst: int
    edge_type: str
    weight: float


class HeterogeneousGraph:
    """
    Typed, weighted graph with column-stochastic transition matrix.
    """

    def __init__(self, config: GraphConfig):
        self.config = config
        self.node_to_idx: dict[tuple[NodeType, str], int] = {}
        self.idx_to_node: list[tuple[NodeType, str]] = []
        self.edges: list[GraphEdge] = []
        self.transition: sparse.csr_matrix | None = None
        self.dangling_mask: np.ndarray | None = None
        self.film_metadata: dict[str, dict] = {}
        self.slug_to_idx: dict[str, int] = {}
        self.idx_to_slug: dict[int, str] = {}

    # Node helpers -----------------------------------------------------
    def add_node(self, node_type: NodeType, node_id: str, metadata: dict | None = None) -> int:
        key = (node_type, node_id)
        if key in self.node_to_idx:
            return self.node_to_idx[key]

        idx = len(self.idx_to_node)
        self.node_to_idx[key] = idx
        self.idx_to_node.append(key)

        if node_type == NodeType.FILM:
            self.slug_to_idx[node_id] = idx
            self.idx_to_slug[idx] = node_id
            if metadata is not None:
                self.film_metadata[node_id] = metadata

        return idx

    def get_idx(self, node_type: NodeType, node_id: str) -> int | None:
        return self.node_to_idx.get((node_type, node_id))

    # Edge helpers -----------------------------------------------------
    def add_edge(self, src_idx: int, dst_idx: int, edge_type: str, weight: float) -> None:
        if weight <= 0:
            return
        self.edges.append(GraphEdge(src_idx, dst_idx, edge_type, weight))

    def build_transition_matrix(self) -> sparse.csr_matrix:
        """Column-stochastic transition matrix with optional type budgets."""
        num_nodes = len(self.idx_to_node)
        outgoing: list[list[GraphEdge]] = [[] for _ in range(num_nodes)]
        for e in self.edges:
            outgoing[e.src].append(e)

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        dangling = np.zeros(num_nodes, dtype=bool)

        for src_idx, edges in enumerate(outgoing):
            if not edges:
                dangling[src_idx] = True
                continue

            if self.config.use_type_budgets:
                # Legacy behavior: allocate probability mass per edge type budget
                edges_by_type: dict[str, list[GraphEdge]] = defaultdict(list)
                for e in edges:
                    edges_by_type[e.edge_type].append(e)

                type_budgets = {
                    etype: self.config.edge_type_weights.get(etype, 1.0)
                    for etype in edges_by_type
                }
                total_type_weight = sum(type_budgets.values()) or 1.0

                for etype, etype_edges in edges_by_type.items():
                    budget = type_budgets[etype] / total_type_weight
                    total_weight = sum(self._relation_weighted(e) for e in etype_edges) or 1.0
                    for e in etype_edges:
                        prob = budget * (self._relation_weighted(e) / total_weight)
                        rows.append(e.dst)
                        cols.append(src_idx)
                        data.append(prob)
            else:
                # New default: normalize relation-weighted edges directly
                total_weight = sum(self._relation_weighted(e) for e in edges) or 1.0
                for e in edges:
                    prob = self._relation_weighted(e) / total_weight
                    rows.append(e.dst)
                    cols.append(src_idx)
                    data.append(prob)

        self.transition = sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
        self.dangling_mask = dangling
        return self.transition

    def _relation_weighted(self, edge: GraphEdge) -> float:
        """Apply relation-type multiplier to a raw edge weight."""
        rel_weight = self.config.relation_weights.get(edge.edge_type, 1.0)
        return rel_weight * edge.weight

    # Cache helpers ----------------------------------------------------
    def save(self, path: Path) -> None:
        if self.transition is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        coo = self.transition.tocoo()
        payload = {
            "rows": coo.row,
            "cols": coo.col,
            "data": coo.data,
            "shape": coo.shape,
            "idx_to_node": json.dumps([(t.value, v) for t, v in self.idx_to_node]),
            "slug_to_idx": json.dumps(self.slug_to_idx),
            "film_metadata": json.dumps(self.film_metadata),
        }
        if self.dangling_mask is not None:
            payload["dangling"] = self.dangling_mask.astype(np.int8)
        payload["meta"] = json.dumps(
            {
                "config_key": getattr(self.config, "cache_key", None),
                "cache_version": getattr(self.config, "cache_version", None),
                "created_at": time.time(),
            }
        )
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path, config: GraphConfig) -> "HeterogeneousGraph | None":
        if not path.exists():
            return None
        try:
            blob = np.load(path, allow_pickle=True)
            graph = cls(config)
            meta_raw = blob["meta"] if "meta" in blob else "{}"
            meta = json.loads(str(meta_raw))
            cached_key = meta.get("config_key")
            cached_version = meta.get("cache_version")
            if cached_key is None or cached_version is None:
                logger.info("Graph cache %s missing metadata; rebuilding.", path)
                return None
            if cached_key != config.cache_key or cached_version != config.cache_version:
                logger.info(
                    "Graph cache %s does not match config (have %s/%s, want %s/%s); rebuilding.",
                    path,
                    cached_version,
                    cached_key,
                    config.cache_version,
                    config.cache_key,
                )
                return None
            rows = blob["rows"]
            cols = blob["cols"]
            data = blob["data"]
            shape = tuple(blob["shape"])
            graph.transition = sparse.csr_matrix((data, (rows, cols)), shape=shape)
            graph.idx_to_node = [
                (NodeType(t), v) for t, v in json.loads(str(blob["idx_to_node"]))
            ]
            graph.node_to_idx = {pair: idx for idx, pair in enumerate(graph.idx_to_node)}
            graph.slug_to_idx = {k: int(v) for k, v in json.loads(str(blob["slug_to_idx"])).items()}
            graph.idx_to_slug = {idx: slug for slug, idx in graph.slug_to_idx.items()}
            graph.film_metadata = json.loads(str(blob["film_metadata"]))
            if "dangling" in blob:
                graph.dangling_mask = blob["dangling"].astype(bool)
            return graph
        except Exception as exc:  # pragma: no cover - cache is best-effort
            logger.warning(f"Failed to load graph cache {path}: {exc}")
            return None


class PPREngine:
    """Plain iterative Personalized PageRank."""

    @staticmethod
    def compute(
        matrix: sparse.csr_matrix,
        restart: np.ndarray,
        dangling_mask: np.ndarray,
        alpha: float,
        max_iter: int,
        tol: float,
    ) -> np.ndarray:
        pi = restart.copy()
        for _ in range(max_iter):
            walk = (1.0 - alpha) * (matrix @ pi)
            if dangling_mask is not None and dangling_mask.any():
                dangling_mass = (1.0 - alpha) * pi[dangling_mask].sum()
                walk += dangling_mass * restart
            pi_new = alpha * restart + walk
            if np.linalg.norm(pi_new - pi, 1) < tol:
                return pi_new
            pi = pi_new
        return pi


class PathExplainer:
    """
    Small BFS-based explainer to surface short paths.

    Lightweight by design; we cap path length to keep runtime predictable.
    """

    def __init__(self, graph: HeterogeneousGraph, max_length: int = 4, max_paths: int = 3):
        self.graph = graph
        self.max_length = max_length
        self.max_paths = max_paths
        # Build adjacency with edge types for quick BFS
        self._adjacency: list[list[tuple[int, str]]] = [[] for _ in range(len(graph.idx_to_node))]
        self._edge_lookup: dict[tuple[int, int], str] = {}
        if graph.edges:
            for edge in graph.edges:
                self._adjacency[edge.src].append((edge.dst, edge.edge_type))
                self._edge_lookup[(edge.src, edge.dst)] = edge.edge_type
        elif graph.transition is not None:
            coo = graph.transition.tocoo()
            for dst, src in zip(coo.row, coo.col):
                self._adjacency[src].append((dst, "transition"))
                self._edge_lookup[(src, dst)] = "transition"

    def find_paths(self, src_idx: int, dst_idx: int) -> list[list[int]]:
        paths: list[list[int]] = []
        queue: deque[tuple[int, list[int]]] = deque([(src_idx, [src_idx])])
        while queue and len(paths) < self.max_paths:
            node, path = queue.popleft()
            if len(path) > self.max_length:
                continue
            for neighbor, _edge_type in self._adjacency[node]:
                if neighbor in path:
                    continue
                new_path = path + [neighbor]
                if neighbor == dst_idx:
                    paths.append(new_path)
                    if len(paths) >= self.max_paths:
                        break
                else:
                    queue.append((neighbor, new_path))
        return paths

    def format_path(self, path: list[int]) -> str:
        parts = []
        for i, idx in enumerate(path):
            node_type, node_id = self.graph.idx_to_node[idx]
            if node_type == NodeType.FILM:
                film = self.graph.film_metadata.get(node_id, {})
                title = film.get("title", node_id)
                parts.append(title)
            else:
                parts.append(f"{node_type.value}: {node_id}")
            if i < len(path) - 1:
                edge_type = self._edge_lookup.get((idx, path[i + 1]), "")
                if edge_type:
                    parts.append(f"({edge_type})")
        return " → ".join(parts)


class GraphRecommender:
    """
    Heterogeneous graph recommender using Personalized PageRank.
    """

    def __init__(self, conn=None, config: GraphConfig | None = None, rebuild: bool = False):
        self.config = config or GraphConfig()
        self.config.validate()
        self.conn = conn
        self.graph = self._load_or_build_graph(rebuild=rebuild)
        self.engine = PPREngine()
        self.explainer = PathExplainer(self.graph, self.config.max_path_length, self.config.max_explanation_paths)

    # Public API -------------------------------------------------------
    def recommend(
        self,
        username: str,
        n: int = 20,
        min_year: int | None = None,
        max_year: int | None = None,
        genres: Sequence[str] | None = None,
        exclude_genres: Sequence[str] | None = None,
        min_rating: float | None = None,
        explain: bool = False,
        recent_only: bool = False,
        seed_slugs: Sequence[str] | None = None,
    ) -> list[Recommendation]:
        user_films = self._load_user_films(username)
        if not user_films:
            logger.error(f"No data for '{username}'.")
            return []

        if self._should_fallback(user_films):
            logger.warning(
                f"User '{username}' has insufficient history for graph strategy; "
                "consider metadata strategy instead."
            )

        restart = self._build_restart_distribution(
            user_films,
            username=username,
            recent_only=recent_only,
            seed_slugs=seed_slugs,
        )
        if restart.sum() == 0:
            logger.error("Restart distribution is empty; cannot run graph PageRank.")
            return []

        scores = self.engine.compute(
            self.graph.transition,
            restart,
            self.graph.dangling_mask,
            self.config.alpha,
            self.config.max_iterations,
            self.config.tolerance,
        )

        seen = {f["slug"] for f in user_films if f.get("slug")}
        user_pref_attrs = self._aggregate_user_attributes(user_films)
        recs: list[Recommendation] = []

        for slug, film in self.graph.film_metadata.items():
            idx = self.graph.slug_to_idx.get(slug)
            if idx is None:
                continue
            if slug in seen:
                continue
            if not self._passes_filters(film, min_year, max_year, min_rating, genres, exclude_genres):
                continue

            score = float(scores[idx])
            if score <= 0:
                continue

            reasons = self._reason_from_attributes(film, user_pref_attrs)
            if not reasons:
                reasons = ["High graph relevance from your likes/watchlist"]
            recs.append(
                Recommendation(
                    slug=slug,
                    title=film.get("title", slug),
                    year=film.get("year"),
                    score=score,
                    reasons=reasons,
                )
            )

        recs.sort(key=lambda r: r.score, reverse=True)
        if explain:
            recs = self._attach_paths(recs[: n * 2], user_films)  # more options for path search
        return recs[:n]

    # Internal helpers -------------------------------------------------
    def _load_or_build_graph(self, rebuild: bool) -> HeterogeneousGraph:
        cache_path = self.config.cache_file_path
        if self.config.cache_enabled and not rebuild:
            cached = HeterogeneousGraph.load(cache_path, self.config)
            if cached:
                logger.info(f"Loaded graph cache from {cache_path}")
                return cached

        if self.conn is not None:
            graph = self._build_graph(self.conn)
        else:
            with get_db(read_only=True) as conn:
                graph = self._build_graph(conn)
        graph.build_transition_matrix()

        if self.config.cache_enabled:
            graph.save(cache_path)
            logger.info(f"Saved graph cache to {cache_path}")
        return graph

    def _build_graph(self, conn) -> HeterogeneousGraph:
        graph = HeterogeneousGraph(self.config)
        idf = load_idf() if self.config.use_idf else {}

        films = conn.execute("SELECT * FROM films").fetchall()
        for row in films:
            film = dict(row)
            slug = film["slug"]
            graph.add_node(NodeType.FILM, slug, metadata=film)
            # Decade node (optional)
            year = film.get("year")
            if year:
                decade = int(year) // 10 * 10
                dec_idx = graph.add_node(NodeType.DECADE, str(decade))
                film_idx = graph.get_idx(NodeType.FILM, slug)
                if film_idx is not None:
                    graph.add_edge(film_idx, dec_idx, "decade", 1.0)
                    graph.add_edge(dec_idx, film_idx, "decade", 1.0)

        # Film-attribute edges (normalized tables)
        self._add_attribute_edges(
            conn,
            graph,
            query="SELECT film_slug, director FROM film_directors",
            attr_type=NodeType.DIRECTOR,
            idf_key="director",
            edge_type="director",
            idf=idf,
        )
        self._add_attribute_edges(
            conn,
            graph,
            query="SELECT film_slug, genre FROM film_genres",
            attr_type=NodeType.GENRE,
            idf_key="genre",
            edge_type="genre",
            idf=idf,
        )
        self._add_attribute_edges(
            conn,
            graph,
            query="SELECT film_slug, actor FROM film_cast",
            attr_type=NodeType.ACTOR,
            idf_key="actor",
            edge_type="actor",
            idf=idf,
        )
        self._add_attribute_edges(
            conn,
            graph,
            query="SELECT film_slug, theme FROM film_themes",
            attr_type=NodeType.THEME,
            idf_key="theme",
            edge_type="theme",
            idf=idf,
        )

        # JSON-backed attributes
        self._add_json_attribute(conn, graph, "writers", NodeType.WRITER, "writer", idf)
        self._add_json_attribute(conn, graph, "cinematographers", NodeType.CINEMATOGRAPHER, "cinematographer", idf)
        self._add_json_attribute(conn, graph, "composers", NodeType.COMPOSER, "composer", idf)
        self._add_json_attribute(conn, graph, "countries", NodeType.COUNTRY, "country", idf)
        self._add_json_attribute(conn, graph, "languages", NodeType.LANGUAGE, "language", idf)

        # Users and interactions
        user_rows = conn.execute(
            "SELECT username, film_slug, rating, watched, watchlisted, liked, scraped_at FROM user_films"
        ).fetchall()
        for row in user_rows:
            username = row["username"]
            film_slug = row["film_slug"]
            film_idx = graph.slug_to_idx.get(film_slug)
            if film_idx is None:
                continue
            user_idx = graph.add_node(NodeType.USER, username)
            edge_type = self._classify_interaction(row)
            base_weight = self.config.interaction_weights.get(edge_type, self.config.restart_weights.get(edge_type, 1.0))
            decay = self._temporal_decay(row.get("scraped_at"))
            weight = base_weight * decay if self.config.use_temporal_decay else base_weight
            graph.add_edge(user_idx, film_idx, edge_type, weight)
            # Allow walks back to users for collaboration
            bridge_weight = self.config.interaction_weights.get("user_bridge", weight)
            graph.add_edge(film_idx, user_idx, "user_bridge", bridge_weight)

        if self.config.use_item_similarity_edges:
            self._add_item_similarity_edges(graph)

        if self.config.use_follow_edges:
            self._add_follow_edges(conn, graph)

        return graph

    def _add_attribute_edges(
        self,
        conn,
        graph: HeterogeneousGraph,
        query: str,
        attr_type: NodeType,
        idf_key: str,
        edge_type: str,
        idf: dict,
    ) -> None:
        rows = conn.execute(query).fetchall()
        base_weight = self.config.edge_type_weights.get(edge_type, 1.0)
        for row in rows:
            slug = row[0]
            value = row[1]
            film_idx = graph.slug_to_idx.get(slug)
            if film_idx is None or not value:
                continue
            attr_idx = graph.add_node(attr_type, value)
            weight = base_weight * self._idf_multiplier(idf, idf_key, value)
            graph.add_edge(film_idx, attr_idx, edge_type, weight)
            graph.add_edge(attr_idx, film_idx, edge_type, weight)

    def _add_json_attribute(
        self,
        conn,
        graph: HeterogeneousGraph,
        field: str,
        attr_type: NodeType,
        idf_key: str,
        idf: dict,
    ) -> None:
        rows = conn.execute(f"SELECT slug, {field} FROM films WHERE {field} IS NOT NULL").fetchall()
        base_weight = self.config.edge_type_weights.get(idf_key, 1.0)
        for row in rows:
            slug = row["slug"]
            values = load_json(row[field]) or []
            film_idx = graph.slug_to_idx.get(slug)
            if film_idx is None:
                continue
            for value in values:
                attr_idx = graph.add_node(attr_type, value)
                weight = base_weight * self._idf_multiplier(idf, idf_key, value)
                graph.add_edge(film_idx, attr_idx, idf_key, weight)
                graph.add_edge(attr_idx, film_idx, idf_key, weight)

    def _idf_multiplier(self, idf: dict, idf_key: str, value: str) -> float:
        if not self.config.use_idf:
            return 1.0
        score = idf.get(idf_key, {}).get(value, 1.0)
        return float(np.clip(score, self.config.idf_floor, self.config.idf_ceiling))

    def _classify_interaction(self, row) -> str:
        if row["liked"]:
            return "liked"
        rating = row["rating"]
        if rating is not None:
            if rating >= 4.0:
                return "rated_high"
            if rating >= 3.0:
                return "rated_mid"
            return "rated_low"
        if row["watchlisted"]:
            return "watchlisted"
        if row["watched"]:
            return "watched"
        return "watched"

    def _temporal_decay(self, scraped_at: str | None) -> float:
        """Exponential half-life decay for recency; best-effort parsing."""
        if not self.config.use_temporal_decay or not scraped_at:
            return 1.0
        try:
            ts = parse_timestamp_naive(scraped_at)
            age_days = max((datetime.now() - ts).days, 0)
            half_life = max(self.config.temporal_decay_half_life_days, 1)
            decay = math.pow(2.0, -age_days / half_life)
            return max(decay, self.config.temporal_decay_min_weight)
        except Exception:
            return 1.0

    def _table_exists(self, conn, table: str) -> bool:
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            return bool(row)
        except Exception:
            return False

    def _add_item_similarity_edges(self, graph: HeterogeneousGraph) -> None:
        """Add M->M similarity edges from cached item similarity file."""
        path = self.config.item_similarity_path
        if not path.exists():
            logger.info("Item similarity cache %s not found; skipping similarity edges.", path)
            return
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            logger.warning("Failed to read item similarity cache %s: %s", path, exc)
            return

        items = payload.get("items")
        if not isinstance(items, dict):
            logger.info("Item similarity cache missing 'items' key; skipping.")
            return

        top_k = self.config.item_similarity_top_k
        weight_scale = self.config.item_similarity_weight
        added = 0

        for src_slug, entries in items.items():
            src_idx = graph.slug_to_idx.get(src_slug)
            if src_idx is None:
                continue
            # entries may be list of dicts or list of [slug, score]
            for entry in entries[:top_k]:
                if isinstance(entry, dict):
                    dst_slug = entry.get("slug")
                    score = entry.get("score", 0.0)
                else:
                    try:
                        dst_slug, score = entry
                    except Exception:
                        continue
                dst_idx = graph.slug_to_idx.get(dst_slug)
                if dst_idx is None or score is None:
                    continue
                weight = float(score) * weight_scale
                if weight <= 0:
                    continue
                graph.add_edge(src_idx, dst_idx, "similar_to", weight)
                added += 1
        logger.info("Added %d item similarity edges.", added)

    def _add_follow_edges(self, conn, graph: HeterogeneousGraph) -> None:
        """Add user follow edges if a follow table exists. No-op otherwise."""
        table_candidates = ["user_follows", "user_relationships"]
        rows = []
        for table in table_candidates:
            if not self._table_exists(conn, table):
                continue
            cols = {row["name"] for row in conn.execute(f"PRAGMA table_info('{table}')").fetchall()}
            if {"follower", "followee"} <= cols:
                rows = conn.execute(f"SELECT follower, followee FROM {table}").fetchall()
                break
            if {"source_user", "target_user"} <= cols:
                rows = conn.execute(f"SELECT source_user, target_user FROM {table}").fetchall()
                break
        if not rows:
            logger.info("No follow edges found; skipping social layer.")
            return

        added = 0
        for row in rows:
            follower = row[0]
            followee = row[1]
            src_idx = graph.add_node(NodeType.USER, follower)
            dst_idx = graph.add_node(NodeType.USER, followee)
            graph.add_edge(src_idx, dst_idx, "follows", self.config.follow_edge_weight)
            added += 1
        logger.info("Added %d follow edges.", added)

    def _load_user_films(self, username: str) -> list[dict]:
        with get_db(read_only=True) as conn:
            rows = conn.execute(
                """
                SELECT film_slug as slug, rating, watched, watchlisted, liked, scraped_at
                FROM user_films
                WHERE username = ?
                """,
                (username,),
            ).fetchall()
        return [dict(r) for r in rows]

    def _build_restart_distribution(
        self,
        user_films: list[dict],
        username: str | None = None,
        recent_only: bool = False,
        seed_slugs: Sequence[str] | None = None,
    ) -> np.ndarray:
        """
        Build personalization with mass split between the user node and liked films.

        - restart_user_weight reserves mass for the user node (if present).
        - Remaining mass is distributed over interacted films using restart_weights.
        """
        r = np.zeros(len(self.graph.idx_to_node), dtype=float)

        # User mass
        user_idx = None
        if username:
            user_idx = self.graph.get_idx(NodeType.USER, username)
        if user_idx is not None:
            r[user_idx] = self.config.restart_user_weight
        else:
            # If user node missing, fall back to film-only mass
            user_mass = 0.0
        user_mass = r.sum()

        # Film mass
        film_mass_budget = max(0.0, 1.0 - user_mass)
        film_total = 0.0
        for film in user_films:
            slug = film.get("slug")
            idx = self.graph.slug_to_idx.get(slug)
            if idx is None:
                continue
            key = self._classify_interaction(film)
            weight = self.config.restart_weights.get(key, 0.0)
            if weight <= 0:
                continue
            decay = self._temporal_decay(film.get("scraped_at")) if (self.config.use_temporal_decay or recent_only) else 1.0
            weighted = weight * decay if recent_only else weight
            r[idx] += weighted
            film_total += weighted

        # Optional cold-start seeds
        if seed_slugs:
            seed_weight = self.config.seed_restart_weight
            for slug in seed_slugs:
                idx = self.graph.slug_to_idx.get(slug)
                if idx is None:
                    continue
                r[idx] += seed_weight
                film_total += seed_weight

        if film_total > 0 and film_mass_budget > 0:
            film_scale = film_mass_budget / film_total
            # Scale only film entries (skip user mass already set)
            for film in user_films:
                idx = self.graph.slug_to_idx.get(film.get("slug", ""))
                if idx is None:
                    continue
                r[idx] *= film_scale

        # Normalize fallback if empty
        if r.sum() == 0 and user_idx is not None:
            r[user_idx] = 1.0
        elif r.sum() > 0:
            r /= r.sum()
        return r

    def _aggregate_user_attributes(self, user_films: list[dict]) -> dict[str, set[str]]:
        attrs: dict[str, set[str]] = defaultdict(set)
        liked_slugs = {f["slug"] for f in user_films if f.get("liked")}
        for slug in liked_slugs:
            film = self.graph.film_metadata.get(slug)
            if not film:
                continue
            attrs["director"].update(load_json(film.get("directors", [])) or [])
            attrs["writer"].update(load_json(film.get("writers", [])) or [])
            attrs["cinematographer"].update(load_json(film.get("cinematographers", [])) or [])
            attrs["composer"].update(load_json(film.get("composers", [])) or [])
            attrs["genre"].update(load_json(film.get("genres", [])) or [])
            attrs["actor"].update(load_json(film.get("cast", [])) or [])
            attrs["theme"].update(load_json(film.get("themes", [])) or [])
            attrs["country"].update(load_json(film.get("countries", [])) or [])
            attrs["language"].update(load_json(film.get("languages", [])) or [])
        return attrs

    def _reason_from_attributes(self, film: dict, user_attrs: dict[str, set[str]]) -> list[str]:
        reasons: list[str] = []
        candidates = [
            ("director", load_json(film.get("directors", []))),
            ("writer", load_json(film.get("writers", []))),
            ("cinematographer", load_json(film.get("cinematographers", []))),
            ("composer", load_json(film.get("composers", []))),
            ("genre", load_json(film.get("genres", []))),
            ("actor", load_json(film.get("cast", []))),
            ("theme", load_json(film.get("themes", []))),
            ("country", load_json(film.get("countries", []))),
            ("language", load_json(film.get("languages", []))),
        ]
        for attr_type, values in candidates:
            if not values:
                continue
            overlap = user_attrs.get(attr_type, set()).intersection(values)
            if overlap:
                sample = sorted(overlap)
                reasons.append(f"{attr_type.title()}: {', '.join(sample[:2])}")
            if len(reasons) >= 3:
                break
        return reasons

    def _passes_filters(
        self,
        film: dict,
        min_year: int | None,
        max_year: int | None,
        min_rating: float | None,
        genres: Sequence[str] | None,
        exclude_genres: Sequence[str] | None,
    ) -> bool:
        year = film.get("year")
        if min_year and year and year < min_year:
            return False
        if max_year and year and year > max_year:
            return False
        if min_rating is not None:
            avg_rating = film.get("avg_rating")
            if avg_rating is not None and avg_rating < min_rating:
                return False
        film_genres = set(load_json(film.get("genres", [])) or [])
        if genres and film_genres.isdisjoint(set(genres)):
            return False
        if exclude_genres and set(exclude_genres) & film_genres:
            return False
        return True

    def _attach_paths(self, recs: list[Recommendation], user_films: list[dict]) -> list[Recommendation]:
        liked_slugs = [f["slug"] for f in user_films if f.get("liked")]
        liked_indices = [self.graph.slug_to_idx.get(slug) for slug in liked_slugs]
        liked_indices = [idx for idx in liked_indices if idx is not None]
        for rec in recs:
            rec_idx = self.graph.slug_to_idx.get(rec.slug)
            if rec_idx is None:
                continue
            paths: list[str] = []
            for liked_idx in liked_indices:
                path_list = self.explainer.find_paths(liked_idx, rec_idx)
                for path in path_list:
                    paths.append(self.explainer.format_path(path))
                    if len(paths) >= self.config.max_explanation_paths:
                        break
                if len(paths) >= self.config.max_explanation_paths:
                    break
            if paths:
                rec.reasons = paths
        return recs

    def _should_fallback(self, user_films: list[dict]) -> bool:
        likes = [f for f in user_films if f.get("liked")]
        return len(likes) < self.config.min_likes_for_graph or len(user_films) < self.config.min_films_for_graph

