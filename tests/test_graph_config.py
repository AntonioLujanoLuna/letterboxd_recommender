import numpy as np
import pytest
from scipy import sparse

from letterboxd_rec.graph_config import GraphConfig
from letterboxd_rec.graph_recommender import HeterogeneousGraph, NodeType


def _tiny_graph(cfg: GraphConfig) -> HeterogeneousGraph:
    graph = HeterogeneousGraph(cfg)
    idx = graph.add_node(NodeType.FILM, "a", metadata={"title": "A"})
    graph.transition = sparse.csr_matrix(([1.0], ([idx], [idx])), shape=(1, 1))
    graph.dangling_mask = np.array([False])
    return graph


def test_cache_path_uses_fingerprint(tmp_path):
    cfg = GraphConfig(cache_path=tmp_path / "graph_cache.npz")

    path = cfg.cache_file_path

    assert path.parent == tmp_path
    assert cfg.cache_key in path.name
    assert path.name.startswith("graph_cache-")


def test_restart_weights_must_have_mass():
    with pytest.raises(ValueError):
        GraphConfig(restart_weights={"liked": 0.0}, interaction_weights={"liked": 1.0})


def test_interaction_weights_non_negative():
    with pytest.raises(ValueError):
        GraphConfig(interaction_weights={"liked": -1.0, "watched": 0.1})


def test_cache_rejects_mismatched_config(tmp_path):
    cfg = GraphConfig(cache_path=tmp_path / "graph_cache.npz")
    graph = _tiny_graph(cfg)
    cache_path = cfg.cache_file_path

    graph.save(cache_path)
    assert HeterogeneousGraph.load(cache_path, cfg) is not None

    cfg_mismatch = GraphConfig(alpha=0.25, cache_path=tmp_path / "graph_cache.npz")
    assert HeterogeneousGraph.load(cache_path, cfg_mismatch) is None


@pytest.mark.parametrize(
    "kwargs, message_part",
    [
        ({"alpha": 0}, "alpha"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"tolerance": 0}, "tolerance"),
        ({"max_explanation_paths": 0}, "path limits"),
        ({"min_likes_for_graph": -1}, "minimum thresholds"),
    ],
)
def test_validate_rejects_out_of_bounds_values(kwargs, message_part):
    with pytest.raises(ValueError) as exc:
        GraphConfig(**kwargs)

    assert message_part in str(exc.value)


def test_validate_requires_weight_dicts():
    with pytest.raises(ValueError):
        GraphConfig(restart_weights="oops")


