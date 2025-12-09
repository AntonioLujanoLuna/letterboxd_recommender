import json

import pytest

from letterboxd_rec.feature_weights import (
    DEFAULT_WEIGHT,
    MAX_WEIGHT,
    MIN_WEIGHT,
    FeatureWeights,
    load_feature_weights,
    save_feature_weights,
)


def test_factor_defaults_and_non_dict_tables():
    weights = FeatureWeights()
    assert weights.factor("genre", None) == DEFAULT_WEIGHT

    weights.genre = "invalid"
    assert weights.factor("genre", "any") == DEFAULT_WEIGHT


def test_factor_clamps_and_handles_invalid_values():
    weights = FeatureWeights(
        genre={"horror": 10.0, "2000": -5, "bad": "oops"},
    )

    assert weights.factor("genre", "Horror") == MAX_WEIGHT
    assert weights.factor("genre", 2000) == MIN_WEIGHT
    assert weights.factor("genre", "bad") == DEFAULT_WEIGHT


def test_from_dict_normalizes_keys_and_types():
    weights = FeatureWeights.from_dict(
        {
            "genre": {"Horror": "2.5"},
            "director": {"Nolan": 1.5},
            "decade": {1990: "1.2"},
            "metadata": {"source": "test"},
        }
    )

    assert weights.genre["horror"] == pytest.approx(2.5)
    assert weights.director["nolan"] == pytest.approx(1.5)
    assert weights.decade["1990"] == pytest.approx(1.2)
    assert weights.metadata["source"] == "test"


def test_load_feature_weights_handles_missing_and_reads_file(tmp_path):
    missing_path = tmp_path / "missing.json"
    assert load_feature_weights(missing_path) is None

    payload = {"genre": {"drama": 1.7}}
    path = tmp_path / "weights.json"
    path.write_text(json.dumps(payload))

    weights = load_feature_weights(path)
    assert weights is not None
    assert weights.genre["drama"] == pytest.approx(1.7)


def test_save_feature_weights_round_trips(tmp_path):
    weights = FeatureWeights(genre={"drama": 2.1})
    path = tmp_path / "saved_weights.json"

    saved_path = save_feature_weights(weights, path)
    assert saved_path.exists()

    loaded = json.loads(saved_path.read_text())
    assert loaded["genre"]["drama"] == pytest.approx(2.1)


