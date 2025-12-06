import importlib
import sys
from pathlib import Path

import pytest

# Ensure the package under test is importable without installation
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture
def fresh_config(monkeypatch, tmp_path):
    """
    Reload config with a temporary database path to keep tests isolated.
    """
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("LETTERBOXD_DB", str(db_path))
    import letterboxd_rec.config as config

    importlib.reload(config)
    return config


@pytest.fixture
def fresh_db(monkeypatch, tmp_path):
    """
    Reload config/database modules with a temp DB and cleanly close the pool after use.
    """
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("LETTERBOXD_DB", str(db_path))

    import letterboxd_rec.config as config
    import letterboxd_rec.database as database

    importlib.reload(config)
    importlib.reload(database)

    yield database
    database.close_pool()


@pytest.fixture
def fresh_recommender_modules(monkeypatch, tmp_path):
    """
    Reload recommender-related modules with isolated caches/DB.
    """
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("LETTERBOXD_DB", str(db_path))

    import letterboxd_rec.config as config
    import letterboxd_rec.database as database
    import letterboxd_rec.profile as profile
    import letterboxd_rec.recommender as recommender

    importlib.reload(config)
    importlib.reload(database)
    importlib.reload(profile)
    importlib.reload(recommender)

    # Avoid writing caches into the repository
    monkeypatch.setattr(recommender, "ITEM_SIM_CACHE_PATH", tmp_path / "item_sim_cache.json", raising=False)

    yield recommender, profile, database
    database.close_pool()

