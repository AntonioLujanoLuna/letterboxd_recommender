import importlib

from letterboxd_rec import config


def test_env_overrides_and_validation(monkeypatch, tmp_path):
    monkeypatch.setenv("LETTERBOXD_SCRAPER_DELAY", "2.5")
    monkeypatch.setenv("LETTERBOXD_ASYNC_DELAY", "-1")  # should clamp to min
    monkeypatch.setenv("LETTERBOXD_MAX_CONCURRENT", "0")  # min clamp

    cfg = importlib.reload(config)

    assert cfg.DEFAULT_SCRAPER_DELAY == 2.5
    assert cfg.DEFAULT_ASYNC_DELAY == 0.0
    assert cfg.DEFAULT_MAX_CONCURRENT == 1


def test_db_path_respects_env(monkeypatch, tmp_path):
    db_path = tmp_path / "custom.db"
    monkeypatch.setenv("LETTERBOXD_DB", str(db_path))

    cfg = importlib.reload(config)

    assert cfg.DB_PATH == db_path


def test_invalid_env_values_fall_back_to_defaults(monkeypatch):
    # Use clearly invalid strings to exercise the ValueError branches
    monkeypatch.setenv("LETTERBOXD_SCRAPER_DELAY", "not-a-float")
    monkeypatch.setenv("LETTERBOXD_ASYNC_DELAY", "oops")
    monkeypatch.setenv("LETTERBOXD_MAX_CONCURRENT", "bad-int")

    cfg = importlib.reload(config)

    assert cfg.DEFAULT_SCRAPER_DELAY == 1.0
    assert cfg.DEFAULT_ASYNC_DELAY == 0.2
    assert cfg.DEFAULT_MAX_CONCURRENT == 5

def test_cookie_fallback_reads_line_dotenv_rejects(tmp_path):
    from letterboxd_rec.config import _read_cookie_from_dotenv

    env = tmp_path / ".env"
    # An embedded quote makes python-dotenv fail on lines like this one.
    env.write_text('LETTERBOXD_COOKIE="a=1; b={"x":"y"}; c=3"\n', encoding="utf-8")
    assert _read_cookie_from_dotenv(env) == 'a=1; b={"x":"y"}; c=3'
    assert _read_cookie_from_dotenv(tmp_path / "missing.env") is None
    env.write_text("OTHER=1\n", encoding="utf-8")
    assert _read_cookie_from_dotenv(env) is None


def test_anonymous_flag_ignores_configured_cookie(monkeypatch, tmp_path):
    import importlib
    import letterboxd_rec.config as config

    monkeypatch.setenv("LETTERBOXD_COOKIE", "a=1; b=2")
    monkeypatch.setenv("LETTERBOXD_DB", str(tmp_path / "t.db"))
    try:
        monkeypatch.delenv("LETTERBOXD_ANONYMOUS", raising=False)
        assert importlib.reload(config).LETTERBOXD_COOKIE == "a=1; b=2"
        monkeypatch.setenv("LETTERBOXD_ANONYMOUS", "1")
        assert importlib.reload(config).LETTERBOXD_COOKIE is None
    finally:
        monkeypatch.delenv("LETTERBOXD_ANONYMOUS", raising=False)
        monkeypatch.delenv("LETTERBOXD_COOKIE", raising=False)
        importlib.reload(config)
