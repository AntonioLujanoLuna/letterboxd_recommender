import sys
from types import SimpleNamespace

import pytest

from letterboxd_rec import cli


def test_validate_slug_and_username():
    assert cli._validate_slug("valid-slug") == "valid-slug"
    assert cli._validate_slug("film:482919") == "film:482919"
    with pytest.raises(ValueError):
        cli._validate_slug("Invalid slug!")

    # Username should be sanitized to lowercase alphanumeric/underscore/hyphen
    assert cli._validate_username("Bad Name!") == "badname"


def test_main_dispatches_to_subcommand(monkeypatch):
    called = {}

    def fake_stats(args):
        called["command"] = args.command

    monkeypatch.setattr(cli, "cmd_stats", fake_stats)
    monkeypatch.setattr(sys, "argv", ["prog", "stats"])

    cli.main()

    assert called["command"] == "stats"


def test_send_notification_uses_webhook(monkeypatch):
    payload = {}

    def fake_post(url, json, timeout):
        payload.update({"url": url, "json": json, "timeout": timeout})

    dummy_httpx = SimpleNamespace(post=fake_post)
    monkeypatch.setitem(sys.modules, "httpx", dummy_httpx)
    monkeypatch.setattr(cli, "NOTIFICATION_WEBHOOK_URL", "https://hook.test")

    cli.send_notification("hello world")

    assert payload["url"] == "https://hook.test"
    assert payload["json"]["content"] == "hello world"
    assert payload["timeout"] == 10

