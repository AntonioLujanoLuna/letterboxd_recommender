import sys

import pytest

from letterboxd_rec import cli


def test_validate_slug_and_username():
    assert cli._validate_slug("valid-slug") == "valid-slug"
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

