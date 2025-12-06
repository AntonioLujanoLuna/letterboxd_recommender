import sys

from letterboxd_rec import cli


def test_cli_dispatch_stats(monkeypatch):
    called = {}

    def fake_stats(args):
        called["command"] = args.command

    monkeypatch.setattr(cli, "cmd_stats", fake_stats)
    monkeypatch.setattr(sys, "argv", ["prog", "stats"])

    cli.main()
    assert called["command"] == "stats"


def test_cli_parses_recommend_args(monkeypatch):
    captured = {}

    def fake_recommend(args):
        captured["username"] = args.username
        captured["strategy"] = args.strategy
        captured["limit"] = args.limit
        captured["genres"] = args.genres

    monkeypatch.setattr(cli, "cmd_recommend", fake_recommend)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "recommend",
            "alice",
            "--strategy",
            "collaborative",
            "--limit",
            "5",
            "--genres",
            "Horror",
            "Drama",
        ],
    )

    cli.main()

    assert captured["username"] == "alice"
    assert captured["strategy"] == "collaborative"
    assert captured["limit"] == 5
    assert captured["genres"] == ["Horror", "Drama"]

