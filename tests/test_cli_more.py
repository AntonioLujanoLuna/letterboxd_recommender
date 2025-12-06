import sys

from letterboxd_rec import cli


def _run_cli(monkeypatch, argv, target, capture):
    monkeypatch.setattr(target[0], target[1], lambda args: capture.append(args))
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()


def test_cli_refresh_metadata(monkeypatch):
    called = []
    _run_cli(
        monkeypatch,
        ["prog", "refresh-metadata", "--limit", "5", "--maintenance"],
        (cli, "cmd_refresh_metadata"),
        called,
    )
    args = called[0]
    assert args.limit == 5
    assert args.maintenance is True


def test_cli_prune_pending(monkeypatch):
    called = []
    _run_cli(
        monkeypatch,
        ["prog", "prune-pending", "--older-than", "10", "--max-priority", "5"],
        (cli, "cmd_prune_pending"),
        called,
    )
    args = called[0]
    assert args.older_than == 10
    assert args.max_priority == 5


def test_cli_triage(monkeypatch):
    called = []
    _run_cli(
        monkeypatch,
        ["prog", "triage", "alice", "--limit", "3"],
        (cli, "cmd_triage"),
        called,
    )
    args = called[0]
    assert args.username == "alice"
    assert args.limit == 3


def test_cli_similar(monkeypatch):
    called = []
    _run_cli(
        monkeypatch,
        ["prog", "similar", "the-matrix", "--limit", "7"],
        (cli, "cmd_similar"),
        called,
    )
    args = called[0]
    assert args.slug == "the-matrix"
    assert args.limit == 7


def test_cli_gaps(monkeypatch):
    called = []
    _run_cli(
        monkeypatch,
        ["prog", "gaps", "alice", "--min-score", "1.5", "--limit", "4"],
        (cli, "cmd_gaps"),
        called,
    )
    args = called[0]
    assert args.username == "alice"
    assert args.min_score == 1.5
    assert args.limit == 4


def test_cli_export_import(monkeypatch, tmp_path):
    called_export = []
    called_import = []

    export_path = tmp_path / "out.json"

    _run_cli(
        monkeypatch,
        ["prog", "export", str(export_path)],
        (cli, "cmd_export"),
        called_export,
    )
    _run_cli(
        monkeypatch,
        ["prog", "import", str(export_path), "--maintenance"],
        (cli, "cmd_import"),
        called_import,
    )

    assert called_export[0].file == str(export_path)
    assert called_import[0].file == str(export_path)
    assert called_import[0].maintenance is True


def test_cli_svd_info(monkeypatch):
    called = []
    _run_cli(monkeypatch, ["prog", "svd-info"], (cli, "cmd_svd_info"), called)
    assert called


def test_cli_rebuild_idf(monkeypatch):
    called = []
    _run_cli(monkeypatch, ["prog", "rebuild-idf"], (cli, "cmd_rebuild_idf"), called)
    assert called


def test_cli_stats_verbose(monkeypatch):
    called = []
    _run_cli(monkeypatch, ["prog", "--verbose", "stats"], (cli, "cmd_stats"), called)
    assert called[0].verbose is True


def test_cli_discover_continue(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "cmd_discover", lambda args: called.append(args))
    monkeypatch.setattr(sys, "argv", ["prog", "discover", "--continue", "--limit", "3"])
    cli.main()
    assert called[0].continue_mode is True
    assert called[0].limit == 3

