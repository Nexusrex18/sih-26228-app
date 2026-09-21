"""`python -m cva.web.accounts` — how the first account comes into existence.

The admin page needs an admin, so the first one is created from a shell. The property that
matters most is that a password never travels as an argument: argv is readable by every
user on the host through `ps`.
"""
from __future__ import annotations

import io

import pytest

pytest.importorskip("flask", reason="Module E's runtime: pip install -e '.[network]'")

from cva.web import accounts  # noqa: E402


def _run(monkeypatch, capsys, argv, stdin=""):
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    code = accounts.main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def test_create_reads_the_password_from_stdin_and_the_account_authenticates(
        tmp_path, monkeypatch, capsys):
    db = str(tmp_path / "accounts.db")
    code, out, _ = _run(monkeypatch, capsys,
                        ["--db", db, "create", "root.admin", "--role", "admin",
                         "--password-stdin"], "correct-horse-battery\n")
    assert code == 0 and "created root.admin (admin)" in out
    store = accounts.AccountStore(tmp_path / "accounts.db")
    try:
        assert store.authenticate("root.admin", "correct-horse-battery") is not None
    finally:
        store.close()


def test_there_is_no_password_argument_at_all():
    """Not "discouraged" — absent. A flag that exists gets used in a runbook."""
    with pytest.raises(SystemExit):
        accounts.main(["create", "x.y", "--role", "viewer", "--password", "hunter2hunter2"])


def test_a_short_password_is_refused_and_nothing_is_created(tmp_path, monkeypatch, capsys):
    db = str(tmp_path / "accounts.db")
    code, _, err = _run(monkeypatch, capsys,
                        ["--db", db, "create", "a.b", "--role", "viewer",
                         "--password-stdin"], "short\n")
    assert code == 1 and "refused" in err
    code, out, _ = _run(monkeypatch, capsys, ["--db", db, "list"])
    assert "a.b" not in out


def test_list_disable_and_set_role(tmp_path, monkeypatch, capsys):
    db = str(tmp_path / "accounts.db")
    _run(monkeypatch, capsys, ["--db", db, "create", "a.sharma", "--role", "analyst",
                               "--password-stdin"], "correct-horse-battery\n")
    assert _run(monkeypatch, capsys, ["--db", db, "disable", "a.sharma"])[0] == 0
    _, out, _ = _run(monkeypatch, capsys, ["--db", db, "list"])
    assert "a.sharma\tanalyst\tdisabled" in out
    _, out, _ = _run(monkeypatch, capsys, ["--db", db, "set-role", "a.sharma", "approver"])
    assert "now approver" in out
