"""The stores under waitress's threads — the race the real browser found.

A page loads a dozen script chunks in parallel; each request reads the account row to check
the session. With one shared SQLite connection those reads interleaved and returned each
other's rows: 500s on random chunks and spurious redirects to /login. These tests hammer
both stores from many threads at once and require every answer to be the right one.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from cva.web.accounts import AccountStore


def test_account_reads_from_many_threads_never_mix_rows(tmp_path):
    store = AccountStore(tmp_path / "accounts.db")
    actors = [f"user.{i:02d}" for i in range(8)]
    roles = ("viewer", "analyst", "approver", "admin")
    for i, a in enumerate(actors):
        store.create(a, "correct-horse-battery", roles[i % 4])
    expected = {a: roles[i % 4] for i, a in enumerate(actors)}

    errors: list[str] = []
    start = threading.Barrier(16)

    def hammer(n: int) -> None:
        start.wait()
        for k in range(300):
            actor = actors[(n + k) % len(actors)]
            try:
                acct = store.get(actor)
            except Exception as e:  # the old failure: IndexError / TypeError / InterfaceError
                errors.append(f"{type(e).__name__}: {e}")
                continue
            if acct is None or acct.actor_id != actor or acct.role != expected[actor]:
                errors.append(f"asked for {actor}, got {acct}")

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(hammer, range(16)))
    assert not errors, f"{len(errors)} wrong answers, e.g. {errors[:3]}"


def test_index_reads_from_many_threads_while_refreshing(tmp_path):
    from cva.web import fixtures
    from cva.web.reports.index import ReportIndex

    reports = tmp_path / "reports"
    fixtures.write(reports)
    idx = ReportIndex(tmp_path / "index.db", reports)
    idx.refresh()
    want = {r.scan_id: r.n_findings for r in idx.scans()}

    errors: list[str] = []
    start = threading.Barrier(12)

    def reader(n: int) -> None:
        start.wait()
        for _ in range(150):
            try:
                if n % 4 == 0:
                    idx.refresh()
                got = {r.scan_id: r.n_findings for r in idx.scans()}
                if got != want:
                    errors.append(f"scan list differs: {got}")
                for sid, total in want.items():
                    if idx.count(sid) != total:
                        errors.append(f"{sid}: count {idx.count(sid)} != {total}")
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(reader, range(12)))
    assert not errors, f"{len(errors)} wrong answers, e.g. {errors[:3]}"
