"""`cva-seal explain` and `selftest` (plan §7.11): the demo-step-4 tools."""
from __future__ import annotations

import json
import sqlite3

from cva.provenance.seal.cli import main
from cva.provenance.seal.explain import explain_record, selftest
from cva.provenance.seal.verify import export_records

from ._anchor_helpers import AnchorEnv


def test_selftest_passes_and_says_what_it_checked():
    rows = selftest(records=120)
    assert all(ok for _, ok, _ in rows), [r for r in rows if not r[1]]
    names = " | ".join(n for n, _, _ in rows)
    for must in ("RFC 8032", "empty Merkle tree", "round trip", "single flipped bit", "tail truncation is invisible", "caught with one"):
        assert must in names


def test_selftest_cli_exit_code_and_output(capsys):
    assert main(["selftest", "--records", "60"]) == 0
    assert "all checks passed" in capsys.readouterr().out


def test_explain_a_clean_inference_names_model_input_output_chain_and_anchor(tmp_path):
    e = AnchorEnv(tmp_path)
    for i in range(6):
        e.seal(i)
    a = e.anchor()
    e.close()
    text = explain_record(e.ledger_path, e.trust, 4, anchors=[a])
    for must in ("inference", "signature: VERIFIES", "matches", "model: resnet50-v3", "COVERED", "stored output: matches", "findings: none"):
        assert must in text, (must, text)


def test_explain_points_at_the_exact_edited_record_and_says_why(tmp_path):
    """Demo step 4: hand-edit a record; the tool names it."""
    e = AnchorEnv(tmp_path)
    for i in range(8):
        e.seal(i)
    e.close()
    from attacklab.tamper import TamperDb
    t = TamperDb(e.ledger_path, tmp_path / "t.db")
    rec = t.rec(5)
    rec["output"]["jcs_sha256"] = "0" * 64
    t.put(5, rec)
    t.close()
    text = explain_record(tmp_path / "t.db", e.trust, 5)
    assert "DOES NOT VERIFY" in text and "FINDING [critical] record_edit" in text
    assert "DOES NOT MATCH" in explain_record(tmp_path / "t.db", e.trust, 6)          # the successor's link no longer matches
    assert "findings: none" in explain_record(tmp_path / "t.db", e.trust, 2)


def test_explain_says_the_last_record_is_unprotected_by_the_chain_and_a_missing_seq_is_reported(tmp_path):
    e = AnchorEnv(tmp_path)
    for i in range(3):
        e.seal(i)
    e.close()
    last = len(sqlite3.connect(e.ledger_path).execute("SELECT seq FROM records").fetchall()) - 1
    assert "nothing in the chain would notice" in explain_record(e.ledger_path, e.trust, last)
    assert "No record with seq 99" in explain_record(e.ledger_path, e.trust, 99)
    assert "no external anchor" in explain_record(e.ledger_path, e.trust, 1).lower() or "none supplied" in explain_record(e.ledger_path, e.trust, 1)


def test_explain_cli_and_an_export_source(tmp_path, capsys):
    e = AnchorEnv(tmp_path)
    for i in range(4):
        e.seal(i)
    e.close()
    export_records(e.ledger_path, tmp_path / "x.jsonl")
    (tmp_path / "trust.json").write_bytes(e.trust.to_bytes())
    assert main(["explain", "--records", str(tmp_path / "x.jsonl"), "--trust", str(tmp_path / "trust.json"), "--seq", "3"]) == 0
    out = capsys.readouterr().out
    assert "Record seq 3" in out and "not available in this source" in out
    assert json is not None
