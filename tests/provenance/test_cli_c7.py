"""`cva-seal` end to end through its public entry point: the whole ceremony as an operator would run it."""
from __future__ import annotations

import json

from cva.provenance.seal.cli import main

from ._chain_helpers import provider


def run(capsys, *argv):
    code = main([str(a) for a in argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def js(out):
    return json.loads(out[out.index("{"):out.rindex("}") + 1])


def setup(tmp_path, capsys, n=6):
    """keygen x3, init, seal a few records with the real Sealer, extend the trust root with witness/boundary."""
    keys = {}
    for name in ("ledger", "witness", "boundary", "next"):
        code, out, _ = run(capsys, "keygen", "--out", tmp_path / f"{name}.key")
        assert code == 0
        keys[name] = js(out)
    code, out, _ = run(capsys, "init", "--ledger", tmp_path / "l.db", "--key", tmp_path / "ledger.key",
                       "--device-id", "d1", "--unit", "u1", "--checkpoint-every", "1000",
                       "--trust-out", tmp_path / "trust.json")
    assert code == 0
    for role in ("witness", "boundary"):
        code, *_ = run(capsys, "trust", "add", "--trust", tmp_path / "trust.json", "--public-key",
                       keys[role]["public_key"], "--role", role)
        assert code == 0
    from cva.provenance.seal.keys import FileKeyProvider
    from cva.provenance.seal.sealer import Sealer
    with Sealer.open(tmp_path / "l.db", key=FileKeyProvider(tmp_path / "ledger.key"), trust_root=tmp_path / "trust.json") as s:
        m = s.register_model(id="m", weights_sha256="ab" * 32, arch_hash="cd" * 32, format="onnx")
        c = s.register_config(preprocess_spec={"a": 1}, postprocess_spec={"b": 2}, runtime="rt", version_pins_hash="ef" * 32,
                              code_commit="0" * 40)
        for i in range(n):
            s.seal(bytes([i]) * 300, m, c, output={"task": "classify", "top": [{"cls": i, "conf": 0.9}]}, dims=(8, 8))
    return keys


def test_the_full_ceremony_from_the_command_line(tmp_path, capsys):
    keys = setup(tmp_path, capsys)
    T = tmp_path / "trust.json"
    # anchor, cosign, attest, verify the artefact
    code, out, _ = run(capsys, "anchor", "export", "--ledger", tmp_path / "l.db", "--key", tmp_path / "ledger.key",
                       "--out", tmp_path / "a.json", "--medium", "cosign", "--label", "shift-1",
                       "--cosigner", keys["witness"]["key_id"])
    assert code == 0 and js(out)["tree_size"] == 8
    assert run(capsys, "anchor", "cosign", "--anchor", tmp_path / "a.json", "--witness-key", tmp_path / "witness.key",
               "--out", tmp_path / "a.cosigned.json")[0] == 0
    assert run(capsys, "anchor", "attest", "--anchor", tmp_path / "a.cosigned.json", "--boundary-key",
               tmp_path / "boundary.key", "--observed-at", "2099-01-01T00:00:00.000000Z",
               "--out", tmp_path / "a.full.json")[0] == 0
    code, out, _ = run(capsys, "anchor", "verify", "--anchor", tmp_path / "a.full.json", "--trust", T)
    v = js(out)
    assert code == 0 and v["valid"] and v["cosigners"] == [keys["witness"]["key_id"]]
    assert v["sealed_not_after_utc"] == "2099-01-01T00:00:00.000000Z"
    # the operator's logbook
    code, out, _ = run(capsys, "anchor", "print", "--anchor", tmp_path / "a.full.json")
    assert code == 0 and "tree size : 8" in out and "checksum" in out
    # export, remove the database, verify from the export + anchor alone
    assert run(capsys, "export", "--ledger", tmp_path / "l.db", "--out", tmp_path / "x.jsonl",
               "--payloads-out", tmp_path / "x.payloads")[0] == 0
    (tmp_path / "l.db").unlink()
    code, out, _ = run(capsys, "verify", "--records", tmp_path / "x.jsonl", "--trust", T, "--anchor",
                       tmp_path / "a.full.json", "--payloads", tmp_path / "x.payloads")
    assert code == 0 and out.startswith("CLEAN") and "1 anchor(s) verified" in out and "2099-01-01" in out
    # inclusion proof, from nothing but the proof + anchor + trust root
    assert run(capsys, "proof", "inclusion", "--records", tmp_path / "x.jsonl", "--seq", 4, "--anchor",
               tmp_path / "a.full.json", "--out", tmp_path / "p.json")[0] == 0
    (tmp_path / "x.jsonl").unlink()
    code, out, _ = run(capsys, "proof", "verify", "--proof", tmp_path / "p.json", "--anchor",
                       tmp_path / "a.full.json", "--trust", T)
    assert code == 0 and js(out)["valid"] and js(out)["signature_ok"] is True


def test_verify_exits_2_and_names_the_problem_and_says_what_is_unwitnessed(tmp_path, capsys):
    setup(tmp_path, capsys)
    run(capsys, "export", "--ledger", tmp_path / "l.db", "--out", tmp_path / "x.jsonl")
    lines = (tmp_path / "x.jsonl").read_bytes().split(b"\n")[:-1]
    (tmp_path / "x.jsonl").write_bytes(b"".join(x + b"\n" for x in lines[:3] + lines[4:]))         # delete record 3
    code, out, _ = run(capsys, "verify", "--records", tmp_path / "x.jsonl", "--trust", tmp_path / "trust.json")
    assert code == 2 and "PROBLEMS FOUND" in out and "record_delete" in out
    assert "NO external anchor was supplied" in out


def test_verify_without_an_anchor_says_out_loud_that_truncation_cannot_be_excluded(tmp_path, capsys):
    setup(tmp_path, capsys)
    code, out, _ = run(capsys, "verify", "--records", tmp_path / "l.db", "--trust", tmp_path / "trust.json")
    assert code == 0 and "cannot be excluded" in out


def test_a_truncated_ledger_is_caught_by_the_anchor_from_the_command_line(tmp_path, capsys):
    setup(tmp_path, capsys)
    run(capsys, "anchor", "export", "--ledger", tmp_path / "l.db", "--key", tmp_path / "ledger.key",
        "--out", tmp_path / "a.json")
    run(capsys, "export", "--ledger", tmp_path / "l.db", "--out", tmp_path / "x.jsonl")
    lines = (tmp_path / "x.jsonl").read_bytes().split(b"\n")[:-1]
    (tmp_path / "x.jsonl").write_bytes(b"".join(x + b"\n" for x in lines[:-4]))
    T = tmp_path / "trust.json"
    assert run(capsys, "verify", "--records", tmp_path / "x.jsonl", "--trust", T)[0] == 0        # silent without it
    code, out, _ = run(capsys, "verify", "--records", tmp_path / "x.jsonl", "--trust", T, "--anchor", tmp_path / "a.json")
    assert code == 2 and "tail_truncation" in out


def test_a_logbook_entry_can_be_supplied_on_the_command_line(tmp_path, capsys):
    setup(tmp_path, capsys)
    code, out, _ = run(capsys, "anchor", "export", "--ledger", tmp_path / "l.db", "--key", tmp_path / "ledger.key",
                       "--out", tmp_path / "a.json")
    root = js(out)["root_hash"]
    T = tmp_path / "trust.json"
    code, out, _ = run(capsys, "verify", "--records", tmp_path / "l.db", "--trust", T, "--logbook", f"8:{root}")
    assert code == 0 and "1 anchor(s) verified" in out
    code, out, _ = run(capsys, "verify", "--records", tmp_path / "l.db", "--trust", T, "--logbook", f"8:{'0' * 64}")
    assert code == 2 and "ledger_fork" in out
    code, _, err = run(capsys, "verify", "--records", tmp_path / "l.db", "--trust", T, "--logbook", f"8:{root}:00000000")
    assert code == 1 and "mistyped" in err


def test_rotate_from_the_command_line_and_the_trust_root_needs_no_change(tmp_path, capsys):
    keys = setup(tmp_path, capsys)
    code, out, _ = run(capsys, "rotate", "--ledger", tmp_path / "l.db", "--key", tmp_path / "ledger.key",
                       "--new-key", tmp_path / "next.key")
    assert code == 0 and js(out)["incoming_key_id"] == keys["next"]["key_id"]
    code, out, _ = run(capsys, "verify", "--records", tmp_path / "l.db", "--trust", tmp_path / "trust.json", "--json")
    v = js(out)
    assert code == 0 and v["key_rotations"] == 1 and v["clean"]
    # the old key can no longer write
    code, _, err = run(capsys, "anchor", "export", "--ledger", tmp_path / "l.db", "--key", tmp_path / "ledger.key",
                       "--out", tmp_path / "a.json")
    assert code == 1 and "active signing key" in err


def test_the_ceremony_refuses_to_overwrite_an_anchor_and_trust_add_refuses_a_duplicate(tmp_path, capsys):
    keys = setup(tmp_path, capsys)
    args = ("anchor", "export", "--ledger", tmp_path / "l.db", "--key", tmp_path / "ledger.key", "--out", tmp_path / "a.json")
    assert run(capsys, *args)[0] == 0
    code, _, err = run(capsys, *args)
    assert code == 1 and "refusing to overwrite" in err
    code, _, err = run(capsys, "trust", "add", "--trust", tmp_path / "trust.json", "--public-key",
                       keys["witness"]["public_key"], "--role", "witness")
    assert code == 1 and "already in the trust root" in err


def test_a_corrupt_anchor_verifies_as_invalid_with_exit_2(tmp_path, capsys):
    setup(tmp_path, capsys)
    run(capsys, "anchor", "export", "--ledger", tmp_path / "l.db", "--key", tmp_path / "ledger.key", "--out", tmp_path / "a.json")
    a = json.loads((tmp_path / "a.json").read_text())
    a["checkpoint"]["checkpoint"]["root_hash"] = "0" * 64
    (tmp_path / "bad.json").write_text(json.dumps(a))
    code, out, _ = run(capsys, "anchor", "verify", "--anchor", tmp_path / "bad.json", "--trust", tmp_path / "trust.json")
    assert code == 2 and not js(out)["valid"]
    assert provider is not None
