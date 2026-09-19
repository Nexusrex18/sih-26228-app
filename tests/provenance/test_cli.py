"""`cva-seal` (plan §7.11): keygen, init, bench-durability."""
from __future__ import annotations

import json
import stat

import pytest

from cva.provenance.seal.bench import UNRELIABLE_FS, fs_type
from cva.provenance.seal.chain import verify_chain
from cva.provenance.seal.cli import build_parser, main
from cva.provenance.seal.keys import FileKeyProvider, load_trust_root
from cva.provenance.seal.sealer import Sealer
from cva.provenance.seal.store import SealedLedger

from ._ledger_helpers import CLASSIFY


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_keygen_creates_an_owner_only_key_and_prints_its_identity(tmp_path, capsys):
    key = tmp_path / "ledger.key"
    code, out, _ = run(capsys, "keygen", "--out", str(key))
    info = json.loads(out)
    assert code == 0 and stat.S_IMODE(key.stat().st_mode) == 0o600
    assert info["key_id"] == FileKeyProvider(key).key_id and info["public_key"] == FileKeyProvider(key).public_key.hex()
    assert "seed" not in out and key.read_bytes().hex() not in out            # the secret is never printed


def test_keygen_refuses_to_overwrite_an_existing_key(tmp_path, capsys):
    key = tmp_path / "ledger.key"
    run(capsys, "keygen", "--out", str(key))
    before = key.read_bytes()
    code, _, err = run(capsys, "keygen", "--out", str(key))
    assert code == 1 and "refusing to overwrite" in err and key.read_bytes() == before


def test_init_creates_a_ledger_and_a_trust_root_that_a_sealer_can_use(tmp_path, capsys):
    key, ledger, trust = tmp_path / "k", tmp_path / "l.db", tmp_path / "trust_root.json"
    run(capsys, "keygen", "--out", str(key))
    code, out, _ = run(capsys, "init", "--ledger", str(ledger), "--key", str(key), "--device-id", "jetson-07",
                       "--unit", "alpha", "--checkpoint-every", "50", "--trust-out", str(trust))
    info = json.loads(out)
    assert code == 0 and info["durability"] == "per_record" and info["loss_window"] == "0 records"
    tr = load_trust_root(trust)
    assert tr.deployment_manifest_hash == info["deployment_manifest_hash"]
    with Sealer.open(ledger, key=FileKeyProvider(key), trust_root=trust) as s:
        m = s.register_model(id="m", weights_sha256="a" * 64, arch_hash="b" * 64, format="onnx")
        c = s.register_config(preprocess_spec={"m": 1}, postprocess_spec={"t": 1}, runtime="r", version_pins_hash="c" * 64,
                              code_commit="0" * 40)
        s.seal(b"frame", m, c, output=CLASSIFY, dims=(8, 8))
        assert verify_chain(list(s.ledger.stored_records()),
                            ledger_keys={FileKeyProvider(key).key_id: FileKeyProvider(key).public_key}).ok


def test_init_records_the_durability_choice_and_its_loss_window(tmp_path, capsys):
    key, ledger = tmp_path / "k", tmp_path / "l.db"
    run(capsys, "keygen", "--out", str(key))
    code, out, _ = run(capsys, "init", "--ledger", str(ledger), "--key", str(key), "--device-id", "d", "--unit", "u",
                       "--durability", "group_commit", "--group-n", "25", "--group-ms", "40")
    assert code == 0 and json.loads(out)["loss_window"] == "up to 25 records / 40 ms"
    led = SealedLedger.open(ledger, read_only=True)
    assert (led.durability, led.group_n, led.group_ms) == ("group_commit", 25, 40)


def test_init_never_touches_an_existing_ledger(tmp_path, capsys):
    key, ledger = tmp_path / "k", tmp_path / "l.db"
    run(capsys, "keygen", "--out", str(key))
    args = ("init", "--ledger", str(ledger), "--key", str(key), "--device-id", "d", "--unit", "u")
    assert run(capsys, *args)[0] == 0
    size = ledger.stat().st_size
    code, _, err = run(capsys, *args)
    assert code == 1 and "refusing to re-initialise" in err and ledger.stat().st_size == size


def test_init_with_a_missing_key_fails_cleanly_and_creates_no_ledger(tmp_path, capsys):
    code, _, err = run(capsys, "init", "--ledger", str(tmp_path / "l.db"), "--key", str(tmp_path / "absent"),
                       "--device-id", "d", "--unit", "u")
    assert code == 1 and "keygen" in err and not (tmp_path / "l.db").exists()


def test_bench_durability_refuses_to_bless_tmpfs_numbers(tmp_path, capsys):
    if fs_type(tmp_path) not in UNRELIABLE_FS:
        pytest.skip("tmp_path is on a real filesystem")
    code, out, err = run(capsys, "bench-durability", "--dir", str(tmp_path), "--n", "10", "--input-bytes", "50000")
    assert code == 2 and json.loads(out)["recommendation"] == "UNDETERMINED" and "does not honour fsync" in err


def test_the_parser_rejects_missing_and_unknown_arguments():
    p = build_parser()
    for bad in ([], ["keygen"], ["init", "--ledger", "x"], ["nope"], ["init", "--ledger", "x", "--key", "k",
                "--device-id", "d", "--unit", "u", "--durability", "sometimes"]):
        with pytest.raises(SystemExit):
            p.parse_args(bad)


def test_the_console_script_is_declared_in_pyproject():
    import tomllib
    from pathlib import Path
    scripts = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["cva-seal"] == "cva.provenance.seal.cli:main"
