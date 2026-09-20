"""Gate C9 — the C core against the Python reference (plan §12): cross-implementation byte identity on the frozen
vectors; records sealed by the C core verify in the Python verifier and vice versa; identical canonical bytes for
identical logical input; and they agree on what is NOT canonical.

Skipped when there is no C toolchain / libsodium / sqlite3 headers (the build is part of the test).
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from cva.provenance.seal.canonical import NonCanonical, canonical_bytes, parse_strict
from cva.provenance.seal.store import SealedLedger
from cva.provenance.seal.verify import export_records, verify_ledger

from ._chain_helpers import SEED_A, SEED_B, clock, provider, rng
from ._fixtures import BODIES
from ._ledger_helpers import MANIFEST

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "native" / "c"
VEC = json.loads((ROOT / "spec/vectors/cva_seal_v1.json").read_text())


def _unavailable(why: str) -> None:
    """A missing toolchain SKIPS locally but FAILS where it is required (CI sets CVA_SEAL_REQUIRE_NATIVE=1): the C9 gate
    must not be green merely because nothing ran."""
    if os.environ.get("CVA_SEAL_REQUIRE_NATIVE"):
        pytest.fail(f"CVA_SEAL_REQUIRE_NATIVE is set but {why}")
    pytest.skip(why)


@pytest.fixture(scope="session")
def cvseal() -> Path:
    if not (shutil.which("cc") and shutil.which("make")):
        _unavailable("no C toolchain")
    probe = subprocess.run(["cc", "-x", "c", "-", "-o", "/dev/null", "-lsodium", "-lsqlite3"],
                           input=b"#include <sodium.h>\n#include <sqlite3.h>\nint main(void){return 0;}", capture_output=True)
    if probe.returncode:
        _unavailable("libsodium / sqlite3 development files are not installed")
    r = subprocess.run(["make", "-C", str(NATIVE)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return NATIVE / "cvseal"


def run(exe, *args, stdin: bytes | None = None, check=True):
    r = subprocess.run([str(exe), *map(str, args)], input=stdin, capture_output=True)
    if check and r.returncode:
        raise AssertionError(f"cvseal {args[:2]} failed: {r.stderr.decode()}")
    return r


def seed_hex(seed: bytes) -> str:
    return seed.hex()


def export_of(rows: list[str]) -> bytes:
    return "".join(r + "\n" for r in rows).encode()


# --- 1. the frozen vectors ---------------------------------------------------------------------------------------------

def test_the_c_core_reproduces_every_frozen_record_byte_for_byte(cvseal, tmp_path):
    (tmp_path / "seeds").write_text("".join(k["seed"] + "\n" for k in VEC["keys"].values()))
    out = run(cvseal, "reseal", tmp_path / "seeds", stdin=export_of(VEC["records"])).stdout
    assert out == export_of(VEC["records"]), "the C core re-signed and re-chained the frozen ledger differently"


def test_the_c_core_computes_every_frozen_prefix_root(cvseal):
    out = run(cvseal, "roots", stdin=export_of(VEC["records"])).stdout.decode().split("\n")[:-1]
    assert out == [f"{n} {VEC['roots'][str(n)]}" for n in range(1, len(VEC["records"]) + 1)]


def test_the_c_core_verifies_the_frozen_ledger_including_its_rotation_and_checkpoints(cvseal, tmp_path):
    (tmp_path / "x.jsonl").write_bytes(export_of(VEC["records"]))
    r = run(cvseal, "verify", tmp_path / "x.jsonl", VEC["keys"]["ledger"]["public_key"])
    assert b"ok: 20 records" in r.stdout
    man_hash = json.loads(VEC["records"][0])["prev_record_hash"]
    assert run(cvseal, "verify", tmp_path / "x.jsonl", VEC["keys"]["ledger"]["public_key"], man_hash).returncode == 0
    bad = run(cvseal, "verify", tmp_path / "x.jsonl", VEC["keys"]["ledger"]["public_key"], "0" * 64, check=False)
    assert bad.returncode != 0 and b"different deployment" in bad.stderr


def test_the_c_verifier_catches_single_bit_flips_and_structural_edits_the_python_verifier_catches(cvseal, tmp_path):
    from cva.provenance.seal.keys import parse_trust_root
    trust = parse_trust_root(json.dumps(VEC["trust_root"]).encode())
    data = export_of(VEC["records"])
    pub = VEC["keys"]["ledger"]["public_key"]
    rnd = random.Random(99)
    missed = []
    for _ in range(400):
        b = bytearray(data)
        i = rnd.randrange(len(b))
        b[i] ^= 1 << rnd.randrange(8)
        (tmp_path / "m.jsonl").write_bytes(bytes(b))
        c_rejects = run(cvseal, "verify", tmp_path / "m.jsonl", pub, check=False).returncode != 0
        py_finds = not verify_ledger(bytes(b), trust_root=trust).clean
        # a flip in the one thing the C verifier deliberately leaves to Python (per-section schema) may pass in C; it must
        # never be the other way round, and an integrity break (signature/link/canonical/seq) must always fail in C
        if py_finds and not c_rejects:
            missed.append(i)
    lines = data.split(b"\n")[:-1]
    for name, ls in {"delete": lines[:5] + lines[6:], "dup": lines[:5] + [lines[4]] + lines[5:], "swap": lines[:3] + [lines[4], lines[3]] + lines[5:],
                     "truncate_mid_line": None, "blank": lines[:4] + [b""] + lines[4:]}.items():
        blob = data[:-7] if ls is None else b"".join(x + b"\n" for x in ls)
        (tmp_path / "s.jsonl").write_bytes(blob)
        assert run(cvseal, "verify", tmp_path / "s.jsonl", pub, check=False).returncode != 0, name
    # what C leaves to Python is only section-content flips: every miss must be inside a section value, never a header/signature
    for i in missed:
        line_start = data.rfind(b"\n", 0, i) + 1
        line = data[line_start:data.index(b"\n", i)]
        j = i - line_start
        assert b'"signature":"' not in line[j - 20:j + 1] and j < len(line) - 140, (i, line[max(0, j - 30):j + 30])


# --- 2. canonicalisation: identical bytes for identical logical input ----------------------------------------------------------

def _shuffle_json(rnd: random.Random, obj, *, escape=False):
    """The same logical value with random key order, whitespace and (optionally) \\u escapes."""
    def s(x):
        out = json.dumps(x, ensure_ascii=True)
        if escape and isinstance(x, str) and x:
            out = '"' + "".join(f"\\u{ord(c):04x}" if rnd.random() < 0.3 else ("\\" + c if c in '"\\' else c) for c in x) + '"'
        return out

    def ws():
        return rnd.choice(["", " ", "\n", "\t ", "  "])
    if isinstance(obj, dict):
        items = list(obj.items())
        rnd.shuffle(items)
        return "{" + ws() + ("," + ws()).join(s(k) + ws() + ":" + ws() + _shuffle_json(rnd, v, escape=escape) for k, v in items) + ws() + "}"
    if isinstance(obj, list):
        return "[" + ws() + ("," + ws()).join(_shuffle_json(rnd, v, escape=escape) for v in obj) + ws() + "]"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    return s(obj) if isinstance(obj, str) else json.dumps(obj)


def _random_value(rnd: random.Random, depth=0):
    kinds = ["int", "str", "bool"] + (["obj", "arr"] if depth < 4 else [])
    k = rnd.choice(kinds)
    if k == "int":
        return rnd.choice([0, 1, -1, 7, 2**53 - 1, -(2**53 - 1), rnd.randrange(-10**9, 10**9)])
    if k == "str":
        return "".join(rnd.choice(' !"#%&\'()*+,-./09:;<=>?@AZ[\\]^_`az{|}~') for _ in range(rnd.randrange(0, 12)))
    if k == "bool":
        return rnd.random() < 0.5
    if k == "obj":
        return {rnd.choice(["a", "b", "zz", "k_1", "x9", "long_key_name", "0", "_"]) + str(i): _random_value(rnd, depth + 1)
                for i in range(rnd.randrange(0, 5))}
    kind = rnd.choice(["int", "str", "bool", "obj"])
    return [{"a": rnd.randrange(9), "b": "s"} if kind == "obj" else
            (rnd.randrange(-5, 99) if kind == "int" else ("t" * rnd.randrange(3) if kind == "str" else rnd.random() < .5))
            for _ in range(rnd.randrange(0, 4))]


def test_random_logical_objects_canonicalise_identically_in_c_and_python(cvseal):
    rnd = random.Random(20260919)
    n = 0
    for _ in range(400):
        obj = {"top": _random_value(rnd), "z": _random_value(rnd), "a_1": _random_value(rnd)}
        want = canonical_bytes(obj)
        for escape in (False, True):
            text = _shuffle_json(rnd, obj, escape=escape)
            got = run(cvseal, "canon", stdin=text.encode()).stdout
            assert got == want, (text, got, want)
            n += 1
    assert n == 800


@pytest.mark.parametrize("bad", [
    b'{"a":1.5}', b'{"a":1e3}', b'{"a":9007199254740992}', b'{"a":007}', b'{"A":1}', b'{"a-b":1}', b'{"a":1,"a":2}',
    b'{"a":[1,"x"]}', b'{"a":[1,null]}', b'{"a":"caf\xc3\xa9"}', b'{"a":"\\u00e9"}', b'{"a":"\\n"}', b'{"a":"\x01"}',
    b'[1]', b'"x"', b'{"a":1} x', b'{"a":', b'{"a":NaN}', b'{"a":Infinity}', b'',
    b'{"a":{"a":{"a":{"a":{"a":{"a":{"a":{"a":{"a":1}}}}}}}}}',       # nine levels
])
def test_both_implementations_refuse_what_the_profile_forbids(cvseal, bad):
    with pytest.raises((NonCanonical, ValueError)):
        canonical_bytes(parse_strict(bad, require_canonical=False))
    assert run(cvseal, "canon", stdin=bad, check=False).returncode != 0


def test_a_negative_zero_is_read_as_zero_by_both_and_is_never_a_valid_STORED_spelling(cvseal, tmp_path):
    """As logical input `-0` is the integer 0 in both; as stored bytes it is not the canonical spelling and the C
    verifier rejects it because re-serialising gives `0` (byte inequality)."""
    assert canonical_bytes(parse_strict(b'{"a":-0}', require_canonical=False)) == b'{"a":0}'
    assert run(cvseal, "canon", stdin=b'{"a":-0}').stdout == b'{"a":0}'
    (tmp_path / "x.jsonl").write_bytes(export_of(VEC["records"]).replace(b'"seq":0,', b'"seq":-0,', 1))
    assert run(cvseal, "verify", tmp_path / "x.jsonl", VEC["keys"]["ledger"]["public_key"], check=False).returncode != 0


def test_eight_levels_of_nesting_is_the_limit_in_both(cvseal):
    ok = {"a": {"a": {"a": {"a": {"a": {"a": {"a": 1}}}}}}}               # seven object levels + the top = depth 7 containers... plus
    for depth in range(1, 10):
        o: object = 1
        for _ in range(depth):
            o = {"a": o}
        try:
            canonical_bytes(o)  # type: ignore[arg-type]
            py = True
        except NonCanonical:
            py = False
        c = run(cvseal, "canon", stdin=json.dumps(o).encode(), check=False).returncode == 0
        assert py == c, depth
    assert ok


# --- 3. C writes, Python verifies ------------------------------------------------------------------------------------------------

def _c_ledger(cvseal, tmp_path, *, every=1000, n=6, rotate_at=None, name="c.db"):
    db = tmp_path / name
    man = json.dumps({"device_id": MANIFEST["device_id"], "unit": MANIFEST["unit"], "profile_hash": MANIFEST["profile_hash"],
                      "checkpoint_every": every})
    seed = seed_hex(SEED_A)
    run(cvseal, "init", db, seed, man)
    run(cvseal, "append", db, seed, "model_registration", json.dumps({k: BODIES["model_registration"][k] for k in ("model", "config")}))
    for i in range(n):
        if rotate_at is not None and i == rotate_at:
            run(cvseal, "rotate", db, seed, seed_hex(SEED_B))
            seed = seed_hex(SEED_B)
        run(cvseal, "append", db, seed, "inference", json.dumps({k: BODIES["inference"][k] for k in ("input", "model", "config", "output")}))
    return db


def _trust(db: Path, key_seeds=(SEED_A,)):
    from cva.provenance.seal.keys import TrustKey, TrustRoot
    from cva.provenance.seal.records import genesis_prev_hash
    led = SealedLedger.open(db, read_only=True)
    try:
        k = provider(SEED_A)
        return TrustRoot(genesis_prev_hash(led.deployment_manifest), (TrustKey(k.key_id, k.public_key, "ledger"),))
    finally:
        led.close()


def test_a_ledger_written_by_the_c_core_passes_the_python_verifier_clean(cvseal, tmp_path):
    db = _c_ledger(cvseal, tmp_path, every=4, n=12)
    rep = verify_ledger(db, trust_root=_trust(db))
    assert rep.clean and rep.checkpoints_verified == 4 and rep.records_checked == 2 + 12 + 4, rep.classes()
    exp = tmp_path / "c.jsonl"
    export_records(db, exp)
    assert verify_ledger(exp, trust_root=_trust(db)).clean


def test_a_c_ledger_with_a_rotation_verifies_in_python_with_only_the_genesis_key_in_the_trust_root(cvseal, tmp_path):
    db = _c_ledger(cvseal, tmp_path, every=3, n=10, rotate_at=4)
    rep = verify_ledger(db, trust_root=_trust(db))
    assert rep.clean and rep.rotations == 1 and rep.active_key_id == provider(SEED_B).key_id, rep.classes()


def test_the_c_ledger_is_a_first_class_python_store(cvseal, tmp_path):
    """Python opens it (schema, Merkle cache shape and all), checks its own cache, and CONTINUES it; then C continues that."""
    db = _c_ledger(cvseal, tmp_path, every=5, n=7)
    k = provider(SEED_A)
    led = SealedLedger.open(db, key=k, clock=clock(), rng=rng(), background_flush=False)
    assert led.verify_merkle_cache() and led.chain_ok()
    for _ in range(4):
        led.append_typed("inference", {s: BODIES["inference"][s] for s in ("input", "model", "config", "output")})
    led.close()
    run(cvseal, "append", db, seed_hex(SEED_A), "inference", json.dumps({s: BODIES["inference"][s] for s in ("input", "model", "config", "output")}))
    rep = verify_ledger(db, trust_root=_trust(db))
    assert rep.clean, [(f.attack_class, f.seq) for f in rep.findings]
    led = SealedLedger.open(db, read_only=True)
    assert led.verify_merkle_cache()
    led.close()


def test_c_and_python_write_identical_records_and_identical_merkle_nodes_for_the_same_input(cvseal, tmp_path):
    """Same key, manifest, bodies, timestamps and nonces -> the same bytes in `records` and the same rows in `merkle_nodes`."""
    py_dir = tmp_path / "py"
    py_dir.mkdir()
    key = provider(SEED_A)
    led = SealedLedger.init_ledger(py_dir / "l.db", key, {**MANIFEST, "checkpoint_every": 1000}, clock=clock(), rng=rng())
    led.append_typed("model_registration", {k: BODIES["model_registration"][k] for k in ("model", "config")})
    for _ in range(5):
        led.append_typed("inference", {s: BODIES["inference"][s] for s in ("input", "model", "config", "output")})
    new = provider(SEED_B)
    led.rotate_key(new)
    led2_body = {s: BODIES["inference"][s] for s in ("input", "model", "config", "output")}
    led.append_typed("inference", led2_body)
    led.close()
    conn = sqlite3.connect(py_dir / "l.db")
    py_recs = [json.loads(r[0]) for r in conn.execute("SELECT rec FROM records ORDER BY seq")]
    py_nodes = conn.execute("SELECT level, idx, hash FROM merkle_nodes ORDER BY level, idx").fetchall()
    conn.close()
    # replay the same logical operations through C with the timestamps and nonces Python chose
    db = tmp_path / "c.db"
    seed = seed_hex(SEED_A)
    man = json.dumps({k: MANIFEST[k] for k in ("device_id", "unit", "profile_hash")} | {"checkpoint_every": 1000})
    g = py_recs[0]
    run(cvseal, "init", db, seed, man, g["created_at_utc"], g["nonce"])
    for r in py_recs[1:]:
        if r["type"] == "key_rotation":
            run(cvseal, "rotate", db, seed, seed_hex(SEED_B), r["created_at_utc"], r["nonce"])
            seed = seed_hex(SEED_B)
        else:
            body = json.dumps({s: r[s] for s in r if s not in ("v", "type", "seq", "prev_record_hash", "key_id", "created_at_utc", "nonce", "signature")})
            run(cvseal, "append", db, seed, r["type"], body, r["created_at_utc"], r["nonce"])
    conn = sqlite3.connect(db)
    c_recs = [json.loads(r[0]) for r in conn.execute("SELECT rec FROM records ORDER BY seq")]
    c_raw = [r[0] for r in conn.execute("SELECT rec FROM records ORDER BY seq")]
    c_nodes = conn.execute("SELECT level, idx, hash FROM merkle_nodes ORDER BY level, idx").fetchall()
    conn.close()
    conn = sqlite3.connect(py_dir / "l.db")
    py_raw = [r[0] for r in conn.execute("SELECT rec FROM records ORDER BY seq")]
    conn.close()
    assert c_raw == py_raw, [i for i, (a, b) in enumerate(zip(c_raw, py_raw, strict=True)) if a != b]
    assert c_nodes == py_nodes and len(c_recs) == len(py_recs) == 9


# --- 4. Python writes, C verifies ------------------------------------------------------------------------------------------------

def test_a_ledger_written_by_the_python_sealer_verifies_in_the_c_core(cvseal, tmp_path):
    from ._anchor_helpers import NEW_SEED, AnchorEnv
    e = AnchorEnv(tmp_path, checkpoint_every=4)
    for i in range(6):
        e.seal(i)
    e.anchor()
    e.sealer.rotate_key(provider(NEW_SEED))
    for i in range(6, 12):
        e.seal(i)
    e.close()
    r = run(cvseal, "verify", e.ledger_path, e.key.public_key.hex())
    assert b"ok:" in r.stdout
    exp = tmp_path / "x.jsonl"
    export_records(e.ledger_path, exp)
    assert run(cvseal, "verify", exp, e.key.public_key.hex()).returncode == 0
    assert run(cvseal, "export", e.ledger_path).stdout == exp.read_bytes()
    # and a tamper is caught by C exactly as by Python
    raw = bytearray(exp.read_bytes())
    raw[len(raw) // 3] ^= 1
    (tmp_path / "bad.jsonl").write_bytes(bytes(raw))
    assert run(cvseal, "verify", tmp_path / "bad.jsonl", e.key.public_key.hex(), check=False).returncode != 0


def test_every_tamper_scenario_that_breaks_integrity_is_rejected_by_the_c_verifier_too(cvseal, tmp_path):
    from attacklab.tamper import SCENARIOS, run_scenario
    caught, skipped = [], []
    for sid in sorted(SCENARIOS):
        if sid in ("T4c", "T12", "T6", "T7a", "T1b", "T15", "T8", "T9", "T11"):
            skipped.append(sid)                              # need an anchor / counter / resolver / DB-only / another trust root
            continue
        fx, o = run_scenario(sid, tmp_path / sid, seed=1)
        pub = fx.key.public_key.hex()
        for form in o.forms:
            path = o.db if form == "db" else o.export
            r = run(cvseal, "verify", path, pub, check=False)
            assert r.returncode != 0, f"{sid}/{form}: the C verifier accepted a tampered ledger"
        caught.append(sid)
    assert len(caught) >= 14 and "T10a" in caught and "T5a" in caught


def test_the_c_core_refuses_to_rotate_to_the_active_key_and_to_open_with_the_wrong_key(cvseal, tmp_path):
    db = _c_ledger(cvseal, tmp_path, n=2)
    assert b"already active" in run(cvseal, "rotate", db, seed_hex(SEED_A), seed_hex(SEED_A), check=False).stderr
    r = run(cvseal, "append", db, seed_hex(SEED_B), "inference", "{}", check=False)
    assert r.returncode != 0 and b"active signing key" in r.stderr
    r = run(cvseal, "append", db, seed_hex(SEED_A), "genesis", "{}", check=False)
    assert r.returncode != 0 and b"genesis" in r.stderr
    assert run(cvseal, "init", db, seed_hex(SEED_A), '{"device_id":"x","unit":"y","profile_hash":"' + "0" * 64 + '","checkpoint_every":9}', check=False).returncode != 0


def test_c_payloads_are_content_addressed_in_the_same_store(cvseal, tmp_path):
    import hashlib
    db = _c_ledger(cvseal, tmp_path, n=1)
    data = b'{"task":"classify","top":[{"cls":1,"conf_e6":900000}]}'
    (tmp_path / "p.bin").write_bytes(data)
    addr = run(cvseal, "payload", db, seed_hex(SEED_A), tmp_path / "p.bin").stdout.decode().strip()
    assert addr == hashlib.sha256(data).hexdigest()
    led = SealedLedger.open(db, read_only=True)
    assert led.get_payload("sha256:" + addr) == data
    led.close()


# --- 5. the bindings ---------------------------------------------------------------------------------------------------------------

def test_the_cpp_binding_builds_runs_and_its_ledger_verifies_in_python(cvseal, tmp_path):
    if not shutil.which("g++"):
        _unavailable("no C++ compiler")
    exe = tmp_path / "test_cvseal"
    r = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(ROOT / "native/cpp/test_cvseal.cpp"), str(NATIVE / "libcvseal.a"),
                        "-lsodium", "-lsqlite3", "-o", str(exe)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    db = tmp_path / "cpp.db"
    out = subprocess.run([str(exe), str(db), provider(SEED_A).public_key.hex()], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    rep = verify_ledger(db, trust_root=_trust(db))
    assert rep.clean and rep.rotations == 1 and rep.checkpoints_verified >= 2, rep.classes()


def test_the_rust_binding_builds_tests_and_its_ledger_verifies_in_python(cvseal, tmp_path):
    if not shutil.which("cargo"):
        _unavailable("no Rust toolchain")
    env = {**__import__("os").environ, "CVSEAL_OUT": str(tmp_path), "CVSEAL_PUBKEY": provider(SEED_A).public_key.hex(),
           "CARGO_TARGET_DIR": str(tmp_path / "target")}
    r = subprocess.run(["cargo", "test", "--offline", "--quiet"], cwd=ROOT / "native/rust/cvseal", env=env, capture_output=True, text=True)
    if r.returncode and ("offline" in r.stderr or "could not" in r.stderr and "registry" in r.stderr):
        pytest.skip("cargo could not build offline")
    assert r.returncode == 0, r.stdout + r.stderr
    db = tmp_path / "rust.db"
    rep = verify_ledger(db, trust_root=_trust(db))
    assert rep.clean and rep.rotations == 1, rep.classes()


# --- review finding 8: a signing seed must not have to live in argv ------------------------------------------------------------------

def test_the_seed_can_come_from_a_file_or_the_environment_and_a_bare_argv_seed_is_warned_about(cvseal, tmp_path):
    import os
    man = json.dumps({"device_id": "s", "unit": "u", "profile_hash": "0" * 64, "checkpoint_every": 1000})
    (tmp_path / "seed").write_text(seed_hex(SEED_A) + "\n")
    quiet = run(cvseal, "init", tmp_path / "a.db", "@" + str(tmp_path / "seed"), man)
    assert quiet.stderr == b""                                                     # no warning: nothing on the command line
    r = subprocess.run([str(cvseal), "append", str(tmp_path / "a.db"), "env:CVSEAL_TEST_SEED", "scan_record",
                        json.dumps({"scan": {"scan_id": "s", "report_sha256": "b" * 64, "profile_hash": "c" * 64,
                                             "code_commit": "0" * 40, "finding_counts": {"info": 1}}})],
                       capture_output=True, env={**os.environ, "CVSEAL_TEST_SEED": seed_hex(SEED_A)})
    assert r.returncode == 0 and r.stderr == b""
    bare = run(cvseal, "keyid", seed_hex(SEED_A))
    assert b"visible in ps" in bare.stderr and bare.stdout.split()[0].decode() == provider(SEED_A).key_id


@pytest.mark.parametrize("bad", ["A" * 64, "+" + "1" * 63, " " + "1" * 63, "1" * 63, "1" * 65, "0x" + "1" * 62, "g" * 64, ""])
def test_a_malformed_seed_is_refused_not_forgiven(cvseal, bad):
    assert run(cvseal, "keyid", bad, check=False).returncode != 0


def test_a_missing_seed_file_or_variable_is_refused(cvseal, tmp_path):
    assert run(cvseal, "keyid", "@" + str(tmp_path / "nope"), check=False).returncode != 0
    assert run(cvseal, "keyid", "env:CVSEAL_DOES_NOT_EXIST", check=False).returncode != 0


# --- final review N2 / N3: the C core enforces the same header and calendar rules as Python ----------------------------------------

TIMESTAMPS = ["2026-09-19T02:00:00.000007Z", "2024-02-29T23:59:59.999999Z", "2000-02-29T00:00:00.000000Z", "0001-01-01T00:00:00.000000Z",
              "2026-13-45T99:99:99.000000Z", "2026-13-01T00:00:00.000000Z", "2026-00-10T00:00:00.000000Z", "2026-02-29T00:00:00.000000Z",
              "1900-02-29T00:00:00.000000Z", "2026-04-31T00:00:00.000000Z", "2026-09-19T24:00:00.000000Z", "2026-09-19T00:60:00.000000Z",
              "2026-09-19T00:00:60.000000Z", "2026-09-00T00:00:00.000000Z", "0000-01-01T00:00:00.000000Z", "2026-09-19 02:00:00.000000Z",
              "2026-09-19T02:00:00Z", "2026-09-19T02:00:00.00000Z", "2026-9-19T02:00:00.000000Z"]


def _py_ts_ok(ts: str) -> bool:
    from cva.provenance.seal.records import _ts
    try:
        _ts(ts, "$")
        return True
    except InvalidRecordError:
        return False


from cva.provenance.seal.errors import InvalidRecord as InvalidRecordError  # noqa: E402


@pytest.mark.parametrize("ts", TIMESTAMPS)
def test_c_and_python_agree_on_which_timestamps_are_real_calendar_times(cvseal, tmp_path, ts):
    man = json.dumps({"device_id": "t", "unit": "u", "profile_hash": "0" * 64, "checkpoint_every": 1000})
    r = run(cvseal, "init", tmp_path / f"{abs(hash(ts))}.db", seed_hex(SEED_A), man, ts, "00" * 16, check=False)
    assert (r.returncode == 0) == _py_ts_ok(ts), (ts, r.stderr)


def _mutate_header(field_edit):
    """The frozen export with record 3 edited (its signature is now wrong too — the header rules must fire FIRST)."""
    r = json.loads(VEC["records"][3])
    field_edit(r)
    line = json.dumps(r, sort_keys=True, separators=(",", ":"))
    return "".join(x + "\n" for x in [*VEC["records"][:3], line, *VEC["records"][4:]]).encode()


@pytest.mark.parametrize("name,edit,message", [
    ("version", lambda r: r.__setitem__("v", "cva-seal/2"), b"v must be"),
    ("short nonce", lambda r: r.__setitem__("nonce", r["nonce"][:-2]), b"lowercase hex"),
    ("upper nonce", lambda r: r.__setitem__("nonce", r["nonce"].upper() if r["nonce"] != r["nonce"].upper() else "A" * 32), b"lowercase hex"),
    ("bad calendar", lambda r: r.__setitem__("created_at_utc", "2026-13-45T99:99:99.000000Z"), b"real calendar time"),
    ("extra member", lambda r: r.__setitem__("extra", 1), b"exactly the header fields"),
    ("missing section", lambda r: r.pop("input"), b"exactly the header fields"),
    ("negative seq", lambda r: r.__setitem__("seq", -1), b"is not its position"),
])
def test_the_c_verifier_enforces_the_header_rules_it_claims_and_names_the_rule(cvseal, tmp_path, name, edit, message):
    (tmp_path / "h.jsonl").write_bytes(_mutate_header(edit))
    r = run(cvseal, "verify", tmp_path / "h.jsonl", VEC["keys"]["ledger"]["public_key"], check=False)
    assert r.returncode != 0 and message in r.stderr and b"record 3" in r.stderr, (name, r.stderr)
