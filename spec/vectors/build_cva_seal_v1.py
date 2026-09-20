"""Generate `cva_seal_v1.json`: OUR frozen vectors (plan §11.8) — records, links, checkpoints, roots, proofs, an
anchor, a rotation, output hashes and quantisation cases, from a fixed key, a fixed clock and a fixed nonce stream.

The output is a function of nothing but this file and the reference implementation, so it is byte-identical on
every machine and Python version; `tests/provenance/test_vectors_frozen.py` regenerates it and compares it with
the frozen file, and (where a second interpreter has the dependencies) runs the same generator there.

WHAT THAT PROVES, AND WHAT IT DOES NOT. These vectors are generated FROM the reference implementation, so regenerating
and comparing them proves DETERMINISM, not conformance to the specification: a divergence between the reference and
`spec/cva-seal-spec-v1.md` would be frozen into the file as if it were correct, and `test_vectors_frozen` would still
pass. Conformance to the wire format is evidenced separately, by a reader written from the spec text alone that
reproduced every check in the file (see `docs/provenance/VERIFICATION-PROCEDURE.md` §2) — and by the independent
verifier, which shares an author with the reference and so shows only that the spec is self-consistent. Do not read a
green `test_vectors_frozen` as "the spec is right".

Run from the repo root to (re)freeze:  .venv/bin/python spec/vectors/build_cva_seal_v1.py --write
A change to the frozen file is a change to the wire format: it needs a decision entry, not a quiet regeneration.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cva.provenance.seal import anchor as A  # noqa: E402
from cva.provenance.seal.canonical import canonical_bytes  # noqa: E402
from cva.provenance.seal.keys import EnvKeyProvider, TrustKey, TrustRoot  # noqa: E402
from cva.provenance.seal.merkle import MerkleTree, leaf_hash  # noqa: E402
from cva.provenance.seal.outputs import build_output_objects  # noqa: E402
from cva.provenance.seal.proof import make_inclusion_proof  # noqa: E402
from cva.provenance.seal.quantise import q_box64, q_conf, q_px  # noqa: E402
from cva.provenance.seal.records import genesis_prev_hash  # noqa: E402
from cva.provenance.seal.sealer import Sealer  # noqa: E402
from cva.provenance.seal.store import SealedLedger  # noqa: E402
from cva.provenance.seal.verify import export_records, verify_ledger  # noqa: E402

OUT = Path(__file__).resolve().parent / "cva_seal_v1.json"
T0 = datetime(2026, 9, 19, 2, 0, 0, tzinfo=UTC)
MANIFEST = {"device_id": "vector-device-01", "unit": "vector-unit", "profile_hash": "7" * 64, "checkpoint_every": 8}


def _key(seed: bytes) -> EnvKeyProvider:
    return EnvKeyProvider("K", environ={"K": base64.b64encode(seed).decode()})


def _seeds() -> dict[str, bytes]:
    return {name: hashlib.sha256(f"cva-seal-v1-vectors/{name}".encode()).digest()
            for name in ("ledger", "witness", "boundary", "next")}


def _clock():
    t = [T0]

    def tick() -> datetime:
        t[0] += timedelta(milliseconds=7)
        return t[0]
    return tick


def _rng():
    n = [0]

    def draw(k: int) -> bytes:
        n[0] += 1
        return hashlib.sha256(b"nonce/" + n[0].to_bytes(4, "big")).digest()[:k]
    return draw


QUANTISE_CONF = [0.0, 1.0, 0.5, 0.25, 0.993118, 0.8712044, 1.5e-6, 2.5e-7, 4.9999999e-7, 0.9999995, 0.0000005]
QUANTISE_PX = [0.0, 0.0078125, 0.5, 1.5, 2.5, 18.8125, 51.874999, 100.0, 140.5, 0.00390625]
CLASSIFY = {"task": "classify", "top": [{"cls": 7, "conf": 0.993118}, {"cls": 2, "conf": 0.004112}]}
DETECT_RAW = {"task": "detect", "detections": [
    {"cls": 3, "conf": 0.871204, "box": [18.8125, 13.75, 51.875, 37.671875]},
    {"cls": 1, "conf": 0.31, "box": [100.0, 100.0, 140.5, 160.25]},
    {"cls": 1, "conf": 0.12, "box": [5.0, 5.0, 9.0, 9.0]}]}
DETECT_FILTERED = {"task": "detect", "detections": DETECT_RAW["detections"][:2],
                   "filter": {"conf_thr": 0.25, "nms_iou": 0.45}}
GAP = {"gap": {"first_unsealed_utc": "2026-09-19T03:00:00.000000Z", "last_unsealed_utc": "2026-09-19T03:00:05.000000Z",
               "reported_count": 12, "reason": "ledger_unwritable", "spill_sha256": "a" * 64}}


def build() -> dict:
    seeds = _seeds()
    keys = {n: _key(s) for n, s in seeds.items()}
    clk, rng = _clock(), _rng()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ledger.db"
        led = SealedLedger.init_ledger(path, keys["ledger"], MANIFEST, clock=clk, rng=rng)
        manifest = dict(led.deployment_manifest)
        led.close()
        trust = TrustRoot(genesis_prev_hash(manifest), tuple(
            TrustKey(keys[n].key_id, keys[n].public_key, role)
            for n, role in (("ledger", "ledger"), ("witness", "witness"), ("boundary", "boundary"))))
        inputs: dict[int, str] = {}
        with Sealer.open(path, key=keys["ledger"], trust_root=trust, clock=clk, rng=rng, background_flush=False) as s:
            m = s.register_model(id="resnet50-v3", weights_sha256="ab" * 32, arch_hash="cd" * 32, format="onnx")
            c = s.register_config(preprocess_spec={"mean_e6": [485000, 456000, 406000], "std_e6": [229000, 224000, 225000]},
                                  postprocess_spec={"conf_thr_e6": 250000, "nms_iou_e6": 450000},
                                  runtime="onnxruntime 1.17.1 / CPUExecutionProvider", version_pins_hash="ef" * 32,
                                  code_commit="0" * 40)
            for i in range(5):
                buf = hashlib.sha256(f"input/{i}".encode()).digest() * 16
                r = s.seal(buf, m, c, output=CLASSIFY, dims=(64, 64))
                inputs[r.seq] = buf.hex()
            for i in range(5, 9):
                buf = hashlib.sha256(f"input/{i}".encode()).digest() * 16
                r = s.seal(buf, m, c, output=DETECT_RAW, filtered=DETECT_FILTERED, dims=(1920, 1080))
                inputs[r.seq] = buf.hex()
        with SealedLedger.open(path, key=keys["ledger"], clock=clk, rng=rng, background_flush=False) as led:
            led.append_typed("degraded_marker", GAP)
            led.rotate_key(keys["next"])
        with Sealer.open(path, key=keys["next"], trust_root=trust, clock=clk, rng=rng, background_flush=False) as s:
            m = s.register_model(id="resnet50-v3", weights_sha256="ab" * 32, arch_hash="cd" * 32, format="onnx")
            c = s.register_config(preprocess_spec={"mean_e6": [485000, 456000, 406000], "std_e6": [229000, 224000, 225000]},
                                  postprocess_spec={"conf_thr_e6": 250000, "nms_iou_e6": 450000},
                                  runtime="onnxruntime 1.17.1 / CPUExecutionProvider", version_pins_hash="ef" * 32,
                                  code_commit="0" * 40)
            for i in range(9, 12):
                buf = hashlib.sha256(f"input/{i}".encode()).digest() * 16
                r = s.seal(buf, m, c, output=CLASSIFY, dims=(64, 64))
                inputs[r.seq] = buf.hex()
        with SealedLedger.open(path, key=keys["next"], clock=clk, rng=rng, background_flush=False) as led:
            anchor = A.export_anchor(led, Path(d) / "a.json", medium="cosign", label="frozen",
                                     cosigner_key_ids=(keys["witness"].key_id,))
        anchor = A.attest(A.cosign(anchor, keys["witness"]), keys["boundary"], "2026-09-19T03:30:00.000000Z")
        export_records(path, Path(d) / "x.jsonl")
        export = (Path(d) / "x.jsonl").read_bytes()
        rows = export.split(b"\n")[:-1]
        proof = make_inclusion_proof(export, 4, anchor)
        rep = verify_ledger(export, trust_root=trust, anchors=[anchor])
    tree = MerkleTree.from_leaves([leaf_hash(r) for r in rows])
    recs = [json.loads(r) for r in rows]
    fine_vectors = []
    for name, raw, fil in (("classify", CLASSIFY, None), ("detect_filtered", DETECT_RAW, DETECT_FILTERED)):
        r_o, f_o, c_o = build_output_objects(raw, fil)
        fine_vectors.append({
            "name": name, "raw": raw, "filtered": fil,
            "raw_object": canonical_bytes(r_o, max_bytes=None).decode(), "fine_object": canonical_bytes(f_o, max_bytes=None).decode(),
            "coarse_object": canonical_bytes(c_o, max_bytes=None).decode(),
            "raw_jcs_sha256": hashlib.sha256(canonical_bytes(r_o, max_bytes=None)).hexdigest(),
            "jcs_sha256": hashlib.sha256(canonical_bytes(f_o, max_bytes=None)).hexdigest(),
            "decision_sha256": hashlib.sha256(canonical_bytes(c_o, max_bytes=None)).hexdigest()})
    return {
        "spec": "cva-seal/1", "frozen": True,
        "note": "Generated by spec/vectors/build_cva_seal_v1.py from the reference implementation. A change here is a "
                "wire-format change and needs a decision entry.",
        "keys": {n: {"seed": seeds[n].hex(), "public_key": keys[n].public_key.hex(), "key_id": keys[n].key_id}
                 for n in seeds},
        "trust_root": json.loads(trust.to_bytes()),
        "deployment_manifest": manifest,
        "quantise": {"conf_e6": [{"in": x, "out": q_conf(x)} for x in QUANTISE_CONF],
                     "box_q64": [{"in": x, "out": q_box64(x)} for x in QUANTISE_PX],
                     "box_px": [{"in": x, "out": q_px(x)} for x in QUANTISE_PX]},
        "outputs": fine_vectors,
        "inputs_hex": {str(k): v for k, v in inputs.items()},
        "records": [r.decode("ascii") for r in rows],
        "record_hashes": [hashlib.sha256(canonical_bytes({k: v for k, v in r.items() if k != "signature"})).hexdigest()
                          for r in recs],
        "leaf_hashes": [leaf_hash(r).hex() for r in rows],
        "roots": {str(n): tree.root(n).hex() for n in range(1, len(rows) + 1)},
        "anchor": anchor,
        "inclusion_proof": proof,
        "verify": {"records": len(rows), "checkpoints_verified": rep.checkpoints_verified, "rotations": rep.rotations,
                   "anchors_verified": rep.anchors_verified, "clean": rep.clean,
                   "declared_gaps": rep.declared_gaps, "anchored_records": rep.anchored_records,
                   "unwitnessed_records": rep.unwitnessed_records, "sealed_not_after_utc": rep.attested_not_after},
    }


def encode(doc: dict) -> bytes:
    return json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=True).encode("ascii") + b"\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="overwrite the frozen file")
    ap.add_argument("--stdout", action="store_true")
    a = ap.parse_args()
    data = encode(build())
    if a.stdout:
        sys.stdout.buffer.write(data)
    elif a.write:
        OUT.write_bytes(data)
        print(f"wrote {OUT} ({len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()})")
    else:
        same = OUT.read_bytes() == data
        print("frozen file matches" if same else "DIFFERS from the frozen file")
        return 0 if same else 1          # a CI matrix cell per Python version turns this into cross-version byte identity
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
