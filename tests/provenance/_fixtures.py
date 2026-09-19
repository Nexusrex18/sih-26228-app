"""Valid example sections and records for every type (values are arbitrary but well-formed)."""
from __future__ import annotations

import copy
import hashlib
from typing import Any

from cva.provenance.seal.records import build_record, genesis_prev_hash


def H(c): return c * 64
KEY_ID = "1" * 64
PUB = "5" * 64                                # a 32-byte value; only its SHA-256 matters for key_id_of
SIG = "ab" * 64
NONCE = "cd" * 16
TS = "2026-09-19T02:14:07.482913Z"
COMMIT = "0" * 40

MODEL = {"id": "resnet50-v3", "weights_sha256": H("a"), "format": "onnx", "arch_hash": H("b")}
CONFIG = {"preprocess_hash": H("c"), "preprocess_ref": "sha256:" + H("d"), "postprocess_hash": H("e"),
          "runtime": "onnxruntime 1.17.1 / CPUExecutionProvider", "version_pins_hash": H("f"),
          "code_commit": COMMIT}
INPUT = {"sha256": H("1"), "source_kind": "encoded_file", "phash": None,
         "phash_omitted_reason": "not_computed", "dims": [1920, 1080]}
OUTPUT = {"jcs_sha256": H("2"), "decision_sha256": H("3"), "raw_jcs_sha256": H("4"),
          "payload_ref": "sha256:" + H("2")}          # the ref must name the payload jcs_sha256 commits to
MANIFEST = {"device_id": "jetson-07", "key_id": KEY_ID, "profile_hash": H("7"), "unit": "alpha-coy",
            "checkpoint_every": 1000, "spec": "cva-seal/1"}
ANALYST_OVERRIDE = {
    "actor_id": "a.sharma", "role": "analyst", "action": "override", "scan_id": "s-2026-09-19-0007",
    "target_type": "sample", "target_ref": "4471", "finding_id": "9f3c1a", "new_disposition": "review",
    "reason_code": "quality_issue", "justification": "Duplicate%20cluster%20confirmed",
    "expected_prev_seq": 0, "request_id": "9" * 32}

BODIES: dict[str, dict[str, Any]] = {
    "genesis": {"deployment_manifest": MANIFEST},
    "model_registration": {"model": MODEL, "config": CONFIG},
    "inference": {"input": INPUT, "model": MODEL, "config": CONFIG, "output": OUTPUT},
    "checkpoint": {"checkpoint": {"tree_size": 1000, "root_hash": H("8")}},
    "anchor_event": {"anchor": {"checkpoint_seq": 1001, "tree_size": 1000, "root_hash": H("8"),
                                "cosigner_key_ids": [H("9")], "medium": "cosign", "label": "shift-change"}},
    "key_rotation": {"rotation": {"new_key_id": hashlib.sha256(bytes.fromhex(PUB)).hexdigest(),
                                  "new_public_key": PUB, "effective_seq": 8, "new_key_pop": SIG}},
    "degraded_marker": {"gap": {"first_unsealed_utc": TS, "last_unsealed_utc": "2026-09-19T02:15:00.000000Z",
                                "reported_count": 12, "reason": "ledger_unwritable", "spill_sha256": H("a")}},
    "scan_record": {"scan": {"scan_id": "s-2026-09-19-0007", "report_sha256": H("b"), "profile_hash": H("c"),
                             "code_commit": COMMIT, "finding_counts": {"critical": 1, "info": 3}}},
    "analyst_event": {"analyst": ANALYST_OVERRIDE},
}
SEQ = {"genesis": 0, "key_rotation": 7, "anchor_event": 1002}      # others default to 5


def make_record(rtype: str, *, signed: bool = True, **body_overrides: Any) -> dict[str, Any]:
    body = copy.deepcopy(BODIES[rtype])
    body.update(copy.deepcopy(body_overrides))
    seq = SEQ.get(rtype, 5)
    prev = genesis_prev_hash(body["deployment_manifest"]) if rtype == "genesis" else H("2")
    r = build_record(rtype, seq=seq, prev_record_hash=prev, key_id=KEY_ID, created_at_utc=TS,
                     nonce=NONCE, body=body)
    if signed:
        r["signature"] = SIG
    return r
