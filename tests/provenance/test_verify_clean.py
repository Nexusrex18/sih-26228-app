"""The zero-false-alarm gate (plan §11.5; P2): 10,000 clean records, across real process restarts, with a
backwards clock step mid-run. The verifier must report NOTHING above `info`.

Why real restarts: the design has deliberately NO in-memory counter that could disagree with the database after
a reboot (C-11). Only separate processes re-opening the ledger prove that. The clock step proves NTP cannot
raise a false alarm (`clock_regression` is information, never a tamper signal).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from cva.provenance.seal.keys import TrustKey, TrustRoot
from cva.provenance.seal.records import genesis_prev_hash
from cva.provenance.seal.store import SealedLedger
from cva.provenance.seal.verify import export_records, verify_ledger

from ._chain_helpers import SEED_A, provider
from ._ledger_helpers import MANIFEST

REPO = Path(__file__).resolve().parents[2]
RUNS, PER_RUN = 3, 3400

CHILD = """
import sys, base64, hashlib
sys.path.insert(0, {repo!r})
from datetime import UTC, datetime, timedelta
from cva.provenance.seal.keys import EnvKeyProvider
from cva.provenance.seal.sealer import Sealer
key = EnvKeyProvider("K", environ={{"K": base64.b64encode(bytes(range(32))).decode()}})
RUN, N, BACK = {run}, {n}, {back}
t0 = datetime(2026, 9, 19, 2, 0, 0, tzinfo=UTC) + timedelta(hours=RUN)
calls = [0]
def clock():
    calls[0] += 1
    k = calls[0] - (1000 if (BACK and calls[0] >= BACK) else 0)        # ONE backwards step, then time moves on
    return t0 + timedelta(milliseconds=5 * k)
with Sealer.open({ledger!r}, key=key, trust_root={trust!r}, clock=clock) as s:
    m = s.register_model(id="clean-model", weights_sha256="ab" * 32, arch_hash="cd" * 32, format="onnx")
    c = s.register_config(preprocess_spec={{"mean_e6": [485000]}}, postprocess_spec={{"t": 1}}, runtime="ci 1",
                          version_pins_hash="ef" * 32, code_commit="0" * 40)
    for i in range(N):
        frame = hashlib.sha256(f"{{RUN}}/{{i}}".encode()).digest()
        s.seal(frame, m, c, output={{"task": "classify", "top": [{{"cls": i % 10, "conf": (i % 97) / 100}}]}},
               dims=(32, 32))
"""


def test_ten_thousand_clean_records_across_three_restarts_raise_no_alarm(tmp_path):
    key = provider(SEED_A)
    ledger, trust_path = tmp_path / "ledger.db", tmp_path / "trust_root.json"
    from datetime import UTC, datetime
    led = SealedLedger.init_ledger(ledger, key, {**MANIFEST, "checkpoint_every": 1000}, durability="group_commit",
                                   clock=lambda: datetime(2026, 9, 19, 1, 0, 0, tzinfo=UTC))     # before every child's clock
    trust = TrustRoot(genesis_prev_hash(led.deployment_manifest), (TrustKey(key.key_id, key.public_key, "ledger"),))
    led.close()
    trust_path.write_bytes(trust.to_bytes() + b"\n")

    for run in range(RUNS):                                                     # three separate processes
        back = 1500 if run == 1 else 0                                          # a backwards clock step in the middle one
        p = subprocess.run([sys.executable, "-c", CHILD.format(repo=str(REPO), run=run, n=PER_RUN, back=back,
                                                                ledger=str(ledger), trust=str(trust_path))],
                           cwd=REPO, capture_output=True, text=True, timeout=600, check=False)
        assert p.returncode == 0, p.stderr

    t0 = time.perf_counter()
    rep = verify_ledger(ledger, trust_root=trust)
    elapsed = time.perf_counter() - t0
    total = RUNS * PER_RUN
    assert rep.count_by_type["inference"] == total >= 10_000
    assert [f for f in rep.findings if f.severity != "info"] == [], [(f.attack_class, f.seq) for f in rep.findings[:5]]
    assert rep.clean
    regress = [f for f in rep.findings if f.attack_class == "clock_regression"]
    assert len(regress) == 1 and regress[0].severity == "info"                # the step is noticed, and harmless
    assert rep.checkpoints_verified == rep.count_by_type["checkpoint"] >= 10
    assert rep.payloads_missing == 0 and rep.payloads_checked > 0
    assert rep.count_by_type["model_registration"] == 1                       # restarts did not re-register
    assert elapsed < 60, f"verification took {elapsed:.1f}s for {rep.records_checked} records"

    export_records(ledger, tmp_path / "ledger.jsonl")                         # ... and the export says the same
    ex = verify_ledger(tmp_path / "ledger.jsonl", trust_root=trust)
    assert ex.clean and ex.records_checked == rep.records_checked and ex.checkpoints_verified == rep.checkpoints_verified


def test_three_restarts_leave_a_gapless_sequence_with_no_reset_anywhere(tmp_path):
    """The failure the removed monotonic counter would have caused: a reset at each reboot."""
    key = provider(SEED_A)
    ledger, trust_path = tmp_path / "l.db", tmp_path / "t.json"
    led = SealedLedger.init_ledger(ledger, key, {**MANIFEST, "checkpoint_every": 100})
    trust = TrustRoot(genesis_prev_hash(led.deployment_manifest), (TrustKey(key.key_id, key.public_key, "ledger"),))
    led.close()
    trust_path.write_bytes(trust.to_bytes() + b"\n")
    for run in range(3):
        p = subprocess.run([sys.executable, "-c", CHILD.format(repo=str(REPO), run=run, n=60, back=0,
                                                                ledger=str(ledger), trust=str(trust_path))],
                           cwd=REPO, capture_output=True, text=True, timeout=300, check=False)
        assert p.returncode == 0, p.stderr
    ro = SealedLedger.open(ledger, read_only=True)
    seqs = [r["seq"] for r in ro.records()]
    assert seqs == list(range(len(seqs))) and verify_ledger(ledger, trust_root=trust).clean
    assert pytest is not None
