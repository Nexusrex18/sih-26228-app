"""Gate C8: a verifier written from the spec alone agrees with the reference (plan §11.9).

`spec/independent_verifier.py` imports nothing from `cva` (asserted below). It must (1) reproduce every frozen
vector, (2) verify our clean ledgers and (3) return the same findings as the reference on every tamper artefact and
on a large sample of single-bit mutations. A disagreement is a bug in one of the two or a hole in the spec.
"""
from __future__ import annotations

import ast
import hashlib
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "spec"))
import independent_verifier as iv  # noqa: E402

from attacklab.tamper import SCENARIOS, run_scenario  # noqa: E402
from cva.provenance.seal.keys import parse_trust_root  # noqa: E402
from cva.provenance.seal.verify import (  # noqa: E402
    export_payloads,
    export_records,
    load_payloads,
    verify_ledger,
)

VEC = json.loads((ROOT / "spec/vectors/cva_seal_v1.json").read_text())
EXPORT = "".join(r + "\n" for r in VEC["records"]).encode()
TRUST = VEC["trust_root"]


def signature(rep) -> list[tuple]:
    """What two verifiers must agree on: the ordered classes above info, and the first finding's seq/severity/nature."""
    out = [(f.cls, f.seq, f.severity, f.nature) for f in rep.above_info]
    return out


def ref_signature(rep) -> list[tuple]:
    return [(f.attack_class, f.seq, f.severity, f.nature) for f in rep.findings if f.severity != "info"]


# --- independence ----------------------------------------------------------------------------------------------

def test_the_independent_verifier_imports_nothing_from_the_reference_implementation():
    tree = ast.parse((ROOT / "spec/independent_verifier.py").read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert mods <= {"__future__", "argparse", "base64", "hashlib", "json", "re", "sys", "dataclasses", "datetime",
                    "typing", "collections", "cryptography"}, mods


# --- the frozen vectors ----------------------------------------------------------------------------------------

def test_quantisation_vectors():
    for row in VEC["quantise"]["conf_e6"]:
        assert iv.q_conf(row["in"]) == row["out"]
    for row in VEC["quantise"]["box_q64"]:
        assert iv.q_box64(row["in"]) == row["out"]
    for row in VEC["quantise"]["box_px"]:
        assert iv.q_px(row["in"]) == row["out"]


def test_the_half_way_cases_that_round_would_get_wrong():
    assert iv.q_px(0.5) == 1 and iv.q_px(1.5) == 2 and iv.q_px(2.5) == 3        # round() gives 0, 2, 2
    assert iv.q_box64(0.0078125) == 1                                            # exactly 1/128 px


def test_output_objects_and_their_three_hashes():
    for v in VEC["outputs"]:
        raw_b, fine_b, coarse_b = iv.output_objects(v["raw"], v["filtered"])
        assert raw_b.decode() == v["raw_object"] and fine_b.decode() == v["fine_object"] and coarse_b.decode() == v["coarse_object"]
        assert hashlib.sha256(raw_b).hexdigest() == v["raw_jcs_sha256"]
        assert hashlib.sha256(fine_b).hexdigest() == v["jcs_sha256"]
        assert hashlib.sha256(coarse_b).hexdigest() == v["decision_sha256"]


def test_record_hashes_leaves_and_every_prefix_root():
    rows = [r.encode() for r in VEC["records"]]
    leaves = [iv.leaf(r) for r in rows]
    for i, r in enumerate(rows):
        obj = iv.strict_parse(r)
        iv.validate(obj)
        assert iv.sha(iv.unsigned_canon(obj)).hex() == VEC["record_hashes"][i]
        assert leaves[i].hex() == VEC["leaf_hashes"][i]
    for n in range(1, len(rows) + 1):
        assert iv.mth(leaves[:n]).hex() == VEC["roots"][str(n)]


def test_every_signature_in_the_vectors_verifies_under_the_key_the_chain_says_is_active():
    keys = {k: bytes.fromhex(v["public_key"]) for k, v in VEC["keys"].items()}
    by_id = {v["key_id"]: keys[k] for k, v in VEC["keys"].items()}
    for r in VEC["records"]:
        obj = iv.strict_parse(r.encode())
        assert iv.ed_verify(by_id[obj["key_id"]], iv.TAG_RECORD + iv.unsigned_canon(obj), bytes.fromhex(obj["signature"]))


def test_the_vector_ledger_verifies_with_its_anchor_and_matches_the_expected_summary():
    rep = iv.verify(EXPORT, TRUST, anchors=[VEC["anchor"]])
    want = VEC["verify"]
    assert [f.cls for f in rep.above_info] == ["degraded_gap"]                 # a declared hole is the only finding
    assert (rep.records, rep.rotations, rep.anchors_verified, rep.anchored_records, rep.unwitnessed, rep.time_bound) == (
        want["records"], want["rotations"], want["anchors_verified"], want["anchored_records"],
        want["unwitnessed_records"], want["sealed_not_after_utc"])


def test_the_inclusion_proof_verifies_from_the_anchor_alone():
    p, a = VEC["inclusion_proof"], VEC["anchor"]
    root = bytes.fromhex(a["checkpoint"]["checkpoint"]["root_hash"])
    size = a["checkpoint"]["checkpoint"]["tree_size"]
    assert p["tree_size"] == size and p["root_hash"] == root.hex()
    assert iv.verify_inclusion(iv.leaf(p["record"].encode()), p["seq"], size, [bytes.fromhex(x) for x in p["path"]], root)
    assert not iv.verify_inclusion(iv.leaf(p["record"].encode()), p["seq"] + 1, size, [bytes.fromhex(x) for x in p["path"]], root)


def test_the_independent_and_reference_verifiers_agree_on_the_vector_ledger():
    ref = verify_ledger(EXPORT, trust_root=parse_trust_root(json.dumps(TRUST).encode()), anchors=[VEC["anchor"]])
    assert signature(iv.verify(EXPORT, TRUST, anchors=[VEC["anchor"]])) == ref_signature(ref)


# --- the tamper matrix ---------------------------------------------------------------------------------------------

def _artefacts(outcome, tmp: Path):
    """(export bytes, payloads or None) for each form the outcome has."""
    if outcome.export is not None:
        yield "export", outcome.export.read_bytes(), None
    elif outcome.db is not None:
        export_records(outcome.db, tmp / "from_db.jsonl")
        export_payloads(outcome.db, tmp / "from_db.payloads")
        yield "db->export", (tmp / "from_db.jsonl").read_bytes(), load_payloads(tmp / "from_db.payloads")


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_the_independent_verifier_returns_what_the_reference_returns_on_every_tamper_artefact(tmp_path, scenario, seed):
    fx, o = run_scenario(scenario, tmp_path, seed=seed)
    trust = json.loads(o.trust.to_bytes())
    for form, data, payloads in _artefacts(o, tmp_path):
        kw = dict(o.verify_kwargs)
        anchors = kw.pop("anchors", [])
        ref = verify_ledger(data, trust_root=o.trust, payloads=payloads, anchors=anchors, **kw)
        ind = iv.verify(data, trust, anchors=[json.loads(Path(a).read_bytes()) if isinstance(a, Path) else a for a in anchors],
                        payloads=payloads, reference_manifest=kw.get("reference_manifest"),
                        input_resolver=kw.get("input_resolver"), expected_count=kw.get("expected_count"))
        assert signature(ind) == ref_signature(ref), f"{scenario}/{form}/seed{seed}"
        if form == "export":
            assert [f.cls for f in ind.above_info] == list(o.expected.classes), f"{scenario}: independent vs the lab's expectation"


@pytest.mark.parametrize("scenario", ["T4c", "T12"])
def test_the_undetectable_scenarios_are_silent_for_the_independent_verifier_too_until_it_is_given_a_reference(tmp_path, scenario):
    fx, o = run_scenario(scenario, tmp_path, seed=2)
    export_records(o.db, tmp_path / "x.jsonl")
    data, trust = (tmp_path / "x.jsonl").read_bytes(), json.loads(o.trust.to_bytes())
    assert iv.verify(data, trust).classes == []
    kw = o.reconcile_kwargs
    assert iv.verify(data, trust, expected_count=kw["expected_count"]).classes == ["ledger_incomplete"]
    for _name, exp, kwargs in o.reconcile_more:
        anchors = [json.loads(Path(a).read_bytes()) for a in kwargs["anchors"]]
        assert iv.verify(data, trust, anchors=anchors).classes == list(exp.classes)


# --- clean ledgers with every feature -------------------------------------------------------------------------------

def test_a_rotated_anchored_ledger_verifies_clean_with_or_without_the_external_anchor(tmp_path):
    from attacklab.tamper import build_clean_ledger
    fx = build_clean_ledger(tmp_path, n=40, seed=3, checkpoint_every=9, rotate_at=15, anchor=True)
    export_records(fx.ledger, tmp_path / "x.jsonl")
    data, trust = (tmp_path / "x.jsonl").read_bytes(), json.loads(fx.trust.to_bytes())
    anchor = json.loads(fx.anchor.read_bytes())
    rep = iv.verify(data, trust, anchors=[anchor])
    assert rep.above_info == [] and rep.rotations == 1 and rep.anchors_verified == 1
    bare = iv.verify(data, trust)
    assert bare.above_info == [] and bare.anchors_verified == 0 and bare.anchored_records == 0
    assert rep.anchored_records == 49                 # the anchor fixes tree + its checkpoint; the anchor_event follows it


# --- mutations: the two verifiers must never disagree ---------------------------------------------------------------------

def test_the_two_verifiers_agree_on_many_single_bit_mutations_of_the_vector_ledger():
    trust = parse_trust_root(json.dumps(TRUST).encode())
    rnd = random.Random(20260919)
    n_bytes = len(EXPORT)
    positions = sorted(set([0, n_bytes - 1, n_bytes - 2] + [rnd.randrange(n_bytes) for _ in range(900)]))
    disagreements = []
    for i in positions:
        bit = rnd.randrange(8)
        b = bytearray(EXPORT)
        b[i] ^= 1 << bit
        ref = ref_signature(verify_ledger(bytes(b), trust_root=trust, anchors=[VEC["anchor"]]))
        ind = signature(iv.verify(bytes(b), TRUST, anchors=[VEC["anchor"]]))
        if ref != ind:
            disagreements.append((i, bit, ref[:2], ind[:2]))
    assert disagreements == [], f"{len(disagreements)} disagreements, e.g. {disagreements[:3]}"


def test_the_two_verifiers_agree_on_structural_mutations_of_the_vector_ledger():
    """Delete, duplicate, swap, truncate, blank-line and reorder whole lines."""
    trust = parse_trust_root(json.dumps(TRUST).encode())
    lines = EXPORT.split(b"\n")[:-1]
    cases = {}
    for k in range(1, len(lines) - 1):
        cases[f"delete{k}"] = lines[:k] + lines[k + 1:]
        cases[f"dup{k}"] = lines[:k + 1] + [lines[k]] + lines[k + 1:]
        cases[f"dup_tail{k}"] = lines + [lines[k]]
    for a, b in ((2, 5), (3, 4), (1, 8), (6, 12), (9, 15)):
        sw = list(lines)
        sw[a], sw[b] = sw[b], sw[a]
        cases[f"swap{a}_{b}"] = sw
    for keep in (1, 2, 5, 11, 19):
        cases[f"prefix{keep}"] = lines[:keep]
    cases["blank_line"] = lines[:4] + [b""] + lines[4:]
    cases["reversed"] = lines[::-1]
    bad = []
    for name, ls in cases.items():
        data = b"".join(x + b"\n" for x in ls)
        for anchors in ([], [VEC["anchor"]]):
            ref = ref_signature(verify_ledger(data, trust_root=trust, anchors=anchors))
            ind = signature(iv.verify(data, TRUST, anchors=anchors))
            if ref != ind:
                bad.append((name, bool(anchors), ref[:3], ind[:3]))
    assert bad == [], f"{len(bad)} disagreements, e.g. {bad[:3]}"
    assert len(cases) > 60


def test_an_export_missing_its_final_newline_and_an_empty_export_are_handled_alike():
    trust = parse_trust_root(json.dumps(TRUST).encode())
    for data in (EXPORT[:-1], b"", b"\n"):
        assert signature(iv.verify(data, TRUST)) == ref_signature(verify_ledger(data, trust_root=trust))


# --- the two rules the tamper lab does not reach on its own -------------------------------------------------------------

def test_a_nonce_may_repeat_across_an_anchor_but_not_within_an_interval_for_both_verifiers():
    from ._ledger_helpers import export_bytes, valid_chain

    def factory(dup_at):
        state = {"n": 0}

        def draw(k):
            state["n"] += 1
            if state["n"] == 3:
                return bytes([7]) * k
            if state["n"] == dup_at:
                return bytes([7]) * k
            return state["n"].to_bytes(k, "big")
        return draw
    for dup_at, want in ((9, []), (5, ["nonce_reuse"])):
        chain, trust, _ = valid_chain(2, rng_fn=factory(dup_at))
        chain.anchor_now()
        for _ in range(3):
            from ._fixtures import BODIES
            chain.append("inference", BODIES["inference"])
        data, tj = export_bytes(chain), json.loads(trust.to_bytes())
        ind = iv.verify(data, tj)
        assert ind.classes == want, dup_at
        assert signature(ind) == ref_signature(verify_ledger(data, trust_root=trust))


def test_a_tamper_after_the_anchored_prefix_does_not_hide_a_fork_inside_it_for_both_verifiers(tmp_path):
    from attacklab.tamper import TamperDb, build_clean_ledger
    a = build_clean_ledger(tmp_path / "a", n=12, seed=5, checkpoint_every=7, anchor=True)
    b = build_clean_ledger(tmp_path / "b", n=20, seed=5, checkpoint_every=7, anchor=True, diverge_at=6)
    t = TamperDb(b.ledger, tmp_path / "t.db")
    victim = max(x["seq"] for x in b.records() if x["type"] == "inference")
    rec = t.rec(victim)
    rec["output"]["jcs_sha256"] = "0" * 64
    t.put(rec["seq"], rec)
    t.close()
    export_records(tmp_path / "t.db", tmp_path / "t.jsonl")
    data, trust = (tmp_path / "t.jsonl").read_bytes(), json.loads(b.trust.to_bytes())
    anchor = json.loads(a.anchor.read_bytes())
    ind = iv.verify(data, trust, anchors=[anchor])
    ref = verify_ledger(data, trust_root=b.trust, anchors=[anchor])
    assert signature(ind) == ref_signature(ref)
    assert "ledger_fork" in ind.classes and "record_edit" in ind.classes


def test_a_ledger_whose_only_finding_is_a_declared_gap_is_not_labelled_problems(tmp_path, capsys):
    """A declared hole is evidence, not tampering (final-review nit): the frozen ledger must not read as 'PROBLEMS'."""
    (tmp_path / "x.jsonl").write_bytes(EXPORT)
    (tmp_path / "t.json").write_text(json.dumps(TRUST))
    (tmp_path / "a.json").write_text(json.dumps(VEC["anchor"]))
    sys.argv = ["iv", str(tmp_path / "x.jsonl"), "--trust", str(tmp_path / "t.json"), "--anchor", str(tmp_path / "a.json")]
    assert iv.main() == 2                                                     # exit code unchanged: low > info
    out = capsys.readouterr().out
    assert "INTACT, WITH DECLARED GAPS" in out and "PROBLEMS" not in out
    from cva.provenance.seal.cli import main
    assert main(["verify", "--records", str(tmp_path / "x.jsonl"), "--trust", str(tmp_path / "t.json"), "--anchor", str(tmp_path / "a.json")]) == 2
    assert "INTACT, WITH DECLARED GAPS" in capsys.readouterr().out
