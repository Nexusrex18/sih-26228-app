"""The tamper matrix (Module C plan §10.2; gate C5).

One test per scenario, per form (database file / strict export), per seed. Each asserts the EXACT ordered list
of attack classes — so an unrelated extra finding (a false alarm) and a missing one (a miss) both fail — plus
the target seq, severity, nature, and `primary_check`: the check that actually carries the certainty.
"""
from __future__ import annotations

import pytest

from attacklab.tamper import SCENARIOS, Expected, Outcome, dump_artefact, run_scenario
from cva.provenance.seal.verify import VerifyReport

SEEDS = (1, 2, 3)
ALL = sorted(SCENARIOS)


def above_info(report: VerifyReport):
    return [f for f in report.findings if f.severity != "info"]


def check(report: VerifyReport, exp: Expected, where: str) -> None:
    got = above_info(report)
    assert [f.attack_class for f in got] == list(exp.classes), (
        f"{where}: expected {list(exp.classes)}, got "
        f"{[(f.attack_class, f.seq, f.primary_check) for f in got]}")
    if not exp.classes:
        return
    f = got[0]
    if exp.first_seq is not None:
        assert f.seq == exp.first_seq, f"{where}: first finding at seq {f.seq}, expected {exp.first_seq}"
    if exp.severity:
        assert f.severity == exp.severity, where
    if exp.nature:
        assert f.nature == exp.nature, where
    if exp.primary_check:
        assert f.primary_check == exp.primary_check, where
    assert f.cascade_suppressed >= exp.cascade_min, f"{where}: cascade {f.cascade_suppressed} < {exp.cascade_min}"


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("scenario", ALL)
def test_every_scenario_is_caught_with_the_exact_class_in_every_form(tmp_path, scenario, seed):
    _, outcome = run_scenario(scenario, tmp_path, seed=seed)
    assert outcome.forms, f"{scenario} produced no artefact"
    for form in outcome.forms:
        check(outcome.verify(form), outcome.expected, f"{scenario}/{form}/seed{seed}")


@pytest.mark.parametrize("scenario", ["T4c", "T12"])
def test_the_in_band_undetectable_scenarios_are_silent_until_an_independent_count_is_supplied(tmp_path, scenario):
    """The honest half of the matrix: nothing to see WITHOUT a reference, and a precise finding WITH one."""
    _, outcome = run_scenario(scenario, tmp_path, seed=2)
    assert outcome.expected.classes == ()
    for form in outcome.forms:
        silent = outcome.verify(form)
        assert silent.clean and not silent.findings, f"{scenario}/{form} was not silent: {silent.classes()}"
        assert outcome.reconcile is not None
        check(outcome.verify(form, **outcome.reconcile_kwargs), outcome.reconcile, f"{scenario}/{form}/reconciled")


def test_tail_truncation_leaves_a_chain_that_verifies_clean_at_every_length(tmp_path):
    """D11 stated as a property: deleting the last m records for ANY m < n leaves a valid chain."""
    from attacklab.tamper import build_clean_ledger
    from cva.provenance.seal.verify import export_records, verify_ledger
    fx = build_clean_ledger(tmp_path, n=25, seed=4, checkpoint_every=6)
    export_records(fx.ledger, tmp_path / "full.jsonl")
    lines = (tmp_path / "full.jsonl").read_bytes().split(b"\n")[:-1]
    for keep in range(1, len(lines)):
        data = b"".join(x + b"\n" for x in lines[:keep])
        rep = verify_ledger(data, trust_root=fx.trust)
        assert rep.clean, f"prefix of {keep} records is not clean: {rep.classes()}"


# --- the two scenarios whose LOUDER database variant differs from the export variant --------------------------

def test_replaying_a_record_into_the_database_is_caught_twice_over(tmp_path):
    """A database cannot hold two rows with one seq, so the attacker must invent a seq column — which disagrees
    with the record. Both the derived-column check and the replay check fire."""
    import sqlite3

    from attacklab.tamper import TamperDb, build_clean_ledger
    from cva.provenance.seal.verify import verify_ledger
    fx = build_clean_ledger(tmp_path / "c", n=30, seed=5)
    t = TamperDb(fx.ledger, tmp_path / "t.db")
    j = 10
    (text, rtype, rec_hash) = t.c.execute("SELECT rec, type, rec_hash FROM records WHERE seq=?", (j,)).fetchone()
    n = t.last_seq() + 1
    t.c.execute("INSERT INTO records(seq, type, nonce, rec_hash, rec) VALUES (?,?,?,?,?)",
                (n, rtype, "e" * 32, rec_hash, text))
    t.close()
    rep = verify_ledger(tmp_path / "t.db", trust_root=fx.trust)
    assert "record_replay" in rep.classes() and "derived_column_mismatch" in rep.classes()
    assert sqlite3 is not None


def test_reordering_rows_in_the_database_necessarily_disagrees_with_the_seq_column(tmp_path):
    from attacklab.tamper import TamperDb, build_clean_ledger
    from cva.provenance.seal.verify import verify_ledger
    fx = build_clean_ledger(tmp_path / "c", n=30, seed=6)
    t = TamperDb(fx.ledger, tmp_path / "t.db")
    a, b = 8, 14
    ra, rb = t.rec(a), t.rec(b)
    t.raw(a, __import__("json").dumps(rb, separators=(",", ":"), sort_keys=True))    # swap the record TEXT only:
    t.raw(b, __import__("json").dumps(ra, separators=(",", ":"), sort_keys=True))    # the seq column IS the order
    t.close()
    rep = verify_ledger(tmp_path / "t.db", trust_root=fx.trust)
    assert "derived_column_mismatch" in rep.classes() and not rep.clean


def test_a_model_reload_without_a_reference_manifest_is_information_not_an_accusation(tmp_path):
    """T7a without the external registry: the ledger alone cannot tell a legitimate reload from a swap."""
    _, outcome = run_scenario("T7a", tmp_path, seed=1)
    rep = outcome.verify("db", reference_manifest=None)
    swaps = rep.by_class("model_swap")
    assert len(swaps) == 1 and swaps[0].severity == "info" and rep.clean
    assert "reference manifest" in swaps[0].reason


# --- reproducibility (PS §2.3) ------------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", ALL)
def test_same_scenario_and_seed_produce_a_byte_identical_artefact_twice(tmp_path, scenario):
    _, a = run_scenario(scenario, tmp_path / "run1", seed=7)
    _, b = run_scenario(scenario, tmp_path / "run2", seed=7)
    assert dump_artefact(a) == dump_artefact(b), f"{scenario} is not reproducible"
    assert a.expected == b.expected


def test_different_seeds_attack_different_records(tmp_path):
    targets = {run_scenario("T1a", tmp_path / f"s{s}", seed=s)[1].expected.first_seq for s in range(1, 8)}
    assert len(targets) >= 3


def test_every_outcome_is_an_instance_of_the_contract(tmp_path):
    _, o = run_scenario("T1a", tmp_path, seed=1)
    assert isinstance(o, Outcome) and o.scenario == "T1a" and o.forms == ("db", "export")
