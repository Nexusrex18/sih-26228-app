"""§9.4's registry table and the code must declare the same capabilities.

§12.1 makes `backend_plan.md` §9.4 the authority for every plug-in's
`requires`/`optional`. Nothing enforced that, and it drifted twice in one audit cycle:

  * four Module A detectors shipped with **no row at all**, which inverted §12.1 — the code
    became the authority because the plan was silent (audit item 8);
  * then `data.negative_space`'s row was transcribed *before* item 10's ruling was
    implemented and not revisited, so the table contradicted shipped code for one commit.
    A reviewer caught it. A test should have.

This is that test. It reads the §9.4 section as vendored at `spec/frozen/`, parses each
plug-in row, and asserts the declared sets against the live registries.

**Why the section is vendored rather than read from the sibling vault.** Same reason B1's
contract is (`tests/core/test_types_match_spec.py`): a test that skips when the vault is
absent asserts nothing on every CI runner while reading green. Only §9.4 is copied, not the
whole plan — other seats edit `backend_plan.md` constantly, and vendoring the file wholesale
would turn every unrelated edit into a drift failure here.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import cva.detectors.data.registry  # noqa: F401 — registration happens at the entrypoint
import cva.detectors.model.registry  # noqa: F401
from cva.core.capability import Capability
from cva.core.registry import DETECTOR_REGISTRY, REGISTRY

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "spec" / "frozen" / "plan-9.4-registry-rows.md"
VAULT = REPO.parent / "sih26228-notes" / "Plan" / "backend_plan.md"

#: A backticked SHOUTING token inside a table cell.
_TOKEN = re.compile(r"`([A-Z][A-Z_]+)`")
#: …of which only these are capabilities. The "→ if missing" column also carries the
#: RESOLUTION (`DEGRADED`, `UNAVAILABLE`), which is an Availability state and not something a
#: plug-in declares. Filtering on the enum rather than a hand-kept denylist means a new
#: capability is picked up automatically and a new Availability state cannot be mistaken for
#: one — a denylist would have gone stale the first time either enum grew.
_CAPS = {c.name for c in Capability}
#: A plug-in row: `| `data.ood` | …requires… | …optional… |`
_ROW = re.compile(r"^\|\s*`((?:data|model|drift)\.[a-z_]+)`\s*\|([^|]*)\|([^|]*)\|", re.M)


def _plan_rows() -> dict[str, tuple[frozenset[str], frozenset[str]]]:
    assert SPEC.exists(), (
        f"the frozen §9.4 extract is missing at {SPEC}. It is vendored precisely so this "
        "check cannot pass by skipping; restore it rather than relaxing the test.")
    rows: dict[str, tuple[frozenset[str], frozenset[str]]] = {}
    for cid, requires, optional in _ROW.findall(SPEC.read_text()):
        rows[cid] = (frozenset(_TOKEN.findall(requires)) & _CAPS,
                     frozenset(_TOKEN.findall(optional)) & _CAPS)
    assert rows, "parsed no rows out of the frozen §9.4 extract — the table shape changed"
    return rows


def _declared(cid: str) -> tuple[frozenset[str], frozenset[str]] | None:
    cls = REGISTRY.get(cid) or DETECTOR_REGISTRY.get(cid)
    if cls is None:
        return None
    return (frozenset(str(c).split(".")[-1] for c in cls.requires),
            frozenset(str(c).split(".")[-1] for c in cls.optional))


def test_the_extract_parses_the_rows_it_is_supposed_to():
    """A parser that silently matched nothing would make every assertion below vacuous —
    the same failure shape as the `content_sha256` bug in B1's contract test."""
    rows = _plan_rows()
    assert len(rows) >= 9, rows
    assert "data.negative_space" in rows
    assert rows["data.near_dup"] == (frozenset({"DATASET_IMAGES"}), frozenset())


#: Scoped to `data.*` for now. Running it over `model.*` too surfaces SEVEN pre-existing
#: divergences between §9.4 and Module B's shipped detectors — e.g. §9.4 gives
#: `model.strip` "`MODEL_PREDICT` **only**" while the code declares three capabilities, and
#: gives `model.weight_digest` `MODEL_WEIGHTS, REFERENCE_MANIFEST` while the code declares
#: none ("file access only"). Those are real, and they belong to **ML-2**, who owns Module B
#: (`Plan/Module-B-Model-Integrity-Plan.md:13`; §6 calls it "ML-2's code and ML-2's
#: question"). Each needs a per-row ruling on which side is authoritative, and two of the
#: rows are marked "confirmed by ML-2" in the plan, so they are not a drive-by fix by
#: another seat. Reported as a finding rather than silently excluded —
#: `test_module_b_rows_are_known_to_diverge` below keeps the set visible so it cannot grow.
_SCOPE = "data."


@pytest.mark.parametrize("cid", sorted(c for c in _plan_rows() if c.startswith(_SCOPE)))
def test_every_planned_row_matches_what_the_plugin_declares(cid: str):
    want_req, want_opt = _plan_rows()[cid]
    got = _declared(cid)
    if got is None:
        pytest.skip(f"{cid} is planned but not registered in this build (another seat's)")
    got_req, got_opt = got
    assert got_req == want_req, (
        f"{cid}: §9.4 says requires={sorted(want_req)}, the code declares "
        f"{sorted(got_req)}. §12.1 makes §9.4 the authority — change the code, or amend the "
        f"plan and re-vendor {SPEC.name}. Do not edit one without the other.")
    assert got_opt == want_opt, (
        f"{cid}: §9.4 says optional={sorted(want_opt)}, the code declares {sorted(got_opt)}")


def test_no_registered_data_detector_is_missing_from_the_plan():
    """Audit item 8's failure, as a test: a shipped detector with no authoritative row means
    the registry has quietly become the authority, which is §12.1 inverted."""
    planned = set(_plan_rows())
    missing = sorted(set(DETECTOR_REGISTRY) - planned)
    assert not missing, (
        f"registered detectors with no §9.4 row: {missing}. Add them to backend_plan.md "
        f"§9.4 and re-vendor {SPEC.name}; until then the code is the de-facto authority.")


def test_module_b_rows_are_known_to_diverge_and_the_set_has_not_grown():
    """The Module B divergences this test found, pinned so they stay visible.

    Not an xfail and not an exclusion list: this asserts the EXACT set, so fixing one fails
    here until the set is updated, and a NEW divergence fails here immediately. The point of
    the audit was that a gap nobody can see is worse than one everybody can.
    """
    known = {
        "model.anomalous", "model.fingerprint", "model.graph_structure",
        "model.intrinsic_probes", "model.neural_cleanse", "model.strip",
        "model.weight_digest",
    }
    rows = _plan_rows()
    diverging = set()
    for cid, (want_req, want_opt) in rows.items():
        if not cid.startswith("model."):
            continue
        got = _declared(cid)
        if got is not None and got != (want_req, want_opt):
            diverging.add(cid)
    assert diverging == known, (
        f"the set of §9.4/code divergences in Module B changed.\n"
        f"  newly diverging: {sorted(diverging - known)}\n"
        f"  now agreeing (remove from `known`): {sorted(known - diverging)}\n"
        f"These are ML-2's — Module B's owner. Each needs a ruling on which side is "
        f"authoritative: §12.1 says §9.4 is, but two of these rows are themselves marked "
        f"'confirmed by ML-2' in that table.")


def test_the_vendored_extract_has_not_drifted_from_the_plan():
    """The one test here that may legitimately skip: it needs the vault, and its subject is
    whether the in-repo copy is stale. Everything above reads the vendored copy and runs
    everywhere."""
    if not VAULT.exists():
        pytest.skip(f"no vault clone at {VAULT}; nothing to compare the extract against")
    plan = VAULT.read_text()
    extract = SPEC.read_text()
    head = extract.split("\n", 1)[0]
    assert head in plan, (
        f"{SPEC.name} does not start with a heading present in backend_plan.md — the section "
        "moved or was renamed; re-vendor it.")
    start = plan.index(head)
    assert plan[start:start + len(extract)] == extract, (
        f"{SPEC.name} has drifted from backend_plan.md §9.4. The vendored copy is what every "
        "other test in this file asserts against, so a stale copy checks the wrong table. "
        "Re-copy the section and re-run.")
