"""The two documents a third party needs ship with the code (gate C8; plan §4 puts the spec in `spec/`). The operator guide and
all other Module C notes stay in the notes repository."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = [ROOT / "spec/cva-seal-spec-v1.md", ROOT / "docs/provenance/VERIFICATION-PROCEDURE.md"]   # what a third party needs


def test_the_spec_and_the_verification_procedure_are_in_the_repository():
    for d in DOCS:
        assert d.is_file() and len(d.read_text().splitlines()) > 50, d


def test_every_relative_link_in_the_published_documents_resolves_inside_the_repository():
    for d in DOCS:
        text = d.read_text()
        assert "[[" not in text, f"{d.name}: an unconverted vault wikilink"
        for target in re.findall(r"\]\(([^)#]+)\)", text):
            if "://" in target:
                continue
            assert (d.parent / target).resolve().exists(), f"{d.name} links to {target}, which does not exist"


def test_code_that_cites_the_spec_points_at_a_path_that_exists_in_this_repository():
    for f in (ROOT / "native/c/cvseal.h", ROOT / "spec/independent_verifier.py"):
        text = f.read_text()
        assert "spec/cva-seal-spec-v1.md" in text and "github.com" not in text.split("*/")[0], f
    assert (ROOT / "spec/cva-seal-spec-v1.md").exists()


def test_the_vector_generator_says_what_its_output_does_and_does_not_prove():
    doc = (ROOT / "spec/vectors/build_cva_seal_v1.py").read_text().split('"""')[1]
    assert "proves DETERMINISM, not conformance" in doc and "independent" in doc


def test_the_verification_procedure_states_the_rotation_limit_and_the_independence_qualification():
    t = (ROOT / "docs/provenance/VERIFICATION-PROCEDURE.md").read_text()
    assert "only while the anchor's checkpoint was signed by the genesis key" in t
    assert "same author as the reference verifier" in t and "75 frozen-vector checks" in t
