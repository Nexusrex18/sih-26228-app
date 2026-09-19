"""The single-file HTML report (plan §7.9, §5.5 evidence caps).

Asserts what the page must DO, not how it is worded: no network reference of any kind (the air
gap forbids a CDN, and a report that loses its styling offline reads as a broken tool), scope
stated before conclusions, three sections that always print and say why they are empty,
evidence resolved only from inside the evidence directory, and caps that announce themselves.

Built from `ScanResult` objects directly; PNG evidence is real Pillow output.
"""
from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from cva.core.capability import Availability, Capability, CapabilitySet, Resolution
from cva.core.orchestrator import PlanRow, ScanResult, profile_hash_of, resolve_profile
from cva.core.types import Disposition, Evidence, Finding, Severity
from cva.report.render_html import render


@pytest.fixture(autouse=True)
def _unarmed(monkeypatch):
    monkeypatch.delenv("CVA_DETERMINISTIC", raising=False)


# --- builders ------------------------------------------------------------------

def _png(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buf, "PNG")
    return buf.getvalue()


def _store(root: Path, color: tuple[int, int, int]) -> str:
    """Write a PNG into the evidence directory under its own sha256; return `<hash>.png`."""
    root.mkdir(parents=True, exist_ok=True)
    blob = _png(color)
    name = hashlib.sha256(blob).hexdigest() + ".png"
    (root / name).write_bytes(blob)
    return name


def _profile(**evidence: int) -> dict[str, Any]:
    prof = resolve_profile("deep")
    if evidence:
        prof["evidence"] = {**prof["evidence"], **evidence}
    return prof


def _result(plan=(), findings=(), *, caps: CapabilitySet | None = None,
            verdict: str = "REVIEW", profile: dict[str, Any] | None = None,
            scan_id: str = "s-2026-09-19-0001", **extra: Any) -> ScanResult:
    prof = profile if profile is not None else _profile()
    return ScanResult(
        scan_id=scan_id, model_id="m-1", model_fmt="onnx",
        capabilities=caps or CapabilitySet(), plan=list(plan), findings=list(findings),
        timings={}, verdict=verdict, profile_name="deep", profile_hash=profile_hash_of(prof),
        code_commit="cafe123", access_assumptions="", created_at_utc="2026-09-19T00:00:00+00:00",
        profile=prof, seed=7, target={}, **extra)


def _finding(detector_id: str = "model.x", *,
             availability: Availability = Availability.OK,
             disposition: Disposition = Disposition.REVIEW,
             severity: Severity = Severity.MEDIUM, confidence: float = 0.7,
             evidence: list[Evidence] | None = None, reason: str = "names its evidence",
             ) -> Finding:
    return Finding(
        detector_id=detector_id, detector_version="1.0.0", target_type="model",
        target_ref="m-1", severity=severity, confidence=confidence, reason=reason,
        attack_class="model_substitution", scan_id="s-2026-09-19-0001",
        produced_by="cafe123/profile:abcdef012345", disposition=disposition,
        disposition_rule="D3", availability=availability, evidence=evidence or [])


def _row(cid: str, state: Availability) -> PlanRow:
    return PlanRow(cid, Resolution(state, "r"), {"model_substitution"})


def _html(tmp_path: Path, results, *, root: Path | None = None) -> str:
    out = render(results, tmp_path / "report.html",
                 evidence_root=root if root is not None else tmp_path / "evidence")
    return out.read_text(encoding="utf-8")


def _findings_part(html: str) -> str:
    """From the Findings heading on: the stylesheet at the top names every class."""
    return html[html.index("<h2>Findings"):]


# --- self-containment ----------------------------------------------------------------

def test_the_page_references_no_network_resource_of_any_kind(tmp_path):
    cal = {"method": "isotonic", "brier": 0.1, "excluded_detectors": [],
           "reliability_bins": [{"p_mean": 0.3, "empirical": 0.2, "n": 10}]}
    res = _result([_row("model.x", Availability.OK)], [_finding()], calibration=cal,
                  contributor_risk=[{"group_key": "contributor", "group_value": "B",
                                     "n_samples": 10, "n_flagged": 3, "posterior_mean": 0.3,
                                     "ci_low": 0.1, "ci_high": 0.5, "disposition": "review"}],
                  permutation_test={"statistic": 1.0, "p_value": 0.5, "n_permutations": 100,
                                    "conclusion": "no evidence of non-random flags"})
    html = _html(tmp_path, [res])
    lowered = html.lower()
    assert "http://" not in lowered and "https://" not in lowered
    assert "<script src" not in lowered and "<link" not in lowered
    assert "<script" not in lowered, "the report needs no script at all"
    assert "@import" not in lowered and "url(http" not in lowered


def test_a_single_result_not_wrapped_in_a_list_renders_the_same(tmp_path):
    res = _result([_row("model.x", Availability.OK)], [_finding()])
    a = render(res, tmp_path / "a.html", evidence_root=tmp_path / "evidence").read_text()
    b = render([res], tmp_path / "b.html", evidence_root=tmp_path / "evidence").read_text()
    assert a == b


def test_untrusted_text_is_escaped(tmp_path):
    res = _result([_row("model.x", Availability.OK)],
                  [_finding(reason="<script>alert(1)</script> & <img src=x onerror=y>")])
    html = _html(tmp_path, [res])
    assert "<script" not in html.lower()
    assert "<img src=x" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


# --- the verdict states its scope ---------------------------------------------------------

def test_the_banner_says_how_many_planned_checks_ran(tmp_path):
    plan = [_row("model.a", Availability.OK), _row("model.b", Availability.DEGRADED),
            _row("model.c", Availability.UNAVAILABLE)]
    html = _html(tmp_path, [_result(plan, verdict="REVIEW")])
    assert "2 of 3 planned checks ran" in html
    assert "VERDICT: REVIEW" in html
    assert "no check ran" not in html.lower()


def test_a_check_that_raised_does_not_count_as_having_run(tmp_path):
    plan = [_row("model.a", Availability.OK), _row("model.b", Availability.OK)]
    findings = [_finding("model.b", availability=Availability.ERROR)]
    html = _html(tmp_path, [_result(plan, findings)])
    assert "1 of 2 planned checks ran" in html


def test_a_zero_check_scan_says_no_check_ran(tmp_path):
    html = _html(tmp_path, [_result(verdict="ACCEPT")])
    assert "no check ran" in html.lower()
    assert "planned checks ran" not in html
    # And every plan row unavailable is the same statement.
    html = _html(tmp_path, [_result([_row("model.a", Availability.UNAVAILABLE)],
                                    verdict="ACCEPT")])
    assert "no check ran" in html.lower()


# --- three sections that always print -------------------------------------------------------

def test_contributor_provenance_and_shift_always_print_and_say_why_they_are_empty(tmp_path):
    html = _html(tmp_path, [_result()])
    for heading in ("Contributor risk", "Provenance", "Distribution shift"):
        assert f"<h2>{heading}</h2>" in html, heading
    lowered = html.lower()
    contrib = lowered.split("<h2>contributor risk</h2>")[1].split("<h2>")[0]
    assert "not computed" in contrib
    prov = lowered.split("<h2>provenance</h2>")[1].split("<h2>")[0]
    assert "no inference ledger" in prov
    shift = lowered.split("<h2>distribution shift</h2>")[1].split("<h2>")[0]
    assert "not computed" in shift


def test_with_an_inference_ledger_the_provenance_note_changes(tmp_path):
    caps = CapabilitySet(frozenset({Capability.INFERENCE_LEDGER}))
    prov = _html(tmp_path, [_result(caps=caps)]).lower().split("<h2>provenance</h2>")[1]
    assert "no inference ledger" not in prov.split("<h2>")[0]


def test_provenance_prints_the_counts_that_are_true_and_the_seal_state(tmp_path):
    plain = _html(tmp_path, [_result()])
    prov = plain.split("<h2>Provenance</h2>")[1].split("<h2>")[0]
    assert "<b>0</b><span>records verified</span>" in prov
    assert "<b>0</b><span>anchors checked</span>" in prov
    assert "not sealed" in prov.lower()

    sealed = _html(tmp_path, [_result(ledger_seq="seq-9")])
    assert "seq-9" in sealed.split("<h2>Provenance</h2>")[1].split("<h2>")[0]


def test_contributor_rows_replace_the_not_computed_note(tmp_path):
    rows = [{"group_key": "contributor", "group_value": "team_b", "n_samples": 214,
             "n_flagged": 190, "posterior_mean": 0.88, "ci_low": 0.82, "ci_high": 0.92,
             "disposition": "quarantine"}]
    perm = {"statistic": 12.5, "p_value": 0.001, "n_permutations": 2000,
            "conclusion": "flags are not randomly distributed"}
    html = _html(tmp_path, [_result(contributor_risk=rows, permutation_test=perm)])
    section = html.split("<h2>Contributor risk</h2>")[1].split("<h2>")[0]
    assert "contributor:team_b" in section and "214" in section and "0.880" in section
    assert "2000 permutations" in section
    assert "not computed" not in section.lower()


# --- evidence resolution ----------------------------------------------------------------------

def test_module_a_prefixed_and_bare_evidence_paths_both_inline_as_png(tmp_path):
    root = tmp_path / "evidence"
    bare = _store(root, (200, 30, 30))
    prefixed = _store(root, (30, 200, 30))
    ev = [Evidence("image_crop", "bare form", path=bare),
          Evidence("image_crop", "prefixed form", path=f"evidence/{prefixed}")]
    html = _html(tmp_path, [_result([_row("model.x", Availability.OK)],
                                    [_finding(evidence=ev)])], root=root)
    assert html.count('src="data:image/png;base64,') == 2
    assert "evidence file not found" not in html


def test_a_path_that_escapes_the_evidence_directory_is_refused_and_says_so(tmp_path):
    root = tmp_path / "evidence"
    _store(root, (1, 2, 3))
    # Real PNGs sit at every location a lax resolver would reach, so a refusal is the
    # renderer's doing and not just a missing file.
    outside = tmp_path / "x"
    outside.write_bytes(_png((9, 9, 9)))
    nested = root / "evidence"
    nested.mkdir()
    (nested / "x").write_bytes(_png((8, 8, 8)))
    hidden = root / ".secret.png"
    hidden.write_bytes(_png((7, 7, 7)))

    refs = ["evidence/../x", "evidence/evidence/x", ".secret.png", "../x", "sub/dir.png",
            "evidence/.secret.png", "a\\b.png"]
    ev = [Evidence("image_crop", f"ref {i}", path=ref) for i, ref in enumerate(refs)]
    html = _html(tmp_path, [_result([_row("model.x", Availability.OK)],
                                    [_finding(evidence=ev)])], root=root)

    assert "data:image" not in html, "a refused path must never be inlined"
    assert html.count("evidence file not found") == len(refs)


def test_a_missing_but_well_formed_reference_is_a_visible_note_not_silence(tmp_path):
    ev = [Evidence("image_crop", "gone", path="f" * 64 + ".png")]
    html = _html(tmp_path, [_result([_row("model.x", Availability.OK)],
                                    [_finding(evidence=ev)])])
    assert "evidence file not found" in html
    assert "f" * 64 in html


def test_json_evidence_renders_as_preformatted_text_inline_and_from_the_store(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()
    (root / ("a" * 64 + ".json")).write_text('{"stored_key": [1, 2]}')
    ev = [Evidence("json", "inline payload", data={"inline_key": 7}),
          Evidence("json", "stored payload", path="a" * 64 + ".json")]
    html = _html(tmp_path, [_result([_row("model.x", Availability.OK)],
                                    [_finding(evidence=ev)])], root=root)
    # Only the Findings section: the Reproduction block further down is a <pre> as well.
    cards = _findings_part(html).split("<h2>Provenance</h2>")[0]
    assert cards.count("<pre>") == 2
    assert "inline_key" in cards and "stored_key" in cards
    assert "data:image" not in html


# --- reliability diagram --------------------------------------------------------------------------

def test_the_reliability_svg_appears_only_when_calibration_has_bins(tmp_path):
    cal = {"method": "isotonic", "brier": 0.12, "excluded_detectors": ["prov.chain"],
           "reliability_bins": [{"p_mean": 0.2, "empirical": 0.1, "n": 40},
                                {"p_mean": 0.8, "empirical": 0.9, "n": 12}]}
    html = _html(tmp_path, [_result(calibration=cal)])
    assert "<svg" in html
    assert "xmlns" not in html
    assert html.count("<circle") == 2
    assert "not applied" not in html.lower()


def test_without_calibration_there_is_no_diagram_and_the_note_says_not_applied(tmp_path):
    html = _html(tmp_path, [_result()])
    assert "<svg" not in html
    assert "calibration: not applied" in html.lower()


def test_calibration_without_bins_draws_nothing_and_says_so(tmp_path):
    html = _html(tmp_path, [_result(calibration={"method": "isotonic", "brier": None,
                                                 "reliability_bins": [],
                                                 "excluded_detectors": []})])
    assert "<svg" not in html
    assert "no reliability bins" in html.lower()
    assert "not computed" in html.lower()        # the Brier score is not invented


# --- caps announce themselves ------------------------------------------------------------------------

def test_a_findings_cap_and_an_image_cap_each_print_how_many_were_omitted(tmp_path):
    root = tmp_path / "evidence"
    imgs = [Evidence("image_crop", f"img {i}", path=_store(root, (10 * i, 40, 40)))
            for i in range(3)]
    top = _finding("model.top", disposition=Disposition.QUARANTINE, severity=Severity.HIGH,
                   evidence=imgs)
    low = _finding("model.low", disposition=Disposition.REVIEW, severity=Severity.LOW)
    prof = _profile(max_images_per_finding=1, max_findings_rendered=1)
    html = _html(tmp_path, [_result([_row("model.top", Availability.OK),
                                     _row("model.low", Availability.OK)],
                                    [low, top], profile=prof)], root=root)

    assert re.search(r"\b1 of 2 findings rendered", html), "findings truncation not announced"
    assert re.search(r"\b2 images omitted", html), "image truncation not announced"
    part = _findings_part(html)
    assert part.count('src="data:image/png;base64,') == 1
    assert "<code>model.top</code>" in part, "the highest-priority finding is the one kept"
    assert "<code>model.low</code>" not in part.split("<h3 style='margin-top:26px'>")[0]


def test_nothing_is_announced_when_no_cap_was_hit(tmp_path):
    root = tmp_path / "evidence"
    ev = [Evidence("image_crop", "one", path=_store(root, (5, 5, 5)))]
    html = _html(tmp_path, [_result([_row("model.x", Availability.OK)],
                                    [_finding(evidence=ev)])], root=root)
    assert "findings rendered" not in html
    assert "images omitted" not in html
    assert html.count('src="data:image/png;base64,') == 1


def test_the_render_byte_budget_is_the_smallest_cap_across_results(tmp_path):
    root = tmp_path / "evidence"
    ev = [Evidence("image_crop", "crop", path=_store(root, (50, 60, 70)))]
    roomy = _result([_row("model.x", Availability.OK)], [_finding(evidence=ev)],
                    profile=_profile(max_report_bytes=10_000_000), scan_id="s-2026-09-19-0001")
    tight = _result([_row("model.x", Availability.OK)], [_finding(evidence=ev)],
                    profile=_profile(max_report_bytes=2_000), scan_id="s-2026-09-19-0002")

    # Control: on its own the roomy result inlines its image.
    alone = _html(tmp_path, [roomy], root=root)
    assert alone.count("data:image/png") == 1

    # Together the tight result's cap governs the whole page, INCLUDING the roomy result that
    # comes first: its own cap would have kept the image, so an omission proves the minimum.
    both = _html(tmp_path, [roomy, tight], root=root)
    assert "data:image/png" not in both
    assert len(re.findall(r"images omitted", both)) == 2


# --- grouping ---------------------------------------------------------------------------------------------

def test_findings_are_grouped_quarantine_then_review_then_accept(tmp_path):
    findings = [
        _finding("model.acc", disposition=Disposition.ACCEPT, severity=Severity.CRITICAL),
        _finding("model.rev", disposition=Disposition.REVIEW, severity=Severity.CRITICAL),
        _finding("model.quar", disposition=Disposition.QUARANTINE, severity=Severity.INFO),
    ]
    plan = [_row(f.detector_id, Availability.OK) for f in findings]
    part = _findings_part(_html(tmp_path, [_result(plan, findings)]))

    order = [part.index(f"<code>{cid}</code>") for cid in ("model.quar", "model.rev", "model.acc")]
    assert order == sorted(order), "grouping must be by what to DO, not by severity"

    heads = [part.index(f"pill p-{name}'>") for name in ("quarantine", "review", "accept")]
    assert heads == sorted(heads)
    # Each group heading sits directly before its own card.
    assert heads[0] < order[0] < heads[1] < order[1] < heads[2] < order[2]


def test_within_a_group_higher_severity_comes_first(tmp_path):
    findings = [_finding("model.low", severity=Severity.LOW),
                _finding("model.high", severity=Severity.HIGH)]
    plan = [_row(f.detector_id, Availability.OK) for f in findings]
    part = _findings_part(_html(tmp_path, [_result(plan, findings)]))
    assert part.index("<code>model.high</code>") < part.index("<code>model.low</code>")


def test_findings_that_did_not_run_are_listed_apart_and_never_rendered_as_cards(tmp_path):
    findings = [_finding("model.failed", availability=Availability.ERROR, reason="boom"),
                _finding("model.skipped", availability=Availability.UNAVAILABLE,
                         severity=Severity.INFO, reason="no gradients")]
    plan = [_row(f.detector_id, Availability.UNAVAILABLE) for f in findings]
    part = _findings_part(_html(tmp_path, [_result(plan, findings)]))
    assert 'class="card' not in part
    assert "<code>model.failed</code>" in part and "<code>model.skipped</code>" in part
    assert "boom" in part and "no gradients" in part
    assert "0 assessed" in part


def test_all_four_states_appear_in_the_plan_table_at_equal_prominence(tmp_path):
    plan = [_row("model.a", Availability.OK), _row("model.b", Availability.DEGRADED),
            _row("model.c", Availability.UNAVAILABLE), _row("model.d", Availability.ERROR)]
    html = _html(tmp_path, [_result(plan)])
    stats = html.split("<h2>Check plan")[1].split("</div><div class=scroll>")[0]
    for state in ("OK", "DEGRADED", "UNAVAILABLE", "ERROR"):
        assert f"<b>1</b><span>{state}</span>" in stats, state
