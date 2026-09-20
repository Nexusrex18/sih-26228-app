"""Gate N1: every fixture state renders faithfully."""
from __future__ import annotations

import pytest

from tests.web.conftest import post, sign_in

ALL_UNAVAILABLE = "s-2026-09-19-0002"
CLEAN = "s-2026-09-19-0003"
BLACKBOX = "s-2026-09-19-0004"


def test_everything_needs_a_login(client):
    for url in ("/", "/scans/s-2026-09-19-0001", "/audit/",
                "/evidence/" + "0" * 64):
        resp = client.get(url)
        assert resp.status_code in (302, 303), url
        assert "/login" in resp.headers["Location"], url


def test_the_scan_list_renders(signed_in, scan_id):
    body = signed_in.get("/").get_data(as_text=True)
    assert scan_id in body
    assert ALL_UNAVAILABLE in body


@pytest.mark.parametrize("path", [
    "", "/findings", "/contributors", "/provenance", "/coverage", "/reproduction", "/drift",
])
def test_every_section_renders(signed_in, scan_id, path):
    resp = signed_in.get(f"/scans/{scan_id}{path}")
    assert resp.status_code == 200, resp.get_data(as_text=True)[:600]


def test_the_read_order_matches_the_single_file_report(signed_in, scan_id):
    """Both artefacts tell the same story in the same sequence (plan §7.2)."""
    import re
    body = signed_in.get(f"/scans/{scan_id}").get_data(as_text=True)
    # The section nav repeats these words above the content, so match the <h2> headings
    # themselves rather than the first occurrence of the string.
    headings = re.findall(r"<h2>([^<]+)</h2>", body)
    order = ["Access assumptions", "Contributor risk", "Provenance", "Coverage",
             "Reproduction"]
    found = [h for h in headings if h in order]
    assert found == order, f"sections out of order: {found}"
    assert "Verdict" in body


def test_a_scan_where_everything_was_unavailable_does_not_look_green(signed_in):
    """Gate N1's sharpest requirement."""
    body = signed_in.get(f"/scans/{ALL_UNAVAILABLE}").get_data(as_text=True)
    assert "UNAVAILABLE" in body
    assert "hatched" in body, "absence must be rendered as measured, not greyed away"
    assert "A short report here is a small statement, not a clean one." in body


def test_zero_findings_says_which_detectors_ran(signed_in):
    """'0 findings' must display which detectors ran and which did not."""
    body = signed_in.get(f"/scans/{CLEAN}/findings").get_data(as_text=True)
    assert "produced no findings" in body
    assert "data.label_flip" in body, "the plan rows must appear beside the zero"
    assert "prov.ledger_verify" in body


def test_the_provenance_section_is_present_even_when_clean(signed_in):
    """Module C O9: absence of evidence must not render as evidence of absence."""
    body = signed_in.get(f"/scans/{CLEAN}").get_data(as_text=True)
    assert "Provenance" in body
    assert "Unwitnessed window" in body
    assert "Records verified" in body


def test_a_report_with_no_provenance_section_says_so(signed_in):
    body = signed_in.get(f"/scans/{BLACKBOX}/provenance").get_data(as_text=True)
    assert "No provenance section in this report" in body
    assert "That is not a clean provenance result. It is the absence of one." in body


def test_unavailable_carries_its_exclusion_reason(signed_in, scan_id):
    """Capability and budget are different facts and are never collapsed."""
    body = signed_in.get(f"/scans/{scan_id}").get_data(as_text=True)
    assert "capability" in body
    assert "budget" in body
    assert "41 min" in body, "a budget exclusion prints what it would have cost"


def test_severity_and_confidence_are_separate_fields(signed_in, scan_id, app):
    from cva.web.reports.loader import load
    report = load(app.config["CVA"].reports_dir, scan_id)
    fid = next(f["finding_id"] for f in report.findings
               if f["detector_id"] == "data.label_flip")
    body = signed_in.get(f"/scans/{scan_id}/findings/{fid}").get_data(as_text=True)
    assert "Severity" in body and "Confidence" in body
    assert "impact if true" in body
    assert "calibrated belief" in body


def test_the_finding_page_shows_the_rule_that_fired(signed_in, scan_id, app):
    from cva.web.reports.loader import load
    report = load(app.config["CVA"].reports_dir, scan_id)
    fid = next(f["finding_id"] for f in report.findings if f["disposition_rule"] == "D3")
    body = signed_in.get(f"/scans/{scan_id}/findings/{fid}").get_data(as_text=True)
    assert "D3" in body


def test_an_exif_cluster_attribution_is_labelled_a_hypothesis(signed_in, scan_id):
    body = signed_in.get(f"/scans/{scan_id}/contributors").get_data(as_text=True)
    assert "hypothesis" in body
    assert "camera-serial cluster, not a declared identity" in body


def test_the_contributor_table_precedes_individual_findings(signed_in, scan_id):
    body = signed_in.get(f"/scans/{scan_id}").get_data(as_text=True)
    assert body.index("Contributor risk") < body.index("Provenance")


def test_the_grouping_toggle_offers_contributor_batch_and_source(signed_in, scan_id):
    for group in ("contributor", "batch", "source"):
        resp = signed_in.get(f"/scans/{scan_id}/contributors?group={group}")
        assert resp.status_code == 200
        assert f'aria-pressed="true"' in resp.get_data(as_text=True)


def test_the_blackbox_coverage_statement_is_visibly_smaller(signed_in):
    """Gate N6."""
    rich = signed_in.get(f"/scans/{CLEAN}/coverage").get_data(as_text=True)
    narrow = signed_in.get(f"/scans/{BLACKBOX}/coverage").get_data(as_text=True)
    assert "model.weight_digest" in narrow, "the lost checks are named"
    assert "was not asked to" in narrow, "and the reason is 'not asked', not 'could not'"
    assert rich.count("chip-accept") >= narrow.count("chip-accept")


def test_coverage_comparison_shows_what_the_narrower_profile_gave_up(signed_in):
    body = signed_in.get(
        f"/scans/{CLEAN}/coverage?compare={BLACKBOX}").get_data(as_text=True)
    assert "Assessed here and not there" in body
    assert "record_edit" in body


def test_an_unknown_schema_version_is_refused_not_rendered(app, signed_in, tmp_path):
    import json
    from pathlib import Path
    root = Path(app.config["CVA"].reports_dir)
    bad = root / "s-2027-01-01-0001"
    bad.mkdir()
    payload = json.loads((root / "s-2026-09-19-0003" / "report.json").read_text())
    payload["schema_version"] = "2.0.0"
    payload["scan_id"] = "s-2027-01-01-0001"
    (bad / "report.json").write_text(json.dumps(payload))

    resp = signed_in.get("/scans/s-2027-01-01-0001")
    assert resp.status_code == 422
    body = resp.get_data(as_text=True)
    assert "2.0.0" in body
    assert "mostly renders is a report that can hide a field" in body


def test_an_unreadable_scan_is_listed_with_its_reason(app, signed_in):
    from pathlib import Path
    root = Path(app.config["CVA"].reports_dir)
    broken = root / "s-2027-02-02-0002"
    broken.mkdir()
    (broken / "report.json").write_text("{ not json")
    body = signed_in.get("/").get_data(as_text=True)
    assert "s-2027-02-02-0002" in body
    assert "unreadable" in body


def test_pagination_holds_a_large_finding_set(app, signed_in, scan_id):
    """§9: the findings page is paginated server-side, so a 10k report never materialises."""
    resp = signed_in.get(f"/scans/{scan_id}/findings?page=1")
    assert resp.status_code == 200
    resp = signed_in.get(f"/scans/{scan_id}/findings?page=9999")
    assert resp.status_code == 200, "an out-of-range page is empty, not an error"


def test_filters_narrow_the_list(signed_in, scan_id):
    all_body = signed_in.get(f"/scans/{scan_id}/findings").get_data(as_text=True)
    quarantined = signed_in.get(
        f"/scans/{scan_id}/findings?disposition=quarantine").get_data(as_text=True)
    assert all_body.count('class="finding"') > quarantined.count('class="finding"')


def test_the_nature_filter_separates_quality_from_adversarial(signed_in, scan_id):
    quality = signed_in.get(
        f"/scans/{scan_id}/findings?nature=quality").get_data(as_text=True)
    assert "data.near_dup" in quality
    assert "data.label_flip" not in quality or quality.count('class="finding"') < 4
