"""Drift findings use the shared engine and a validated, explicit profile."""
from cva.core.capability import Availability
from cva.core.orchestrator import _verdict
from cva.risk.engine import assess


def apply_policy(findings, profile):
    # Batch-level findings do not participate in contributor aggregation.
    assess(findings, None, profile, seed=profile['psi']['seed'])
    for f in findings:
        f.limitations.append('Risk confidence is uncalibrated (0 placeholder); no learned attack probability is claimed.')
    verdict = _verdict(findings)
    # Backend's pending ERROR-verdict fix is not on modules yet. Preserve quarantine
    # precedence and surface a crash without changing any finding's disposition.
    if verdict == 'ACCEPT' and any(f.availability == Availability.ERROR for f in findings):
        return 'REVIEW'
    return verdict
