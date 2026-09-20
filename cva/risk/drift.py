"""Drift findings use the shared engine and a validated, explicit profile."""
from cva.core.orchestrator import _verdict
from cva.risk.engine import assess


def apply_policy(findings, profile):
    # Batch-level findings do not participate in contributor aggregation.
    assess(findings, None, profile, seed=profile['psi']['seed'])
    for f in findings:
        f.limitations.append('Risk confidence is uncalibrated (0 placeholder); no learned attack probability is claimed.')
    return _verdict(findings)
