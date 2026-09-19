"""Drift profile policy evaluated by the shared risk engine.

No trained calibration artifact exists yet: preserve the declared zero-confidence
placeholder, with p/q evidence separate. Material effect triggers review, not quarantine.
"""
from cva.core.capability import Availability
from cva.core.types import Disposition
from cva.risk.disposition import default_policy
from cva.risk.engine import assess


def policy():
    base = default_policy()
    base['rules'].insert(0,{'id':'DRIFT_MATERIAL','detector_prefix':'drift.',
                           'min_severity':'medium','disposition':'review','cap':'review',
                           'description':'Material statistical shift; no attack probability assumed'})
    return base


def apply_policy(findings):
    # Batch-level findings, not sample flags: contributor aggregation does not apply.
    assess(findings,None,{'disposition':policy()},seed=0)
    for f in findings:
        if f.availability != Availability.OK:
            f.disposition = Disposition.REVIEW
            f.disposition_rule = 'risk.drift.incomplete_assessment'
        f.limitations.append('Risk confidence is uncalibrated (0 placeholder); no learned attack probability is claimed.')
    return 'REVIEW' if any(f.disposition == Disposition.REVIEW for f in findings) else 'ACCEPT'
