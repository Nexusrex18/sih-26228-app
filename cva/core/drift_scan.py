"""Dataset scan orchestration, independent of Module B's ModelHandle orchestration."""
import hashlib
import json
import time
from dataclasses import asdict

from cva.core.capability import Availability, Capability, CapabilitySet, Resolution
from cva.core.drift import DriftBatch, DriftConfig
from cva.core.orchestrator import PlanRow, ScanResult
from cva.core.registry import DRIFT_REGISTRY
from cva.core.types import Disposition, Finding, Nature, Severity

DEFERRED = {
    'drift.semantic_axis': 'Semantic cluster assessment has not been implemented.',
    'drift.vs_manipulation': 'Intent classifier unavailable: no trained, held-out validated calibration artifact.',
}


def scan_drift(reference: DriftBatch | None, incoming: DriftBatch,
               config: DriftConfig | None = None) -> ScanResult:
    config = config or DriftConfig()
    if not DRIFT_REGISTRY:
        raise RuntimeError('No drift plug-ins registered; import cva.detectors.drift.registry at the entrypoint')
    identity = json.dumps({'reference':reference.batch_id if reference else None,
                          'incoming':incoming.batch_id, 'config':asdict(config), 'version':'1.0'}, sort_keys=True)
    scan_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
    caps = {Capability.DATASET_IMAGES}
    if reference is not None:
        caps.add(Capability.REFERENCE_CLEAN_SET)
    plan: list[PlanRow] = []
    findings: list[Finding] = []
    timings: dict[str, float] = {}
    for cid in sorted(set(DRIFT_REGISTRY) | set(DEFERRED)):
        start = time.perf_counter()
        state, reason = Availability.OK, 'Reference and incoming batch available'
        if cid in DEFERRED:
            state, reason = Availability.UNAVAILABLE, DEFERRED[cid]
        elif reference is None:
            state, reason = Availability.UNAVAILABLE, 'No declared reference distribution supplied'
        elif min(len(reference.sample_ids),len(incoming.sample_ids)) < config.min_samples:
            state, reason = Availability.UNAVAILABLE, f'Need at least {config.min_samples} samples in each batch'
        if state == Availability.UNAVAILABLE:
            got = [Finding(cid,'1.0','batch',incoming.batch_id,Severity.INFO,0.,reason,
                           'distribution_shift',availability=state,nature=Nature.QUALITY,
                           limitations=['This check was not performed.'])]
        else:
            try:
                assert reference is not None
                got = DRIFT_REGISTRY[cid]().assess(reference,incoming,config)
                states = {f.availability for f in got}
                if not got:
                    raise RuntimeError('Drift plug-in returned no assessment')
                state = next((s for s in (Availability.ERROR, Availability.UNAVAILABLE,
                                          Availability.DEGRADED) if s in states),Availability.OK)
                reason = got[0].reason if state != Availability.OK else reason
            except Exception as exc:
                state, reason = Availability.ERROR, f'{type(exc).__name__}: {exc}'
                got = [Finding(cid,'1.0','batch',incoming.batch_id,Severity.MEDIUM,0.,
                               'Drift check failed: '+reason, 'tool.error', availability=state)]
        for f in got:
            f.scan_id = scan_id
            f.access_assumptions += ['Declared reference is representative; batch observations are independently sampled.']
        plan.append(PlanRow(cid,Resolution(state,reason),{'distribution_shift'}))
        findings.extend(got)
        timings[cid] = round(time.perf_counter()-start,3)
    # Incomplete coverage never produces an unqualified ACCEPT.
    verdict = 'REVIEW' if any(f.disposition == Disposition.REVIEW or
                f.availability != Availability.OK for f in findings) else 'ACCEPT'
    result = ScanResult(scan_id,incoming.batch_id,'dataset',CapabilitySet(frozenset(caps)),
                       plan,findings,timings,verdict)
    return result
