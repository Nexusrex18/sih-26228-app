from cva.core.capability import Availability
from cva.core.types import Disposition, Evidence, Finding, Nature, Severity

from .statistics import bh_adjust, compare_axis


def finding(detector, incoming, reason, data=None, shifted=False, state=Availability.OK):
    return Finding(
        detector_id=detector, detector_version='1.0', target_type='batch',
        target_ref=incoming.batch_id,
        severity=Severity.MEDIUM if shifted else Severity.INFO,
        confidence=0., reason=reason, attack_class='distribution_shift',
        nature=Nature.QUALITY, availability=state,
        evidence=[Evidence('json', 'Two-sample drift evidence', data=data)] if data else [],
        disposition=Disposition.REVIEW if shifted or state != Availability.OK else Disposition.ACCEPT,
        disposition_rule='drift.review' if shifted or state != Availability.OK else 'drift.no_material_shift',
        limitations=['Confidence is uncalibrated (0 placeholder); p/q values are not probabilities of drift or attack.',
                     'Reference representativeness and independent sampling are assumed; no inference about intent.',
                     'No detected shift does not establish equivalence or safety.'] + list(incoming.limitations))


def compare_features(reference, incoming, config, detector, x, y):
    names = sorted(set(x) & set(y))
    if not names:
        return [finding(detector, incoming, 'No shared numeric axes available.', state=Availability.UNAVAILABLE)]
    rows = {name: compare_axis(x[name], y[name], config) for name in names}
    # Family comprises both statistics on every axis in this detector. BH assumes
    # independence/positive dependence; correlated image axes are an explicit limitation.
    q = bh_adjust([rows[name][key] for name in names for key in ('psi_p','ks_p')])
    changed = []
    for i, name in enumerate(names):
        row = rows[name]
        row['psi_q'], row['ks_q'] = map(float, q[2*i:2*i+2])
        row['material_shift'] = bool(min(row['psi_q'], row['ks_q']) <= config.alpha
                                     and row['ks'] >= config.min_ks_effect)
        if row['material_shift']:
            changed.append(name)
    reason = ('Material shift on ' + ', '.join(
        f'{name} (mean {rows[name]["reference_mean"]:.3g} → {rows[name]["incoming_mean"]:.3g}, '
        f'KS={rows[name]["ks"]:.3g})' for name in changed)
        if changed else 'No material shift detected on the assessed axes.')
    f = finding(detector, incoming, reason, {'axes': rows, 'alpha': config.alpha,
                'minimum_ks_effect': config.min_ks_effect, 'correction': 'BH within detector'}, bool(changed))
    f.score_raw = max(row['ks'] for row in rows.values())
    f.threshold = config.min_ks_effect
    f.limitations += list(reference.limitations) + [
        'KS p-values assume continuous independent samples; ties can make them conservative.',
        'BH correction is within this detector, not across repeated scans; dependence may affect control.']
    missing = sorted(set(x) ^ set(y))
    if missing:
        f.availability = Availability.DEGRADED
        f.limitations.append('Unmatched axes not assessed: ' + ', '.join(missing))
        f.disposition = Disposition.REVIEW
    return [f]
