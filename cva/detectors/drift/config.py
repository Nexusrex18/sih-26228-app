"""Module D algorithm options, not a replacement for the frozen core contracts."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DriftConfig:
    alpha: float = .05
    bins: int = 10
    min_samples: int = 20
    min_ks_effect: float = .15
    permutations: int = 199
    seed: int = 0
    axis_thresholds: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1 or not 0 < self.min_ks_effect <= 1:
            raise ValueError('alpha and min_ks_effect must lie in (0,1), effect may equal 1')
        if any(not isinstance(v, int) or isinstance(v, bool) for v in
               (self.bins, self.min_samples, self.permutations, self.seed)) or self.seed < 0:
            raise ValueError('Counts and seed must be integers; seed must be nonnegative')
        if self.bins < 2 or self.min_samples < 2 or self.permutations < 19:
            raise ValueError('Require bins >= 2, min_samples >= 2, permutations >= 19')


    @classmethod
    def from_profile(cls, profile):
        psi = profile['psi']
        return cls(alpha=psi['alpha'], bins=psi['n_bins'], min_samples=psi['min_samples'],
                   min_ks_effect=psi['min_ks_effect'], permutations=psi['permutations'],
                   seed=psi['seed'], axis_thresholds=psi['axis_thresholds'])


def load_profile(path=None, *, config=None, overrides=None, selftest=False):
    """Resolve file settings and explicit CLI overrides before any data is loaded."""
    import copy
    import json
    from pathlib import Path

    from cva.core.orchestrator import validate_profile_schema

    default = Path(__file__).resolve().parents[3] / 'profiles/drift.json'
    profile = json.loads(default.read_text())
    if path is not None:
        supplied = copy.deepcopy(path) if isinstance(path, dict) else json.loads(Path(path).read_text())
        # Validate the input first: no unknown fields hidden by a merge.
        validate_profile_schema(supplied, supplied.get('name', 'baseline'))
        profile.update(supplied)
        profile['psi'] = {**json.loads(default.read_text())['psi'], **supplied.get('psi', {})}
        if 'min_samples' in supplied.get('psi', {}):
            for key in ('min_reference_n', 'min_incoming_n'):
                if key not in supplied['psi']:
                    profile['psi'][key] = supplied['psi']['min_samples']
    options = dict(overrides or {})
    if config is not None:
        from dataclasses import asdict
        options = {**asdict(config), **options}
    options = {key: value for key, value in options.items() if value is not None}
    if 'min_samples' in options:
        profile['psi']['min_reference_n'] = profile['psi']['min_incoming_n'] = options['min_samples']
    for key, value in options.items():
        profile['psi']['n_bins' if key == 'bins' else key] = value
    if selftest:
        profile.update(name='selftest', pin_clock=True, scan_id_from_seed=True)
    validate_profile_schema(profile, profile['name'])
    if min(profile['psi']['min_reference_n'], profile['psi']['min_incoming_n']) < 2:
        raise ValueError('Drift comparisons require at least two samples per batch')
    # Profiles cannot silently carry unapplied check/budget selections.
    known = {'drift.distribution', 'drift.interpretable_axes', 'drift.semantic_axis', 'drift.vs_manipulation'}
    for key in ('checks', 'except_checks', 'disabled_checks'):
        if set(profile.get(key) or ()) - known:
            raise ValueError(f'{key} contains unknown drift check IDs')
    return profile
