"""Canonical DriftTest for human-readable photometric shifts."""
import numpy as np

from cva.core.capability import Capability
from cva.loaders.drift import measure_dataset

from .common import compare_features, finding
from .config import DriftConfig


class InterpretableAxes:
    id = 'drift.interpretable_axes'
    version = '1.1'
    requires = frozenset({Capability.DATASET_IMAGES})
    optional = frozenset()
    attack_classes = frozenset({'distribution_shift'})

    def __init__(self, config=None):
        self.config = config or DriftConfig()

    def assess(self, reference, incoming, reference_dist=None, incoming_dist=None):
        inc = measure_dataset(incoming)
        if reference is None:
            summary = {name: {'n': len(values), 'mean': float(np.mean(values)),
                             'min': float(np.min(values)), 'max': float(np.max(values))}
                       for name, values in sorted(inc.features.items())}
            f = finding(self.id, inc, 'Incoming image axes measured; no reference supplied, so no shift comparison was performed.',
                        {'incoming_axes': summary, 'mode': 'description_only'})
            f.limitations.append('Description only: PSI/KS and drift magnitude were not assessed without a reference.')
            return [f]
        ref = measure_dataset(reference)
        return compare_features(ref, inc, self.config, self.id, ref.features, inc.features)
