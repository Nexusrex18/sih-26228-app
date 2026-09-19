"""Canonical DriftTest for human-readable photometric shifts."""
from cva.core.capability import Capability
from cva.loaders.drift import measure_dataset

from .common import compare_features
from .config import DriftConfig


class InterpretableAxes:
    id = 'drift.interpretable_axes'
    version = '1.1'
    requires = frozenset({Capability.DATASET_IMAGES, Capability.REFERENCE_CLEAN_SET})
    optional = frozenset()
    attack_classes = frozenset({'distribution_shift'})

    def __init__(self, config=None):
        self.config = config or DriftConfig()

    def assess(self, reference, incoming, reference_dist=None, incoming_dist=None):
        ref, inc = measure_dataset(reference), measure_dataset(incoming)
        return compare_features(ref, inc, self.config, self.id, ref.features, inc.features)
