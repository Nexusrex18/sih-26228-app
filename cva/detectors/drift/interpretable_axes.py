"""Human-readable photometric axes, never an intent detector."""
from cva.core.capability import Capability

from .common import compare_features


class InterpretableAxes:
    id = 'drift.interpretable_axes'
    version = '1.0'
    requires = {Capability.DATASET_IMAGES, Capability.REFERENCE_CLEAN_SET}
    optional = set()

    def assess(self, reference, incoming, config):
        return compare_features(reference, incoming, config, self.id,
                                reference.features, incoming.features)
