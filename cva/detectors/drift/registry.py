"""Explicit composition of frozen DriftTest implementations; no alternate core registry."""
from cva.core.capability import Availability, Capability
from cva.core.types import unavailable_finding
from cva.loaders.drift import dataset_id

from .embedding_distance import EmbeddingDistance
from .interpretable_axes import InterpretableAxes


class DeferredSemantic:
    id = 'drift.semantic_axis'
    version = '1.1'
    requires = frozenset({Capability.DATASET_IMAGES,Capability.REFERENCE_CLEAN_SET})
    optional = frozenset()
    attack_classes = frozenset({'semantic_shift'})
    reason = 'Semantic cluster assessment has not been implemented.'

    def assess(self,reference,incoming,reference_dist=None,incoming_dist=None):
        return [unavailable_finding(self.id,self.version,dataset_id(incoming),self.reason,(),
                                    next(iter(self.attack_classes)),state=Availability.UNAVAILABLE,
                                    target_type='batch')]


class DeferredManipulation(DeferredSemantic):
    id = 'drift.vs_manipulation'
    attack_classes = frozenset({'suspicious_manipulation'})
    reason = 'Intent classifier unavailable: no held-out validated calibration artifact.'


def build_checks(config=None):
    return [EmbeddingDistance(config),InterpretableAxes(config),DeferredSemantic(),DeferredManipulation()]
