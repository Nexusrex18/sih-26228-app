"""Independent feature-space comparisons. No model is downloaded or executed here."""
import numpy as np

from cva.core.capability import Availability, Capability
from cva.loaders.drift import EmbeddingRows, FeatureTable, dataset_id

from .common import compare_features, finding
from .config import DriftConfig


class EmbeddingDistance:
    id = 'drift.distribution'
    version = '1.1'
    requires = frozenset({Capability.DATASET_IMAGES, Capability.REFERENCE_CLEAN_SET})
    optional = frozenset()
    attack_classes = frozenset({'distribution_shift'})

    def __init__(self, config=None):
        self.config = config or DriftConfig()

    def assess(self, reference, incoming, reference_dist=None, incoming_dist=None):
        config = self.config
        ref_ids = tuple(s.sample_id for s in reference.samples)
        inc_ids = tuple(s.sample_id for s in incoming.samples)
        ref_table, inc_table = FeatureTable(dataset_id(reference)), FeatureTable(dataset_id(incoming))
        if not isinstance(reference_dist, EmbeddingRows) or not isinstance(incoming_dist, EmbeddingRows):
            return [finding(self.id, inc_table, 'Embedding observations unavailable: summary moments alone cannot support KS.', state=Availability.UNAVAILABLE)]
        if reference_dist.sample_ids != ref_ids or incoming_dist.sample_ids != inc_ids:
            return [finding(self.id, inc_table, 'Embedding sample alignment differs from Dataset.', state=Availability.UNAVAILABLE)]
        reference, incoming = reference_dist, incoming_dist
        x, y = reference.embeddings, incoming.embeddings
        if x is None or y is None:
            return [finding(self.id, inc_table, 'Embedding checks unavailable: supply both embedding matrices '
                            'and their extractor identity/version.', state=Availability.UNAVAILABLE)]
        if (reference.extractor_id, reference.extractor_version) != (incoming.extractor_id, incoming.extractor_version) or x.shape[1] != y.shape[1]:
            return [finding(self.id, inc_table, 'Embedding extractor/version or dimensions differ; '
                            'comparison would be invalid.', state=Availability.UNAVAILABLE)]
        x, y = np.asarray(x, float), np.asarray(y, float)
        # Fixed seed projections, not axes fitted to the incoming shift. Bounded work
        # and multiplicity for large embedding widths. Include every coordinate if small.
        if x.shape[1] <= 16:
            basis = np.eye(x.shape[1])
        else:
            basis = np.random.default_rng(config.seed).normal(size=(x.shape[1], 16))
            basis /= np.linalg.norm(basis, axis=0)
        px, py = x@basis, y@basis
        rows = compare_features(ref_table, inc_table, config, self.id,
            {f'projection_{i}': px[:,i] for i in range(px.shape[1])},
            {f'projection_{i}': py[:,i] for i in range(py.shape[1])})
        delta = y.mean(axis=0)-x.mean(axis=0)
        variance = np.var(x, axis=0, ddof=1)
        floor = max(float(variance.mean()) * 1e-6, 1e-12)
        rows[0].evidence[0].data['centroid'] = {
            'euclidean': float(np.linalg.norm(delta)),
            'diagonal_mahalanobis': float(np.sqrt(np.sum(delta**2/np.maximum(variance, floor)))),
            'variance_floor': floor, 'extractor': reference.extractor_id,
            'extractor_version': reference.extractor_version}
        rows[0].limitations.append('Fixed projections can miss shifts; centroid distance is descriptive, '
                                  'not a calibrated significance test. No semantic terrain labels inferred.')
        return rows
