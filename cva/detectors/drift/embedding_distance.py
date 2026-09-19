"""Independent feature-space comparisons. No model is downloaded or executed here."""
import numpy as np

from cva.core.capability import Availability, Capability

from .common import compare_features, finding


class EmbeddingDistance:
    id = 'drift.distribution'
    version = '1.0'
    requires = {Capability.DATASET_IMAGES, Capability.REFERENCE_CLEAN_SET}
    optional = set()

    def assess(self, reference, incoming, config):
        x, y = reference.embeddings, incoming.embeddings
        if x is None or y is None:
            return [finding(self.id, incoming, 'Embedding checks unavailable: supply both embedding matrices '
                            'and their extractor identity/version.', state=Availability.UNAVAILABLE)]
        if (reference.extractor_id, reference.extractor_version) != (incoming.extractor_id, incoming.extractor_version) or x.shape[1] != y.shape[1]:
            return [finding(self.id, incoming, 'Embedding extractor/version or dimensions differ; '
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
        rows = compare_features(reference, incoming, config, self.id,
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
