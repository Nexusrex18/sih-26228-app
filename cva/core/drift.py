"""Dataset-level contracts. Raw rows are necessary for KS; moments alone are insufficient."""
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from cva.core.types import Finding


@dataclass(frozen=True)
class DriftBatch:
    batch_id: str
    sample_ids: tuple[str, ...]
    features: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    embeddings: NDArray[np.float64] | None = None
    extractor_id: str | None = None
    extractor_version: str | None = None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        n = len(self.sample_ids)
        if not n or len(set(self.sample_ids)) != n:
            raise ValueError('A batch needs nonempty, unique sample IDs')
        for name, values in self.features.items():
            if np.asarray(values).shape != (n,) or not np.isfinite(values).all():
                raise ValueError(f'{name}: expected one finite value per sample')
        if self.embeddings is not None:
            x = np.asarray(self.embeddings)
            if x.ndim != 2 or x.shape[0] != n or x.shape[1] == 0 or not np.isfinite(x).all():
                raise ValueError('Embeddings must be a finite N x D matrix aligned to sample IDs')
            if not self.extractor_id or not self.extractor_version:
                raise ValueError('Embeddings require extractor_id and extractor_version')


@dataclass(frozen=True)
class DriftConfig:
    alpha: float = .05
    bins: int = 10
    min_samples: int = 20
    min_ks_effect: float = .15
    permutations: int = 199
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1 or not 0 < self.min_ks_effect <= 1:
            raise ValueError('alpha and min_ks_effect must lie in (0,1), effect may equal 1')
        if any(not isinstance(v, int) or isinstance(v, bool) for v in
               (self.bins, self.min_samples, self.permutations, self.seed)) or self.seed < 0:
            raise ValueError('Counts and seed must be integers; seed must be nonnegative')
        if self.bins < 2 or self.min_samples < 2 or self.permutations < 19:
            raise ValueError('Require bins >= 2, min_samples >= 2, permutations >= 19')


class BatchDriftTest(Protocol):
    id: str
    version: str
    def assess(self, reference: DriftBatch, incoming: DriftBatch,
               config: DriftConfig) -> list[Finding]: ...
