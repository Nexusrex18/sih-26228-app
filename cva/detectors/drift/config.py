"""Module D algorithm options, not a replacement for the frozen core contracts."""
from dataclasses import dataclass


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

