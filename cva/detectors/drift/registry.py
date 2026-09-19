from cva.core.registry import register_drift

from .embedding_distance import EmbeddingDistance
from .interpretable_axes import InterpretableAxes

register_drift(EmbeddingDistance)
register_drift(InterpretableAxes)
