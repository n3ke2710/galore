"""
GaLore Framework: Memory-Efficient Training via Gradient Low-Rank Projection.

Implements:
  - Standard GaLore (fixed rank, truncated SVD)
  - Proximal GaLore (dynamic rank, Singular Value Thresholding / nuclear norm)
"""

from .projector import GaLoreProjector, ProximalGaLoreProjector
from .optimizers import StandardAdamW, GaLoreAdamW, ProximalGaLoreAdamW
from .utils import TrainingTracker, compute_memory_footprint

__all__ = [
    "GaLoreProjector",
    "ProximalGaLoreProjector",
    "StandardAdamW",
    "GaLoreAdamW",
    "ProximalGaLoreAdamW",
    "TrainingTracker",
    "compute_memory_footprint",
]
