"""sc_neurocore.spatial -- Tier: research (experimental / research)."""

__tier__ = "research"

from .representations import VoxelGrid, PointCloud
from .transformer_3d import SpatialTransformer3D

__all__ = [
    "VoxelGrid",
    "PointCloud",
    "SpatialTransformer3D",
]
