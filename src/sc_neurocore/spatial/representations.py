from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class VoxelGrid:
    """
    A 3D Voxel Grid representation for SC.
    Each voxel stores a probability of being 'occupied'.
    """

    resolution: int
    data: np.ndarray = None

    def __post_init__(self):
        if self.data is None:
            self.data = np.zeros((self.resolution, self.resolution, self.resolution))

    def set_voxel(self, x: int, y: int, z: int, prob: float):
        if 0 <= x < self.resolution and 0 <= y < self.resolution and 0 <= z < self.resolution:
            self.data[x, y, z] = prob

    def get_as_bitstream(self, length: int = 256) -> np.ndarray:
        """
        Converts the voxel grid to a 4D bitstream (X, Y, Z, Length).
        """
        rands = np.random.random((*self.data.shape, length))
        return (rands < self.data[..., None]).astype(np.uint8)


@dataclass
class PointCloud:
    """
    A Point Cloud representation.
    Each point has (x, y, z) coordinates and an associated probability/intensity.
    """

    points: np.ndarray  # (N, 3)
    intensities: np.ndarray  # (N,)

    def normalize(self):
        self.points = (self.points - np.min(self.points)) / (
            np.max(self.points) - np.min(self.points) + 1e-9
        )
        self.intensities = np.clip(self.intensities, 0, 1)
