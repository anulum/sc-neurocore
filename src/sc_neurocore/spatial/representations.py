# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — A 3D Voxel Grid representation for SC

from __future__ import annotations
from typing import Any
from dataclasses import dataclass
import numpy as np


@dataclass
class VoxelGrid:
    """
    A 3D Voxel Grid representation for SC.
    Each voxel stores a probability of being 'occupied'.
    """

    resolution: int
    data: np.ndarray[Any, Any] = None

    def __post_init__(self) -> None:
        if self.data is None:
            self.data = np.zeros((self.resolution, self.resolution, self.resolution))

    def set_voxel(self, x: int, y: int, z: int, prob: float) -> None:
        if 0 <= x < self.resolution and 0 <= y < self.resolution and 0 <= z < self.resolution:
            self.data[x, y, z] = prob

    def get_as_bitstream(self, length: int = 256) -> np.ndarray[Any, Any]:
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

    points: np.ndarray[Any, Any]  # (N, 3)
    intensities: np.ndarray[Any, Any]  # (N,)

    def normalize(self) -> None:
        self.points = (self.points - np.min(self.points)) / (
            np.max(self.points) - np.min(self.points) + 1e-9
        )
        self.intensities = np.clip(self.intensities, 0, 1)
