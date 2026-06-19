# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — A transformer block specialized for 3D spatial data

from __future__ import annotations
from typing import Any
from dataclasses import dataclass
import numpy as np
from ..layers.attention import StochasticAttention


@dataclass
class SpatialTransformer3D:
    """
    A transformer block specialized for 3D spatial data.
    Processes voxel grids using SC attention.
    """

    resolution: int
    dim_k: int

    def __post_init__(self) -> None:
        self.attention = StochasticAttention(dim_k=self.dim_k)

    def forward(self, voxel_grid: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Input: voxel_grid (res, res, res)
        We flatten the spatial dims to (res^3, 1) or similar to apply attention.
        For simplicity, we treat each voxel as a 'token'.
        """
        res = self.resolution
        # Flatten spatial dims: (res^3, 1)
        # We need a 'feature' dimension. Let's assume features=1 for now.
        flat_grid = voxel_grid.flatten()[:, np.newaxis]

        # Self-attention: Q, K, V are all projections of flat_grid
        # Since we have only 1 feature, attention weights will be simple.
        # In a real model, we'd project to dim_k features.

        # Mock projection to dim_k
        Q = np.repeat(flat_grid, self.dim_k, axis=1)
        K = Q
        V = Q

        attn_out = self.attention.forward(Q, K, V)

        # Reshape back to spatial dims
        # We take the mean of features to get back to 1 value per voxel
        output_grid: np.ndarray[Any, Any] = np.mean(attn_out, axis=1).reshape((res, res, res))

        return output_grid
