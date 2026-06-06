# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Holographic interconnect router

"""Route 3D optical holographic interconnects using SLM phase arrays."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class HolographicRouter:
    """Holographic interconnect router specification.

    Attributes
    ----------
    slm_grid_size : tuple[int, int]
    diffraction_limit_nm : float
    optical_fanout_per_beam : int
    phase_array_complexity : float
    """

    slm_grid_size: tuple[int, int]
    diffraction_limit_nm: float
    optical_fanout_per_beam: int
    phase_array_complexity: float


def route_holographic_interconnects(num_neurons: int, connections: int) -> HolographicRouter:
    """Route 3D optical holographic interconnects using SLM phase arrays."""
    pixels = int(math.ceil(math.sqrt(connections * 2)))
    grid_edge = 1 << (pixels - 1).bit_length()

    fanout = max(1, connections // num_neurons)
    complexity = math.log2(max(2, connections)) * 1.5

    return HolographicRouter(
        slm_grid_size=(grid_edge, grid_edge),
        diffraction_limit_nm=1550.0 / 2.0,
        optical_fanout_per_beam=fanout,
        phase_array_complexity=round(complexity, 2),
    )
