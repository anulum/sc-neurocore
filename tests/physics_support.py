# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_physics.py

"""Shared stochastic heat solver builder for the focused physics suites."""

from __future__ import annotations

import numpy as np
from sc_neurocore.physics.heat import StochasticHeatSolver


def _make_uniform_solver(
    length: float, num_walkers: int, diffusivity: float, seed: int = 0
) -> StochasticHeatSolver:
    """Build a heat solver whose walkers start from a uniform initial distribution."""
    s = StochasticHeatSolver(
        length=length, num_walkers=num_walkers, diffusivity=diffusivity, seed=seed
    )
    s.set_initial_distribution(lambda x: np.ones_like(x))
    return s


__all__ = ["_make_uniform_solver"]
