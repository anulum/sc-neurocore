# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_physics.py

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.physics.heat import StochasticHeatSolver
from sc_neurocore.physics.wolfram_hypergraph import WolframHypergraph
def _make_uniform_solver(length: float, num_walkers: int, diffusivity: float, seed: int = 0):
    s = StochasticHeatSolver(
        length=length, num_walkers=num_walkers, diffusivity=diffusivity, seed=seed
    )
    s.set_initial_distribution(lambda x: np.ones_like(x))
    return s

__all__ = ['np', 'pytest', 'StochasticHeatSolver', 'WolframHypergraph', '_make_uniform_solver']
