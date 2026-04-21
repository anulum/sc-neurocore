# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real physics routes for the safe alternative-path harness

from __future__ import annotations

import math

import numpy as np

from sc_neurocore.physics.heat import FeynmanKacHeatSolver

from .alternative_path import AlternativePathRoute


def _heat_cosine_mode_baseline(
    x_0: float,
    horizon: float,
    mode_index: int = 1,
    *,
    length: float = 1.0,
    diffusivity: float = 0.5,
    num_walkers: int = 40_000,
    dt: float = 1e-4,
    seed: int = 7,
) -> float:
    solver = FeynmanKacHeatSolver(
        length=length,
        diffusivity=diffusivity,
        num_walkers=num_walkers,
        dt=dt,
        seed=seed,
    )
    solver.set_initial_delta(x_0)
    solver.evolve_to(horizon)
    wavenumber = mode_index * math.pi / length
    return solver.expectation(lambda x: np.cos(wavenumber * x))


def _heat_cosine_mode_candidate(
    x_0: float,
    horizon: float,
    mode_index: int = 1,
    *,
    length: float = 1.0,
    diffusivity: float = 0.5,
    num_walkers: int = 40_000,
    dt: float = 1e-4,
    seed: int = 7,
) -> float:
    del num_walkers, dt, seed
    decay = math.exp(-diffusivity * (mode_index * math.pi / length) ** 2 * horizon)
    return decay * math.cos(mode_index * math.pi * x_0 / length)


def make_heat_cosine_mode_route() -> AlternativePathRoute[float]:
    """Route Monte Carlo heat evolution against an exact Neumann cosine mode."""

    return AlternativePathRoute(
        name="physics.heat.cosine-mode",
        baseline=_heat_cosine_mode_baseline,
        candidate=_heat_cosine_mode_candidate,
        summary="Feynman-Kac Monte Carlo vs exact Neumann cosine-mode heat solution",
        expected_behavior=(
            "For cosine-mode initial data, the exact candidate should match the "
            "Monte Carlo baseline within sampling tolerance"
        ),
    )
