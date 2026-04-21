# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Built-in routes and cases for the experimental harness

from __future__ import annotations

import numpy as np

from .alternative_path import AlternativePathCase, AlternativePathRegistry
from .examples import make_demo_sigmoid_route
from .physics_routes import (
    make_harmonic_symplectic_route,
    make_heat_cosine_mode_route,
    make_kuramoto_noiseless_symplectic_lift_route,
)
from .solver_routes import make_lif_subthreshold_exact_route


def build_builtin_registry() -> AlternativePathRegistry:
    """Register all built-in experimental routes."""

    registry = AlternativePathRegistry()
    registry.register(make_demo_sigmoid_route())
    registry.register(make_heat_cosine_mode_route())
    registry.register(make_harmonic_symplectic_route())
    registry.register(make_kuramoto_noiseless_symplectic_lift_route())
    registry.register(make_lif_subthreshold_exact_route())
    return registry


def builtin_cases_for_route(route_name: str) -> list[AlternativePathCase]:
    """Return the default case set for a built-in route."""

    if route_name == "demo.affine-sigmoid":
        return [
            AlternativePathCase("small", args=([0.0, 1.0, -1.0],)),
            AlternativePathCase("biased", args=([2.0, -2.0],), kwargs={"bias": 0.25}),
        ]
    if route_name == "physics.heat.cosine-mode":
        return [
            AlternativePathCase(
                "mode1_short",
                args=(0.2, 0.01),
                kwargs={
                    "mode_index": 1,
                    "length": 1.0,
                    "diffusivity": 0.5,
                    "num_walkers": 40_000,
                    "dt": 1e-4,
                    "seed": 11,
                },
            ),
            AlternativePathCase(
                "mode2_short",
                args=(0.7, 0.008),
                kwargs={
                    "mode_index": 2,
                    "length": 1.0,
                    "diffusivity": 0.25,
                    "num_walkers": 50_000,
                    "dt": 1e-4,
                    "seed": 17,
                },
            ),
        ]
    if route_name == "physics.oscillator.harmonic-symplectic":
        return [
            AlternativePathCase(
                "quarter_turn",
                args=(1.0, 0.0, 0.5 * 3.141592653589793),
                kwargs={"dt": 5e-3},
            ),
            AlternativePathCase(
                "longer_horizon",
                args=(0.3, 0.9, 10.0),
                kwargs={"dt": 5e-3},
            ),
        ]
    if route_name == "physics.kuramoto.noiseless-symplectic-lift":
        return [
            AlternativePathCase(
                "triad_short",
                args=(np.array([0.1, 1.2, 2.4], dtype=np.float64), 0.01),
                kwargs={
                    "omegas": np.array([0.8, 1.0, 1.1], dtype=np.float64),
                    "coupling": 0.18,
                    "dt": 5e-4,
                },
            ),
            AlternativePathCase(
                "quartet_short",
                args=(np.array([0.2, 1.4, 3.0, 4.8], dtype=np.float64), 0.008),
                kwargs={
                    "omegas": np.array([0.9, 1.05, 1.15, 0.95], dtype=np.float64),
                    "coupling": 0.12,
                    "dt": 4e-4,
                },
            ),
        ]
    if route_name == "solver.lif.subthreshold-exact":
        return [
            AlternativePathCase(
                "steady_subthreshold",
                args=(-65.0, 10.0, 20.0),
                kwargs={
                    "tau": 20.0,
                    "v_rest": -65.0,
                    "v_thresh": -50.0,
                    "r_m": 1.0,
                    "dt": 1e-2,
                },
            ),
            AlternativePathCase(
                "elevated_subthreshold",
                args=(-60.0, 12.0, 15.0),
                kwargs={
                    "tau": 12.0,
                    "v_rest": -65.0,
                    "v_thresh": -50.0,
                    "r_m": 1.0,
                    "dt": 5e-3,
                },
            ),
        ]
    raise KeyError(f"No built-in cases for route: {route_name}")
