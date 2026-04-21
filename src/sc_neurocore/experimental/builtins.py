# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Built-in routes and cases for the experimental harness

from __future__ import annotations

from .alternative_path import AlternativePathCase, AlternativePathRegistry
from .examples import make_demo_sigmoid_route
from .physics_routes import make_heat_cosine_mode_route


def build_builtin_registry() -> AlternativePathRegistry:
    """Register all built-in experimental routes."""

    registry = AlternativePathRegistry()
    registry.register(make_demo_sigmoid_route())
    registry.register(make_heat_cosine_mode_route())
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
    raise KeyError(f"No built-in cases for route: {route_name}")
