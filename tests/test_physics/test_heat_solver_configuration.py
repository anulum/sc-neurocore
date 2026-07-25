# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Feynman-Kac heat-solver configuration contracts

from __future__ import annotations

from typing import Any, cast

import pytest

from sc_neurocore.physics.heat import FeynmanKacHeatSolver, StochasticHeatSolver


def test_alias_points_at_feynman_kac() -> None:
    """The legacy `StochasticHeatSolver` name is now the new class."""
    assert StochasticHeatSolver is FeynmanKacHeatSolver


def test_walkers_must_be_initialised_before_step() -> None:
    s = FeynmanKacHeatSolver(num_walkers=100)
    with pytest.raises(RuntimeError, match="walkers not initialised"):
        s.step()


def test_density_and_expectation_require_initialised_walkers() -> None:
    s = FeynmanKacHeatSolver(num_walkers=100)
    with pytest.raises(RuntimeError, match="walkers not initialised"):
        s.get_density()
    with pytest.raises(RuntimeError, match="walkers not initialised"):
        s.expectation(lambda x: x)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"length": 0.0}, "length"),
        ({"length": float("nan")}, "length"),
        ({"length": True}, "length"),
        ({"diffusivity": -1e-9}, "diffusivity"),
        ({"diffusivity": float("inf")}, "diffusivity"),
        ({"diffusivity": False}, "diffusivity"),
        ({"num_walkers": 0}, "num_walkers"),
        ({"num_walkers": 1.5}, "num_walkers"),
        ({"num_walkers": True}, "num_walkers"),
        ({"dt": 0.0}, "dt"),
        ({"dt": float("nan")}, "dt"),
        ({"dt": False}, "dt"),
        ({"seed": 1.2}, "seed"),
        ({"seed": True}, "seed"),
    ],
)
def test_solver_rejects_nonphysical_configuration(kwargs: dict[str, object], match: str) -> None:
    values: dict[str, object] = {
        "length": 1.0,
        "diffusivity": 1.0,
        "num_walkers": 10,
        "dt": 1e-3,
        "seed": 42,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        FeynmanKacHeatSolver(**cast(Any, values))
