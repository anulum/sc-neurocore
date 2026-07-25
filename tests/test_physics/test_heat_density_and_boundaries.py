# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Feynman-Kac heat density and boundary contracts

from __future__ import annotations


import numpy as np
import pytest

from sc_neurocore.physics.heat import FeynmanKacHeatSolver


def test_delta_initial_condition_places_all_walkers_at_x0() -> None:
    s = FeynmanKacHeatSolver(length=2.0, num_walkers=500, seed=7)
    s.set_initial_delta(0.5)
    assert np.all(s.walkers == 0.5)
    assert s.time == 0.0


def test_delta_x0_outside_domain_raises() -> None:
    s = FeynmanKacHeatSolver(length=1.0, num_walkers=10)
    with pytest.raises(ValueError, match="outside"):
        s.set_initial_delta(2.0)


def test_density_integrates_to_unity() -> None:
    """Bin counts × bin width should sum to 1 (probability density)."""
    s = FeynmanKacHeatSolver(length=1.0, num_walkers=5_000, seed=1)
    s.set_initial_delta(0.5)
    s.step(50)
    density = s.get_density(n_bins=64)
    integral = density.sum() * (1.0 / 64)
    assert abs(integral - 1.0) < 1e-9


def test_reflective_boundaries_keep_walkers_in_domain() -> None:
    """No walker should leak outside [0, L] regardless of step count."""
    s = FeynmanKacHeatSolver(
        length=1.0,
        diffusivity=10.0,
        num_walkers=2_000,
        dt=1e-3,
        seed=2026,
    )
    s.set_initial_delta(0.5)
    s.step(500)
    assert s.walkers.min() >= 0.0
    assert s.walkers.max() <= s.length


def test_exact_reflection_handles_arbitrarily_large_overshoot() -> None:
    x = np.array([-4.2, -0.25, 0.25, 1.25, 4.2])
    folded = FeynmanKacHeatSolver._reflect_into_interval(x, 1.0)
    assert np.allclose(folded, [0.2, 0.25, 0.25, 0.75, 0.2])
    assert np.all((folded >= 0.0) & (folded <= 1.0))


def test_zero_diffusivity_preserves_delta_position() -> None:
    s = FeynmanKacHeatSolver(diffusivity=0.0, num_walkers=128, seed=8)
    s.set_initial_delta(0.37)
    s.step(50)
    assert np.all(s.walkers == 0.37)
    assert s.time == pytest.approx(0.05)
