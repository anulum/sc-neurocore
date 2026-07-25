# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Feynman-Kac heat time-evolution contracts

from __future__ import annotations


import pytest

from sc_neurocore.physics.heat import FeynmanKacHeatSolver


def test_evolve_to_advances_clock() -> None:
    s = FeynmanKacHeatSolver(num_walkers=100, dt=1e-3, seed=4)
    s.set_initial_delta(0.5)
    s.evolve_to(0.05)
    # Allow rounding tolerance from int(round((T-t)/dt))
    assert abs(s.time - 0.05) < 1.5e-3


def test_evolve_to_rejects_nonfinite_time_and_density_bins() -> None:
    s = FeynmanKacHeatSolver(num_walkers=10, seed=5)
    s.set_initial_delta(0.5)
    with pytest.raises(ValueError, match="T"):
        s.evolve_to(float("nan"))
    with pytest.raises(ValueError, match="n_bins"):
        s.get_density(n_bins=0)


def test_evolve_to_backwards_raises() -> None:
    s = FeynmanKacHeatSolver(num_walkers=10, seed=5)
    s.set_initial_delta(0.5)
    s.evolve_to(0.1)
    with pytest.raises(ValueError, match="cannot run backwards"):
        s.evolve_to(0.05)
