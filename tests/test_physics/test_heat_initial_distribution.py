# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Feynman-Kac heat initial-distribution contracts

from __future__ import annotations


import numpy as np
import pytest

from sc_neurocore.physics.heat import FeynmanKacHeatSolver


def test_set_initial_distribution_zero_density_raises() -> None:
    """All-zero PDF must be rejected."""
    s = FeynmanKacHeatSolver(num_walkers=10)
    with pytest.raises(ValueError, match="must integrate"):
        s.set_initial_distribution(lambda x: np.zeros_like(x))


def test_set_initial_distribution_rejects_invalid_grid_and_density_contract() -> None:
    s = FeynmanKacHeatSolver(num_walkers=10)
    with pytest.raises(ValueError, match="n_grid"):
        s.set_initial_distribution(lambda x: np.ones_like(x), n_grid=0)
    with pytest.raises(ValueError, match="matching x"):
        s.set_initial_distribution(lambda x: np.ones(x.size + 1), n_grid=8)
    with pytest.raises(ValueError, match="finite"):
        s.set_initial_distribution(lambda x: np.full_like(x, np.nan), n_grid=8)


def test_set_initial_distribution_uniform_is_uniform() -> None:
    """A uniform initial PDF should produce a uniform initial histogram."""
    s = FeynmanKacHeatSolver(length=2.0, num_walkers=50_000, seed=22)
    s.set_initial_distribution(lambda x: np.ones_like(x), n_grid=128)
    density = s.get_density(n_bins=20)
    target = 1.0 / 2.0
    rel_err = float(np.max(np.abs(density - target)) / target)
    assert rel_err < 0.05, f"uniform initial sampling off: max rel-err {rel_err:.4f}"
