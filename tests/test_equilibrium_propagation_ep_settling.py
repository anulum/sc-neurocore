# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEPSettling from former test_equilibrium_propagation.py

"""Focused suite: TestEPSettling from former test_equilibrium_propagation.py."""

from __future__ import annotations

from tests.equilibrium_propagation_support import *  # noqa: F403


class TestEPSettling:
    """Test the free-phase settling process."""

    def test_settle_returns_correct_structure(self) -> None:
        net = EPNetwork([4, 3, 2])
        x = np.array([0.5, 0.3, 0.1, 0.8])
        states = net._settle(x, n_steps=5)
        assert len(states) == 3
        assert states[0].shape == (4,)
        assert states[1].shape == (3,)
        assert states[2].shape == (2,)

    def test_input_stays_clamped(self) -> None:
        net = EPNetwork([4, 3, 2])
        x = np.array([0.5, 0.3, 0.1, 0.8])
        states = net._settle(x, n_steps=20)
        np.testing.assert_array_equal(states[0], x)

    def test_energy_decreases_free_phase(self) -> None:
        net = EPNetwork([5, 4, 3])
        x = np.ones(5) * 0.5
        # Settle progressively and check energy trend
        energies = []
        for steps in [1, 5, 10, 20]:
            states = net._settle(x, n_steps=steps)
            energies.append(net._energy(states))
        # Energy should generally decrease (or be stable) during settling
        # Allow small fluctuations
        assert energies[-1] <= energies[0] + 0.1, f"Energy should decrease: {energies}"
