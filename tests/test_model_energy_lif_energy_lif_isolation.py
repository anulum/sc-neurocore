# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEnergyLIFIsolation from former test_model_energy_lif.py

"""Focused suite: TestEnergyLIFIsolation from former test_model_energy_lif.py."""

from __future__ import annotations

from tests.model_energy_lif_support import *  # noqa: F403


class TestEnergyLIFIsolation:
    def test_construction(self):
        n = EnergyLIFNeuron()
        assert n.epsilon == 1.0

    def test_step_returns_binary(self):
        assert EnergyLIFNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = EnergyLIFNeuron()
        assert sum(n.step(10.0) for _ in range(5000)) == 0

    def test_spikes(self):
        n = EnergyLIFNeuron()
        assert sum(n.step(30.0) for _ in range(5000)) > 10

    def test_energy_depletes(self):
        """ε should decrease after spiking."""
        n = EnergyLIFNeuron()
        for _ in range(5000):
            n.step(50.0)
        assert n.epsilon < 1.0, "energy did not deplete"

    def test_energy_recovers(self):
        """ε should recover toward ε₀ without spiking."""
        n = EnergyLIFNeuron()
        n.epsilon = 0.1
        for _ in range(5000):
            n.step(0.0)
        assert n.epsilon > 0.1, "energy did not recover"

    def test_energy_gates_spiking(self):
        """When ε < 0.1, neuron cannot spike (energy gate)."""
        n = EnergyLIFNeuron()
        n.epsilon = 0.05
        spikes = sum(n.step(50.0) for _ in range(100))
        assert spikes == 0, "spiked with depleted energy"

    def test_energy_nonnegative(self):
        n = EnergyLIFNeuron()
        for _ in range(10000):
            n.step(50.0)
        assert n.epsilon >= 0.0

    def test_reset(self):
        n = EnergyLIFNeuron()
        for _ in range(100):
            n.step(50.0)
        n.reset()
        assert n.v == n.v_rest
        assert n.epsilon == n.epsilon_0
