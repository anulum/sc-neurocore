# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: EnergyLIFNeuron

"""Full pipeline test for EnergyLIFNeuron (Fardet & Levina 2020).

LIF with metabolic energy constraint ε. Spike cost depletes ε."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.energy_lif import EnergyLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count


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


class TestEnergyLIFNetwork:
    def test_population(self):
        assert Population(EnergyLIFNeuron, n=10, label="elif").n == 10

    def test_network_spikes(self):
        pop = Population(EnergyLIFNeuron, n=20, label="elif")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=30.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0

    def test_with_projection(self):
        pop = Population(EnergyLIFNeuron, n=10, label="elif")
        proj = Projection(pop, pop, weight=5.0, probability=0.3, seed=42)
        drive = PoissonInput(n=10, rate_hz=500.0, weight=30.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, drive, mon)
        net.run(duration=0.3, dt=0.001, backend="python")
        assert isinstance(mon.spike_trains, dict)


class TestEnergyLIFAnalysis:
    def _get_train(self):
        n = EnergyLIFNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(40.0)
        return train

    def test_firing_rate(self):
        assert firing_rate(self._get_train(), dt=0.001) > 0

    def test_spike_count(self):
        assert spike_count(self._get_train()) > 10
