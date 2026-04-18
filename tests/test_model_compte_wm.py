# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: CompteWMNeuron

"""Full pipeline test for CompteWMNeuron (Compte et al. 2000).

NMDA-based working memory neuron with Mg²⁺ block."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.compte_wm import CompteWMNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count, isi


class TestCompteIsolation:
    def test_construction(self):
        n = CompteWMNeuron()
        assert n.v == -70.0

    def test_step_returns_binary(self):
        assert CompteWMNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = CompteWMNeuron()
        assert sum(n.step(0.5) for _ in range(5000)) == 0

    def test_spikes(self):
        n = CompteWMNeuron()
        assert sum(n.step(2.0) for _ in range(10000)) > 50

    def test_nmda_gating(self):
        """spike_in should activate NMDA pathway."""
        n = CompteWMNeuron()
        n.step(0.0, spike_in=True)
        assert n.x_nmda > 0

    def test_mg_block_voltage_dependent(self):
        n = CompteWMNeuron()
        b_low = n._mg_block(-80.0)
        b_high = n._mg_block(0.0)
        assert b_high > b_low, "Mg block not voltage-dependent"

    def test_gaba_on_spike(self):
        """Self-inhibition: s_gaba should increase when neuron spikes."""
        n = CompteWMNeuron()
        for _ in range(10000):
            if n.step(3.0):
                assert n.s_gaba > 0
                return
        raise AssertionError("no spike to test GABA")

    def test_state_finite(self):
        n = CompteWMNeuron()
        for _ in range(10000):
            n.step(5.0)
        for attr in ["v", "s_ampa", "s_nmda", "x_nmda", "s_gaba"]:
            assert np.isfinite(getattr(n, attr)), f"{attr} not finite"

    def test_reset(self):
        n = CompteWMNeuron()
        for _ in range(100):
            n.step(3.0, spike_in=True)
        n.reset()
        assert n.v == n.e_l
        assert n.s_nmda == 0.0


class TestCompteNetwork:
    def test_population(self):
        assert Population(CompteWMNeuron, n=10, label="wm").n == 10

    def test_network_spikes(self):
        pop = Population(CompteWMNeuron, n=20, label="wm")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0

    def test_with_projection(self):
        pop = Population(CompteWMNeuron, n=10, label="wm")
        proj = Projection(pop, pop, weight=0.5, probability=0.3, seed=42)
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, drive, mon)
        net.run(duration=0.3, dt=0.001, backend="python")
        assert isinstance(mon.spike_trains, dict)


class TestCompteAnalysis:
    def _get_train(self):
        n = CompteWMNeuron()
        train = np.zeros(10000, dtype=np.int8)
        for t in range(10000):
            train[t] = n.step(3.0)
        return train

    def test_firing_rate(self):
        assert firing_rate(self._get_train(), dt=0.0001) > 0

    def test_spike_count(self):
        assert spike_count(self._get_train()) > 10

    def test_isi(self):
        intervals = isi(self._get_train(), dt=0.0001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
