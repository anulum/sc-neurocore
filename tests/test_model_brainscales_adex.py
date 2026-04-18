# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: BrainScaleSAdExNeuron

"""Full pipeline test for BrainScaleSAdExNeuron (Schemmel 2010).

BrainScaleS-2 analog AdEx with 1000× hardware speedup emulation.
Clipped exponential for numerical safety."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.brainscales_adex import BrainScaleSAdExNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count, isi


class TestBrainScaleSIsolation:
    def test_construction(self):
        n = BrainScaleSAdExNeuron()
        assert n.v == -65.0
        assert n.hw_speedup == 1000.0

    def test_step_returns_binary(self):
        n = BrainScaleSAdExNeuron()
        assert n.step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = BrainScaleSAdExNeuron()
        spikes = sum(n.step(5.0) for _ in range(10_000))
        assert spikes == 0, f"unexpected spikes at I=5: {spikes}"

    def test_spikes_under_drive(self):
        n = BrainScaleSAdExNeuron()
        spikes = sum(n.step(20.0) for _ in range(10_000))
        assert spikes > 0, "no spikes at I=20"

    def test_adaptation_variable(self):
        """w should increase after spiking (b=7 increment)."""
        n = BrainScaleSAdExNeuron()
        w_init = n.w
        for _ in range(10_000):
            n.step(20.0)
        assert n.w > w_init

    def test_exp_clipped(self):
        """Exponential term should not overflow (clipped to [-20, 20])."""
        n = BrainScaleSAdExNeuron()
        n.v = 1000.0  # extreme voltage
        result = n.step(0.0)
        assert np.isfinite(n.v)

    def test_state_finite(self):
        n = BrainScaleSAdExNeuron()
        for _ in range(10_000):
            n.step(30.0)
        assert np.isfinite(n.v)
        assert np.isfinite(n.w)

    def test_reset(self):
        n = BrainScaleSAdExNeuron()
        for _ in range(1000):
            n.step(20.0)
        n.reset()
        assert n.v == n.v_rest
        assert n.w == 0.0


class TestBrainScaleSNetwork:
    def test_population(self):
        pop = Population(BrainScaleSAdExNeuron, n=10, label="bs")
        assert pop.n == 10
        assert pop.model_name == "BrainScaleSAdExNeuron"

    def test_network_spikes(self):
        pop = Population(BrainScaleSAdExNeuron, n=20, label="bs")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=40.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0, "network zero spikes"

    def test_with_projection(self):
        pop = Population(BrainScaleSAdExNeuron, n=10, label="bs")
        proj = Projection(pop, pop, weight=2.0, probability=0.3, seed=42)
        drive = PoissonInput(n=10, rate_hz=500.0, weight=40.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert isinstance(mon.spike_trains, dict)


class TestBrainScaleSAnalysis:
    def _get_binary_train(self):
        n = BrainScaleSAdExNeuron()
        train = np.zeros(10_000, dtype=np.int8)
        for t in range(10_000):
            train[t] = n.step(25.0)
        return train

    def test_firing_rate(self):
        train = self._get_binary_train()
        rate = firing_rate(train, dt=0.0001)
        assert rate >= 0

    def test_spike_count(self):
        train = self._get_binary_train()
        assert spike_count(train) >= 0

    def test_isi(self):
        train = self._get_binary_train()
        intervals = isi(train, dt=0.0001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
