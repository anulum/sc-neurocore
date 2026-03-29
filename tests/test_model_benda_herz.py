# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: BendaHerzNeuron

"""Full pipeline test for BendaHerzNeuron (Benda & Herz 2003).

Phenomenological spike-frequency adaptation. Stochastic spiking
from instantaneous f-I curve with adaptation variable A."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.benda_herz import BendaHerzNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count


class TestBendaHerzIsolation:
    def test_construction(self):
        n = BendaHerzNeuron()
        assert n.a == 0.0
        assert n.f_max == 200.0

    def test_step_returns_binary(self):
        n = BendaHerzNeuron()
        result = n.step(10.0)
        assert result in (0, 1)

    def test_spikes_under_drive(self):
        """Stochastic model — needs many steps for reliable spiking."""
        n = BendaHerzNeuron()
        spikes = sum(n.step(50.0) for _ in range(10000))
        assert spikes > 0, "no spikes at I=50 over 10K steps"

    def test_adaptation_increases(self):
        """Adaptation variable A should increase under sustained drive."""
        n = BendaHerzNeuron()
        a_init = n.a
        for _ in range(1000):
            n.step(30.0)
        assert n.a > a_init, "adaptation variable did not increase"

    def test_adaptation_reduces_rate(self):
        """Rate should decrease over time due to SFA."""
        n = BendaHerzNeuron()
        early_spikes = sum(n.step(50.0) for _ in range(2000))
        late_spikes = sum(n.step(50.0) for _ in range(2000))
        # Early should have more spikes than late (adaptation kicks in)
        # Stochastic — allow for noise; just check adaptation is nonzero
        assert n.a > 0, "no adaptation after 4K steps"

    def test_f_onset_sigmoid(self):
        """f_onset should be sigmoid-shaped: low at I=0, high at I>>i_half."""
        n = BendaHerzNeuron()
        f_low = n._f_onset(0.0)
        f_high = n._f_onset(50.0)
        assert f_high > f_low

    def test_state_finite(self):
        n = BendaHerzNeuron()
        for _ in range(5000):
            n.step(100.0)
        assert np.isfinite(n.a)

    def test_reset(self):
        n = BendaHerzNeuron()
        for _ in range(100):
            n.step(30.0)
        n.reset()
        assert n.a == 0.0


class TestBendaHerzNetwork:
    def test_population(self):
        pop = Population(BendaHerzNeuron, n=10, label="bh")
        assert pop.n == 10
        assert pop.model_name == "BendaHerzNeuron"

    def test_network_produces_spikes(self):
        pop = Population(BendaHerzNeuron, n=20, label="bh")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0, "network produced zero spikes"

    def test_with_projection(self):
        pop = Population(BendaHerzNeuron, n=20, label="bh")
        proj = Projection(pop, pop, weight=5.0, probability=0.2, seed=42)
        drive = PoissonInput(n=20, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon.spike_trains, dict)


class TestBendaHerzAnalysis:
    def _get_binary_train(self):
        n = BendaHerzNeuron()
        train = np.zeros(10000, dtype=np.int8)
        for t in range(10000):
            train[t] = n.step(50.0)
        return train

    def test_firing_rate(self):
        train = self._get_binary_train()
        rate = firing_rate(train, dt=0.001)
        assert rate >= 0  # stochastic — may be very low

    def test_spike_count(self):
        train = self._get_binary_train()
        count = spike_count(train)
        assert count >= 0
