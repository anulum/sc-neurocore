# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: InhibitoryLIFNeuron

"""Full pipeline test for InhibitoryLIFNeuron.

LIF with post-spike inhibitory trace: suppresses membrane for temporal coding."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.ilif import InhibitoryLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestILIFIsolation:
    def test_construction(self):
        n = InhibitoryLIFNeuron()
        assert n.v == 0.0
        assert n.inh_trace == 0.0

    def test_step_returns_binary(self):
        assert InhibitoryLIFNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = InhibitoryLIFNeuron()
        assert sum(n.step(0.0) for _ in range(1000)) == 0

    def test_spikes_under_drive(self):
        n = InhibitoryLIFNeuron()
        assert sum(n.step(0.5) for _ in range(1000)) > 50

    def test_rate_increases_with_input(self):
        n_low = InhibitoryLIFNeuron()
        n_high = InhibitoryLIFNeuron()
        s_low = sum(n_low.step(0.3) for _ in range(2000))
        s_high = sum(n_high.step(1.0) for _ in range(2000))
        assert s_high > s_low

    def test_inhibitory_trace_increases(self):
        """Trace should increase after spiking."""
        n = InhibitoryLIFNeuron()
        for _ in range(1000):
            if n.step(0.5):
                assert n.inh_trace > 0
                break

    def test_inhibitory_trace_decays(self):
        """Without spikes, trace should decay."""
        n = InhibitoryLIFNeuron()
        n.inh_trace = 5.0
        for _ in range(100):
            n.step(0.0)
        assert n.inh_trace < 1.0

    def test_inhibition_reduces_rate(self):
        """Stronger inhibition should reduce spike rate."""
        n_weak = InhibitoryLIFNeuron(inh_strength=0.1)
        n_strong = InhibitoryLIFNeuron(inh_strength=2.0)
        s_weak = sum(n_weak.step(0.5) for _ in range(5000))
        s_strong = sum(n_strong.step(0.5) for _ in range(5000))
        assert s_weak > s_strong

    def test_alpha_precomputed(self):
        n = InhibitoryLIFNeuron()
        assert 0.0 < n.alpha_m < 1.0
        assert 0.0 < n.alpha_inh < 1.0

    def test_numerical_stability(self):
        for I in [0.0, 0.5, 1.0, 5.0]:
            n = InhibitoryLIFNeuron()
            for _ in range(5000):
                n.step(I)
            assert np.isfinite(n.v), f"v NaN at I={I}"
            assert np.isfinite(n.inh_trace), f"inh NaN at I={I}"

    def test_reset(self):
        n = InhibitoryLIFNeuron()
        for _ in range(500):
            n.step(0.5)
        n.reset()
        assert n.v == 0.0
        assert n.inh_trace == 0.0

    def test_deterministic(self):
        n1 = InhibitoryLIFNeuron()
        n2 = InhibitoryLIFNeuron()
        for _ in range(500):
            assert n1.step(0.5) == n2.step(0.5)


class TestILIFNetwork:
    def test_population(self):
        assert Population(InhibitoryLIFNeuron, n=10, label="ilif").n == 10

    def test_network_spikes(self):
        pop = Population(InhibitoryLIFNeuron, n=10, label="ilif")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0


class TestILIFAnalysis:
    def test_spike_count(self):
        n = InhibitoryLIFNeuron()
        train = np.zeros(2000, dtype=np.int8)
        for t in range(2000):
            train[t] = n.step(0.5)
        assert spike_count(train) > 100
