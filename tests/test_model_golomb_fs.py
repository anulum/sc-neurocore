# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: GolombFSNeuron

"""Full pipeline test for GolombFSNeuron (Golomb et al. 2007).

Fast-spiking interneuron with Kv3 potassium channel.
4 state variables: v, h, n, p. 10 sub-steps per step call."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.golomb_fs import GolombFSNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestGolombFSIsolation:
    def test_construction(self):
        n = GolombFSNeuron()
        assert n.v == -65.0
        assert n.g_kv3 == 150.0

    def test_step_returns_binary(self):
        assert GolombFSNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = GolombFSNeuron()
        assert sum(n.step(1.0) for _ in range(1000)) == 0

    def test_spikes_under_drive(self):
        n = GolombFSNeuron()
        assert sum(n.step(5.0) for _ in range(5000)) > 10

    def test_fast_spiking(self):
        """FS interneuron should sustain high rates without adaptation."""
        n = GolombFSNeuron()
        s = sum(n.step(10.0) for _ in range(5000))
        assert s > 20

    def test_rate_increases_with_input(self):
        n_low = GolombFSNeuron()
        n_high = GolombFSNeuron()
        s_low = sum(n_low.step(5.0) for _ in range(5000))
        s_high = sum(n_high.step(10.0) for _ in range(5000))
        assert s_high > s_low

    def test_kv3_gating(self):
        """Kv3 gate p should activate under drive."""
        n = GolombFSNeuron()
        for _ in range(2000):
            n.step(10.0)
        assert n.p > 0.01

    def test_gating_bounded(self):
        """All gating variables should stay in [0, 1]."""
        n = GolombFSNeuron()
        for _ in range(3000):
            n.step(10.0)
        for gate in [n.h, n.n, n.p]:
            assert 0.0 <= gate <= 1.0, f"gate {gate} out of [0,1]"

    def test_numerical_stability(self):
        for I in [0.0, 5.0, 10.0, 20.0]:
            n = GolombFSNeuron()
            for _ in range(3000):
                n.step(I)
            assert np.isfinite(n.v), f"v NaN at I={I}"

    def test_reset(self):
        n = GolombFSNeuron()
        for _ in range(2000):
            n.step(10.0)
        n.reset()
        assert n.v == -65.0
        assert n.p == 0.0


class TestGolombFSNetwork:
    def test_population(self):
        assert Population(GolombFSNeuron, n=5, label="fs").n == 5

    def test_network_spikes(self):
        pop = Population(GolombFSNeuron, n=5, label="fs")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0


class TestGolombFSAnalysis:
    def test_spike_count(self):
        n = GolombFSNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(10.0)
        assert spike_count(train) > 10
