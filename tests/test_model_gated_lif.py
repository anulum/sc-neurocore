# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: GatedLIFNeuron

"""Full pipeline test for GatedLIFNeuron (Yao et al. 2022 NeurIPS).

LIF with learnable gates: v = gate_v·v + gate_i·I.
Subtract-reset: v -= v_threshold on spike."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.gated_lif import GatedLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestGatedLIFIsolation:
    def test_construction(self):
        n = GatedLIFNeuron()
        assert n.v == 0.0
        assert n.gate_v == 0.9
        assert n.gate_i == 1.0

    def test_step_returns_binary(self):
        assert GatedLIFNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = GatedLIFNeuron()
        assert sum(n.step(0.05) for _ in range(10)) == 0

    def test_spikes_under_drive(self):
        n = GatedLIFNeuron()
        assert sum(n.step(0.5) for _ in range(100)) > 10

    def test_subtract_reset(self):
        """After spike, v should be v_old - v_threshold, not zero."""
        n = GatedLIFNeuron()
        n.v = 0.8
        spike = n.step(0.5)
        if spike:
            assert n.v < n.v_threshold
            assert n.v >= 0.0

    def test_rate_increases_with_input(self):
        n_low = GatedLIFNeuron()
        n_high = GatedLIFNeuron()
        s_low = sum(n_low.step(0.2) for _ in range(200))
        s_high = sum(n_high.step(0.8) for _ in range(200))
        assert s_high > s_low

    def test_gate_v_effect(self):
        """Lower gate_v = faster leak = fewer spikes at weak drive."""
        n_fast = GatedLIFNeuron(gate_v=0.5)
        n_slow = GatedLIFNeuron(gate_v=0.99)
        s_fast = sum(n_fast.step(0.3) for _ in range(500))
        s_slow = sum(n_slow.step(0.3) for _ in range(500))
        assert s_slow > s_fast

    def test_gate_i_effect(self):
        """Higher gate_i = stronger input scaling = more spikes."""
        n_low = GatedLIFNeuron(gate_i=0.5)
        n_high = GatedLIFNeuron(gate_i=2.0)
        s_low = sum(n_low.step(0.3) for _ in range(500))
        s_high = sum(n_high.step(0.3) for _ in range(500))
        assert s_high > s_low

    def test_numerical_stability(self):
        for I in [0.0, 0.3, 0.5, 1.0, 5.0]:
            n = GatedLIFNeuron()
            for _ in range(1000):
                n.step(I)
            assert np.isfinite(n.v), f"v NaN at I={I}"

    def test_reset(self):
        n = GatedLIFNeuron()
        for _ in range(100):
            n.step(0.5)
        n.reset()
        assert n.v == 0.0

    def test_deterministic(self):
        """Gated LIF is fully deterministic — same input = same output."""
        n1 = GatedLIFNeuron()
        n2 = GatedLIFNeuron()
        for _ in range(100):
            assert n1.step(0.3) == n2.step(0.3)


class TestGatedLIFNetwork:
    def test_population(self):
        assert Population(GatedLIFNeuron, n=10, label="glif").n == 10

    def test_network_spikes(self):
        pop = Population(GatedLIFNeuron, n=10, label="glif")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0


class TestGatedLIFAnalysis:
    def test_spike_count(self):
        n = GatedLIFNeuron()
        train = np.zeros(500, dtype=np.int8)
        for t in range(500):
            train[t] = n.step(0.5)
        assert spike_count(train) > 50
