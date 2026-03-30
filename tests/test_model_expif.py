# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: ExpIFNeuron

"""Full pipeline test for ExpIFNeuron (Fourcaud-Trocmé et al. 2003).

Exponential integrate-and-fire without adaptation.
Exponential voltage escape near rheobase v_rh with sharpness delta_t."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.expif import ExpIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestExpIFIsolation:
    def test_construction(self):
        n = ExpIFNeuron()
        assert n.v == -65.0
        assert n.delta_t == 2.0

    def test_step_returns_binary(self):
        assert ExpIFNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = ExpIFNeuron()
        assert sum(n.step(0.0) for _ in range(500)) == 0

    def test_spikes_under_drive(self):
        n = ExpIFNeuron()
        assert sum(n.step(50.0) for _ in range(2000)) > 5

    def test_rate_increases_with_input(self):
        n_low = ExpIFNeuron()
        n_high = ExpIFNeuron()
        s_low = sum(n_low.step(20.0) for _ in range(2000))
        s_high = sum(n_high.step(60.0) for _ in range(2000))
        assert s_high > s_low

    def test_exponential_escape(self):
        """Voltage near v_rh should trigger the exponential runaway."""
        n = ExpIFNeuron()
        n.v = n.v_rh - 1.0
        spikes = sum(n.step(15.0) for _ in range(200))
        assert spikes > 0

    def test_exp_clipping(self):
        """Extreme voltage should not overflow (np.clip guards exp argument)."""
        n = ExpIFNeuron()
        n.v = 1000.0
        result = n.step(0.0)
        assert result in (0, 1)
        assert np.isfinite(n.v)

    def test_negative_extreme(self):
        """Very negative voltage stays finite."""
        n = ExpIFNeuron()
        n.v = -1000.0
        for _ in range(100):
            n.step(0.0)
        assert np.isfinite(n.v)

    def test_numerical_stability(self):
        for I in [0, 10, 20, 40]:
            n = ExpIFNeuron()
            for _ in range(1000):
                n.step(float(I))
            assert np.isfinite(n.v), f"v NaN at I={I}"

    def test_reset(self):
        n = ExpIFNeuron()
        for _ in range(100):
            n.step(20.0)
        n.reset()
        assert n.v == n.v_rest

    def test_custom_params(self):
        n = ExpIFNeuron(delta_t=1.0, tau=10.0, v_rh=-60.0)
        assert n.delta_t == 1.0
        assert n.tau == 10.0
        for _ in range(500):
            n.step(15.0)
        assert np.isfinite(n.v)


class TestExpIFNetwork:
    def test_population(self):
        assert Population(ExpIFNeuron, n=10, label="expif").n == 10

    def test_network_spikes(self):
        pop = Population(ExpIFNeuron, n=10, label="expif")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0


class TestExpIFAnalysis:
    def test_spike_count(self):
        n = ExpIFNeuron()
        train = np.zeros(2000, dtype=np.int8)
        for t in range(2000):
            train[t] = n.step(50.0)
        assert spike_count(train) > 5
