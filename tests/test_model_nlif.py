# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: NonlinearLIFNeuron

"""Full pipeline test for NonlinearLIFNeuron (Touboul & Brette 2008).

Cubic nonlinearity: a*(V-V_rest)*(V-V_crit) + adaptation w."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.nlif import NonlinearLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestNLIFIsolation:
    def test_construction(self):
        n = NonlinearLIFNeuron()
        assert n.v == -65.0
        assert n.w == 0.0

    def test_step_returns_binary(self):
        assert NonlinearLIFNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = NonlinearLIFNeuron()
        assert sum(n.step(3.0) for _ in range(2000)) == 0

    def test_spikes_under_drive(self):
        n = NonlinearLIFNeuron()
        assert sum(n.step(20.0) for _ in range(5000)) > 50

    def test_cubic_nonlinearity(self):
        """V above V_crit triggers runaway (a*(V-V_rest)*(V-V_crit) > 0)."""
        n = NonlinearLIFNeuron()
        n.v = -35.0
        cubic = n.a * (n.v - n.v_rest) * (n.v - n.v_crit)
        assert cubic > 0

    def test_w_adaptation(self):
        n = NonlinearLIFNeuron()
        for _ in range(3000):
            n.step(20.0)
        assert n.w != 0.0

    def test_rate_increases(self):
        n_low = NonlinearLIFNeuron()
        n_high = NonlinearLIFNeuron()
        s_low = sum(n_low.step(10.0) for _ in range(5000))
        s_high = sum(n_high.step(30.0) for _ in range(5000))
        assert s_high > s_low

    def test_numerical_stability(self):
        for I in [0.0, 10.0, 20.0, 50.0]:
            n = NonlinearLIFNeuron()
            for _ in range(5000):
                n.step(I)
            assert np.isfinite(n.v)
            assert np.isfinite(n.w)

    def test_reset(self):
        n = NonlinearLIFNeuron()
        for _ in range(2000):
            n.step(20.0)
        n.reset()
        assert n.v == n.v_rest
        assert n.w == 0.0

    def test_deterministic(self):
        n1 = NonlinearLIFNeuron()
        n2 = NonlinearLIFNeuron()
        for _ in range(500):
            assert n1.step(15.0) == n2.step(15.0)


class TestNLIFNetwork:
    def test_population(self):
        assert Population(NonlinearLIFNeuron, n=10, label="nlif").n == 10

    def test_network_spikes(self):
        pop = Population(NonlinearLIFNeuron, n=5, label="nlif")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0


class TestNLIFAnalysis:
    def test_spike_count(self):
        n = NonlinearLIFNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(20.0)
        assert spike_count(train) > 50
