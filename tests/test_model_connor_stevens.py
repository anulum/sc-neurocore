# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: ConnorStevensNeuron

"""Full pipeline test for ConnorStevensNeuron (Connor & Stevens 1977).

Type-I excitability with A-type K⁺ current. 100 sub-steps per step."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.connor_stevens import ConnorStevensNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestConnorStevensIsolation:
    def test_construction(self):
        n = ConnorStevensNeuron()
        assert n.v == -68.0

    def test_step_returns_binary(self):
        assert ConnorStevensNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = ConnorStevensNeuron()
        assert sum(n.step(5.0) for _ in range(500)) == 0

    def test_spikes(self):
        n = ConnorStevensNeuron()
        assert sum(n.step(10.0) for _ in range(1000)) > 5

    def test_type_I_excitability(self):
        """Type-I: firing rate increases continuously from zero near threshold."""
        n = ConnorStevensNeuron()
        rate_low = sum(n.step(8.0) for _ in range(500))
        n.reset()
        rate_high = sum(n.step(20.0) for _ in range(500))
        assert rate_high > rate_low

    def test_a_type_current(self):
        """A-type K⁺ gating variables should change under drive."""
        n = ConnorStevensNeuron()
        a_init, b_init = n.a, n.b
        for _ in range(500):
            n.step(15.0)
        assert n.a != a_init or n.b != b_init

    def test_numerical_stability(self):
        for I in [0, 10, 20]:
            n = ConnorStevensNeuron()
            for _ in range(500):
                n.step(float(I))
            for attr in ["v", "m", "h", "n", "a", "b"]:
                assert np.isfinite(getattr(n, attr)), f"{attr} NaN at I={I}"

    def test_reset(self):
        n = ConnorStevensNeuron()
        for _ in range(100):
            n.step(15.0)
        n.reset()
        assert n.v == -68.0
        assert n.m == 0.01


class TestConnorStevensNetwork:
    def test_population(self):
        assert Population(ConnorStevensNeuron, n=5, label="cs").n == 5

    def test_network_spikes(self):
        pop = Population(ConnorStevensNeuron, n=5, label="cs")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=15.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.2, dt=0.001, backend="python")
        assert mon.count > 0


class TestConnorStevensAnalysis:
    def test_spike_count(self):
        n = ConnorStevensNeuron()
        train = np.zeros(1000, dtype=np.int8)
        for t in range(1000):
            train[t] = n.step(15.0)
        assert spike_count(train) > 5
