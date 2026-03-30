# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: MihalasNieburNeuron

"""Full pipeline test for MihalasNieburNeuron (Mihalas & Niebur 2009).

Generalized IF with dynamic threshold + 2 adaptation currents.
Captures 20 spike patterns via parameter configuration."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.mihalas_niebur import MihalasNieburNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestMNIsolation:
    def test_construction(self):
        n = MihalasNieburNeuron()
        assert n.v == 0.0
        assert n.theta == 1.0
        assert n.i1 == 0.0
        assert n.i2 == 0.0

    def test_step_returns_binary(self):
        assert MihalasNieburNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = MihalasNieburNeuron()
        assert sum(n.step(0.5) for _ in range(2000)) == 0

    def test_spikes_under_drive(self):
        n = MihalasNieburNeuron()
        assert sum(n.step(2.0) for _ in range(5000)) > 100

    def test_dynamic_threshold(self):
        """theta should evolve when a != 0."""
        n = MihalasNieburNeuron(a=0.01)
        theta_init = n.theta
        for _ in range(2000):
            n.step(2.0)
        assert n.theta != theta_init

    def test_adaptation_currents(self):
        """r1 > 0 should increment i1 on spike."""
        n = MihalasNieburNeuron(r1=1.0)
        for _ in range(5000):
            if n.step(2.0):
                assert n.i1 > 0
                break

    def test_i1_decays(self):
        n = MihalasNieburNeuron()
        n.i1 = 5.0
        for _ in range(200):
            n.step(0.0)
        assert n.i1 < 1.0

    def test_i2_decays_slower(self):
        """tau_2 > tau_1 → i2 decays slower."""
        n = MihalasNieburNeuron()
        n.i1 = 5.0
        n.i2 = 5.0
        for _ in range(50):
            n.step(0.0)
        assert n.i2 > n.i1

    def test_rate_increases_with_input(self):
        n_low = MihalasNieburNeuron()
        n_high = MihalasNieburNeuron()
        s_low = sum(n_low.step(1.5) for _ in range(5000))
        s_high = sum(n_high.step(5.0) for _ in range(5000))
        assert s_high > s_low

    def test_tonic_config(self):
        """Negative r1/r2 should produce adapting (fewer spikes over time)."""
        n = MihalasNieburNeuron(r1=-0.5, r2=-0.1)
        s1 = sum(n.step(2.0) for _ in range(2500))
        s2 = sum(n.step(2.0) for _ in range(2500))
        assert s1 >= s2

    def test_numerical_stability(self):
        for I in [0.0, 2.0, 5.0]:
            n = MihalasNieburNeuron()
            for _ in range(5000):
                n.step(I)
            assert np.isfinite(n.v)
            assert np.isfinite(n.theta)

    def test_reset(self):
        n = MihalasNieburNeuron()
        for _ in range(2000):
            n.step(2.0)
        n.reset()
        assert n.v == n.v_rest
        assert n.theta == n.theta_reset
        assert n.i1 == 0.0
        assert n.i2 == 0.0

    def test_deterministic(self):
        n1 = MihalasNieburNeuron()
        n2 = MihalasNieburNeuron()
        for _ in range(500):
            assert n1.step(2.0) == n2.step(2.0)


class TestMNNetwork:
    def test_population(self):
        assert Population(MihalasNieburNeuron, n=10, label="mn").n == 10

    def test_network_spikes(self):
        pop = Population(MihalasNieburNeuron, n=10, label="mn")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0


class TestMNAnalysis:
    def test_spike_count(self):
        n = MihalasNieburNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(2.0)
        assert spike_count(train) > 100
