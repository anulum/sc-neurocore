# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: HindmarshRoseNeuron

"""Full pipeline test for HindmarshRoseNeuron (Hindmarsh & Rose 1984).

3D chaotic bursting: dx/dt = y - x³ + bx² - z + I.
Slow z variable modulates burst-pause pattern."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.hindmarsh_rose import HindmarshRoseNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestHRIsolation:
    def test_construction(self):
        n = HindmarshRoseNeuron()
        assert n.x == -1.6
        assert n.r == 0.001

    def test_step_returns_binary(self):
        assert HindmarshRoseNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = HindmarshRoseNeuron()
        assert sum(n.step(0.5) for _ in range(5000)) == 0

    def test_spikes_under_drive(self):
        n = HindmarshRoseNeuron()
        assert sum(n.step(3.0) for _ in range(20000)) > 20

    def test_isi_variability(self):
        """ISIs should vary significantly — not perfectly regular."""
        n = HindmarshRoseNeuron()
        spike_times = []
        for t in range(20000):
            if n.step(3.0):
                spike_times.append(t)
        if len(spike_times) > 10:
            isis = np.diff(spike_times)
            assert np.max(isis) > 2 * np.min(isis)

    def test_three_state_variables(self):
        n = HindmarshRoseNeuron()
        x0, y0, z0 = n.x, n.y, n.z
        for _ in range(5000):
            n.step(3.0)
        assert n.x != x0
        assert n.y != y0
        assert n.z != z0

    def test_slow_z_dynamics(self):
        """z (r=0.001) should change slowly."""
        n = HindmarshRoseNeuron()
        z0 = n.z
        for _ in range(1000):
            n.step(3.0)
        assert abs(n.z - z0) < abs(n.x - (-1.6))

    def test_rate_increases_with_input(self):
        n_low = HindmarshRoseNeuron()
        n_high = HindmarshRoseNeuron()
        s_low = sum(n_low.step(2.0) for _ in range(20000))
        s_high = sum(n_high.step(5.0) for _ in range(20000))
        assert s_high > s_low

    def test_numerical_stability(self):
        for I in [0.0, 2.0, 3.0, 5.0]:
            n = HindmarshRoseNeuron()
            for _ in range(10000):
                n.step(I)
            assert np.isfinite(n.x), f"x NaN at I={I}"
            assert np.isfinite(n.y), f"y NaN at I={I}"
            assert np.isfinite(n.z), f"z NaN at I={I}"

    def test_bounded_orbit(self):
        n = HindmarshRoseNeuron()
        for _ in range(20000):
            n.step(3.0)
        assert abs(n.x) < 10.0
        assert abs(n.y) < 50.0

    def test_reset(self):
        n = HindmarshRoseNeuron()
        for _ in range(5000):
            n.step(3.0)
        n.reset()
        assert n.x == -1.6
        assert n.y == -10.0
        assert n.z == 2.0


class TestHRNetwork:
    def test_population(self):
        assert Population(HindmarshRoseNeuron, n=10, label="hr").n == 10

    def test_network_spikes(self):
        pop = Population(HindmarshRoseNeuron, n=5, label="hr")
        drive = PoissonInput(n=5, rate_hz=200.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0


class TestHRAnalysis:
    def test_spike_count(self):
        n = HindmarshRoseNeuron()
        train = np.zeros(20000, dtype=np.int8)
        for t in range(20000):
            train[t] = n.step(3.0)
        assert spike_count(train) > 20
