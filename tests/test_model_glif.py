# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: GLIFNeuron

"""Full pipeline test for GLIFNeuron (Teeter et al. 2018, Allen Institute).

GLIF5: LIF + dynamic threshold + 2 after-spike currents.
5 state variables: v, theta, i_asc1, i_asc2, theta_inf."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.glif import GLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestGLIFIsolation:
    def test_construction(self):
        n = GLIFNeuron()
        assert n.v == -70.0
        assert n.theta == -50.0
        assert n.i_asc1 == 0.0
        assert n.i_asc2 == 0.0

    def test_step_returns_binary(self):
        assert GLIFNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = GLIFNeuron()
        assert sum(n.step(10.0) for _ in range(1000)) == 0

    def test_spikes_under_drive(self):
        n = GLIFNeuron()
        assert sum(n.step(30.0) for _ in range(5000)) > 100

    def test_rate_increases_with_input(self):
        n_low = GLIFNeuron()
        n_high = GLIFNeuron()
        s_low = sum(n_low.step(25.0) for _ in range(5000))
        s_high = sum(n_high.step(50.0) for _ in range(5000))
        assert s_high > s_low

    def test_threshold_adaptation(self):
        """Theta should increase after spiking (delta_theta > 0)."""
        n = GLIFNeuron()
        theta_init = n.theta
        for _ in range(5000):
            if n.step(30.0):
                assert n.theta > theta_init
                break

    def test_after_spike_currents(self):
        """i_asc1 and i_asc2 should increase after spike."""
        n = GLIFNeuron()
        for _ in range(5000):
            if n.step(50.0):
                assert n.i_asc1 > 0
                assert n.i_asc2 > 0
                break

    def test_asc_decay(self):
        """After-spike currents should decay without spiking."""
        n = GLIFNeuron()
        n.i_asc1 = 10.0
        n.i_asc2 = 10.0
        for _ in range(1000):
            n.step(0.0)
        assert n.i_asc1 < 1.0
        assert n.i_asc2 < 5.0

    def test_numerical_stability(self):
        for I in [0.0, 30.0, 50.0, 100.0]:
            n = GLIFNeuron()
            for _ in range(5000):
                n.step(I)
            assert np.isfinite(n.v), f"v NaN at I={I}"
            assert np.isfinite(n.theta), f"theta NaN at I={I}"
            assert np.isfinite(n.i_asc1), f"i_asc1 NaN at I={I}"
            assert np.isfinite(n.i_asc2), f"i_asc2 NaN at I={I}"

    def test_reset(self):
        n = GLIFNeuron()
        for _ in range(2000):
            n.step(30.0)
        n.reset()
        assert n.v == n.v_rest
        assert n.theta == n.theta_inf
        assert n.i_asc1 == 0.0
        assert n.i_asc2 == 0.0

    def test_deterministic(self):
        n1 = GLIFNeuron()
        n2 = GLIFNeuron()
        for _ in range(500):
            assert n1.step(30.0) == n2.step(30.0)


class TestGLIFNetwork:
    def test_population(self):
        assert Population(GLIFNeuron, n=10, label="glif5").n == 10

    def test_network_spikes(self):
        pop = Population(GLIFNeuron, n=10, label="glif5")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=30.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0


class TestGLIFAnalysis:
    def test_spike_count(self):
        n = GLIFNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(30.0)
        assert spike_count(train) > 100
