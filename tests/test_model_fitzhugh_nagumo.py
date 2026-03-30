# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: FitzHughNagumoNeuron

"""Full pipeline test for FitzHughNagumoNeuron (FitzHugh 1961, Nagumo 1962).

2D qualitative spike model: cubic v-nullcline + linear w-nullcline.
dv/dt = v - v³/3 - w + I,  dw/dt = ε(v + a - bw)."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.fitzhugh_nagumo import FitzHughNagumoNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestFHNIsolation:
    def test_construction(self):
        n = FitzHughNagumoNeuron()
        assert n.v == -1.0
        assert n.w == -0.5

    def test_step_returns_binary(self):
        assert FitzHughNagumoNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        """Below oscillatory range — model settles to stable fixed point."""
        n = FitzHughNagumoNeuron()
        assert sum(n.step(0.2) for _ in range(2000)) <= 1

    def test_spikes_under_drive(self):
        """Sufficient current triggers oscillatory spiking."""
        n = FitzHughNagumoNeuron()
        assert sum(n.step(0.8) for _ in range(10000)) > 10

    def test_oscillatory_band(self):
        """FHN oscillates in a band of I; too low or too high = no spikes."""
        n_mid = FitzHughNagumoNeuron()
        s_mid = sum(n_mid.step(0.8) for _ in range(10000))
        n_high = FitzHughNagumoNeuron()
        s_high = sum(n_high.step(2.0) for _ in range(10000))
        assert s_mid > 10
        assert s_high < s_mid

    def test_oscillatory_dynamics(self):
        """v should oscillate (cross zero multiple times) under drive."""
        n = FitzHughNagumoNeuron()
        signs = []
        for _ in range(10000):
            n.step(0.8)
            signs.append(n.v > 0)
        transitions = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
        assert transitions > 10

    def test_w_recovery(self):
        """Recovery variable w should change under strong drive."""
        n = FitzHughNagumoNeuron()
        w_init = n.w
        for _ in range(2000):
            n.step(1.5)
        assert n.w != w_init

    def test_numerical_stability(self):
        for I in [0.0, 0.5, 0.8, 1.0, 2.0]:
            n = FitzHughNagumoNeuron()
            for _ in range(5000):
                n.step(I)
            assert np.isfinite(n.v), f"v NaN at I={I}"
            assert np.isfinite(n.w), f"w NaN at I={I}"

    def test_bounded_orbit(self):
        """FHN orbits are bounded — |v| should stay < 5 for reasonable I."""
        n = FitzHughNagumoNeuron()
        for _ in range(10000):
            n.step(0.8)
        assert abs(n.v) < 5.0
        assert abs(n.w) < 5.0

    def test_reset(self):
        n = FitzHughNagumoNeuron()
        for _ in range(1000):
            n.step(0.8)
        n.reset()
        assert n.v == -1.0
        assert n.w == -0.5

    def test_custom_epsilon(self):
        """Faster recovery (higher ε) should still be stable."""
        n = FitzHughNagumoNeuron(epsilon=0.3)
        for _ in range(3000):
            n.step(0.8)
        assert np.isfinite(n.v)


class TestFHNNetwork:
    def test_population(self):
        assert Population(FitzHughNagumoNeuron, n=10, label="fhn").n == 10

    def test_network_spikes(self):
        pop = Population(FitzHughNagumoNeuron, n=10, label="fhn")
        drive = PoissonInput(n=10, rate_hz=200.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0


class TestFHNAnalysis:
    def test_spike_count(self):
        n = FitzHughNagumoNeuron()
        train = np.zeros(10000, dtype=np.int8)
        for t in range(10000):
            train[t] = n.step(0.8)
        assert spike_count(train) > 10
