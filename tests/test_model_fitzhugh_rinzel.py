# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: FitzHughRinzelNeuron

"""Full pipeline test for FitzHughRinzelNeuron (FitzHugh 1976, Rinzel 1987).

3D extension of FHN — adds ultra-slow variable y for bursting dynamics.
dv/dt = v - v³/3 - w + y + I,  dw/dt = δ(a+v-bw),  dy/dt = μ(c-v-dy)."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.fitzhugh_rinzel import FitzHughRinzelNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestFHRIsolation:
    def test_construction(self):
        n = FitzHughRinzelNeuron()
        assert n.v == -1.0
        assert n.y == 0.0

    def test_step_returns_binary(self):
        assert FitzHughRinzelNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = FitzHughRinzelNeuron()
        assert sum(n.step(0.2) for _ in range(5000)) <= 1

    def test_spikes_under_drive(self):
        n = FitzHughRinzelNeuron()
        assert sum(n.step(0.8) for _ in range(20000)) > 20

    def test_three_state_variables(self):
        """All three state variables should evolve under drive."""
        n = FitzHughRinzelNeuron()
        v0, w0, y0 = n.v, n.w, n.y
        for _ in range(5000):
            n.step(0.8)
        assert n.v != v0
        assert n.w != w0
        assert n.y != y0

    def test_slow_variable_drift(self):
        """y (ultra-slow, µ=0.0001) should drift slowly over many steps."""
        n = FitzHughRinzelNeuron()
        for _ in range(10000):
            n.step(0.8)
        assert abs(n.y) > 1e-4

    def test_numerical_stability(self):
        for I in [0.0, 0.5, 0.8, 1.0]:
            n = FitzHughRinzelNeuron()
            for _ in range(10000):
                n.step(I)
            assert np.isfinite(n.v), f"v NaN at I={I}"
            assert np.isfinite(n.w), f"w NaN at I={I}"
            assert np.isfinite(n.y), f"y NaN at I={I}"

    def test_bounded_orbit(self):
        n = FitzHughRinzelNeuron()
        for _ in range(20000):
            n.step(0.8)
        assert abs(n.v) < 5.0
        assert abs(n.w) < 5.0
        assert abs(n.y) < 5.0

    def test_reset(self):
        n = FitzHughRinzelNeuron()
        for _ in range(5000):
            n.step(0.8)
        n.reset()
        assert n.v == -1.0
        assert n.w == -0.5
        assert n.y == 0.0

    def test_custom_mu(self):
        """Higher µ → faster y dynamics, still stable."""
        n = FitzHughRinzelNeuron(mu=0.01)
        for _ in range(5000):
            n.step(0.8)
        assert np.isfinite(n.v)
        assert np.isfinite(n.y)


class TestFHRNetwork:
    def test_population(self):
        assert Population(FitzHughRinzelNeuron, n=10, label="fhr").n == 10

    def test_network_spikes(self):
        pop = Population(FitzHughRinzelNeuron, n=10, label="fhr")
        drive = PoissonInput(n=10, rate_hz=200.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0


class TestFHRAnalysis:
    def test_spike_count(self):
        n = FitzHughRinzelNeuron()
        train = np.zeros(20000, dtype=np.int8)
        for t in range(20000):
            train[t] = n.step(0.8)
        assert spike_count(train) > 20
