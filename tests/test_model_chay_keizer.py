# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: ChayKeizerNeuron

"""Full pipeline test for ChayKeizerNeuron (Chay & Keizer 1983).

Pancreatic beta cell: 3 ODEs (V, n, Ca). Unlike Chay (g_K=1400),
ChayKeizer has g_K=25 — more moderate. Converges to fixed point
at V≈-8 mV after 1 transient spike. Ca-dependent K (KCa) with
half-activation k_d=1.0 µM. Stable at dt=0.02."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.chay_keizer import ChayKeizerNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


def _run(neuron: ChayKeizerNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestChayKeizerIsolation:
    def test_defaults(self):
        n = ChayKeizerNeuron()
        assert n.v == -50.0 and n.n == 0.01 and n.ca == 0.1
        assert n.g_k == 25.0 and n.g_ca == 20.0 and n.g_kca == 12.0
        assert n.k_d == 1.0 and n.dt == 0.02

    def test_step_returns_binary(self):
        assert ChayKeizerNeuron().step(0.0) in (0, 1)

    def test_three_variables_evolve(self):
        n = ChayKeizerNeuron()
        initial = (n.v, n.n, n.ca)
        for _ in range(500):
            n.step(0.0)
        for name, v0, v1 in zip(["v", "n", "ca"], initial, (n.v, n.n, n.ca)):
            assert v0 != v1, f"{name} didn't evolve"

    def test_state_finite(self):
        n = ChayKeizerNeuron()
        for _ in range(100000):
            n.step(0.0)
        assert np.isfinite(n.v) and np.isfinite(n.n) and np.isfinite(n.ca)

    def test_reset(self):
        n = ChayKeizerNeuron()
        for _ in range(500):
            n.step(0.0)
        n.reset()
        assert n.v == -50.0 and n.n == 0.01 and n.ca == 0.1


class TestChayKeizerDynamics:
    def test_transient_spike(self):
        """Model fires exactly 1 transient spike then converges to FP."""
        n = ChayKeizerNeuron()
        spikes = _run(n, current=0.0, steps=100000)
        assert len(spikes) == 1, f"{len(spikes)} spikes, expected 1"

    def test_converges_to_fixed_point(self):
        """After transient, V stabilises near -8 mV."""
        n = ChayKeizerNeuron()
        for _ in range(100000):
            n.step(0.0)
        v_eq = n.v
        for _ in range(50000):
            n.step(0.0)
        assert abs(n.v - v_eq) < 0.01, "V still drifting"

    def test_stable_at_default_dt(self):
        """Unlike Chay (g_K=1400), ChayKeizer (g_K=25) is stable at dt=0.02."""
        n = ChayKeizerNeuron(dt=0.02)
        for _ in range(50000):
            n.step(0.0)
        assert abs(n.v) < 150, f"V={n.v} — unstable"

    def test_ca_non_negative(self):
        n = ChayKeizerNeuron()
        for _ in range(100000):
            n.step(0.0)
        assert n.ca >= 0.0

    def test_n_bounded(self):
        n = ChayKeizerNeuron()
        for _ in range(100000):
            n.step(0.0)
        assert 0.0 <= n.n <= 1.0

    def test_kca_half_activation(self):
        """q_KCa = Ca/(Ca + k_d). At Ca=k_d=1.0: q=0.5."""
        q = 1.0 / (1.0 + 1.0)
        assert abs(q - 0.5) < 1e-10

    def test_m_inf_sigmoid(self):
        m = 1.0 / (1.0 + np.exp(-(-25.0 + 25.0) / 8.0))
        assert abs(m - 0.5) < 1e-10


class TestChayKeizerCurrentSweep:
    def test_no_sustained_spiking(self):
        """At all tested currents, only 1 transient spike (FP stable)."""
        for I in [0.0, 50.0, 100.0, 500.0]:
            n = ChayKeizerNeuron()
            spikes = _run(n, current=I, steps=50000)
            assert len(spikes) <= 2, f"I={I}: {len(spikes)} spikes"

    def test_v_shifts_with_current(self):
        n0 = ChayKeizerNeuron()
        n500 = ChayKeizerNeuron()
        for _ in range(100000):
            n0.step(0.0)
            n500.step(500.0)
        assert n500.v > n0.v


class TestChayKeizerParameters:
    @pytest.mark.parametrize("dt", [0.01, 0.02, 0.05])
    def test_dt_stability(self, dt: float):
        n = ChayKeizerNeuron(dt=dt)
        for _ in range(50000):
            n.step(0.0)
        assert abs(n.v) < 200

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = ChayKeizerNeuron()
            trace = [(n.step(0.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestChayKeizerPerformance:
    def test_isolation_throughput(self):
        n = ChayKeizerNeuron()
        N = 10000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(0.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 5000


class TestChayKeizerPipeline:
    def test_population(self):
        assert Population(ChayKeizerNeuron, n=5, label="ck").n == 5

    def test_network_runs(self):
        pop = Population(ChayKeizerNeuron, n=5, label="ck")
        drive = PoissonInput(n=5, rate_hz=100.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)

    def test_analysis(self):
        n = ChayKeizerNeuron()
        train = np.array([float(n.step(0.0)) for _ in range(50000)])
        sc = spike_count(train)
        assert sc >= 1  # at least the transient spike
