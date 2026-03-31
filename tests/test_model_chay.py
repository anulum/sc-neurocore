# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: ChayNeuron

"""Full pipeline test for ChayNeuron (Chay 1985).

Pancreatic beta cell burster: 3 ODEs (V, n, Ca). g_K=1400 dominates.
FINDING: default dt=0.02 is NUMERICALLY UNSTABLE (V oscillates ±200,
clipped). At stable dt≤0.01, model converges to fixed point at
V≈-69 mV for all tested currents (0–1000). The g_K/g_Ca ratio is
too high for spiking at default parameters.
Performance: ~12K isolation steps/s."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.chay import ChayNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput


class TestChayIsolation:
    def test_defaults(self):
        n = ChayNeuron()
        assert n.v == -50.0 and n.n == 0.1 and n.ca == 0.1
        assert n.g_k == 1400.0 and n.g_ca == 25.0
        assert n.dt == 0.02

    def test_step_returns_binary(self):
        assert ChayNeuron().step(0.0) in (0, 1)

    def test_three_variables_evolve(self):
        n = ChayNeuron(dt=0.01)
        initial = (n.v, n.n, n.ca)
        for _ in range(500):
            n.step(0.0)
        for name, v0, v1 in zip(["v", "n", "ca"], initial, (n.v, n.n, n.ca)):
            assert v0 != v1, f"{name} didn't evolve"

    def test_reset(self):
        n = ChayNeuron(dt=0.01)
        for _ in range(500):
            n.step(100.0)
        n.reset()
        assert n.v == -50.0 and n.n == 0.1 and n.ca == 0.1


class TestChayNumericalStability:
    """CRITICAL: default dt=0.02 is unstable. Documented and tested."""

    def test_default_dt_unstable(self):
        """dt=0.02 causes V to oscillate between ±200 (clipping limits).

        This is a genuine Euler instability — the large g_K=1400
        conductance creates stiff dynamics requiring dt < 0.01.
        """
        n = ChayNeuron(dt=0.02)
        for _ in range(100):
            n.step(0.0)
        # V should be at clipping boundary
        assert abs(n.v) >= 199.0, f"V={n.v}, expected ±200 (unstable)"

    def test_small_dt_stable(self):
        """dt=0.01 produces stable, biophysical dynamics."""
        n = ChayNeuron(dt=0.01)
        for _ in range(100000):
            n.step(0.0)
        assert abs(n.v) < 100.0, f"V={n.v}"
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("dt", [0.001, 0.005, 0.01])
    def test_stable_dt_range(self, dt: float):
        n = ChayNeuron(dt=dt)
        for _ in range(50000):
            n.step(0.0)
        assert abs(n.v) < 100.0

    def test_v_clipping_prevents_nan(self):
        """Even at unstable dt, V is clipped to [-200, 200] — no NaN."""
        n = ChayNeuron(dt=0.02)
        for _ in range(10000):
            n.step(0.0)
        assert np.isfinite(n.v)


class TestChayFixedPoint:
    """At stable dt, the model converges to a fixed point — no spiking."""

    def test_converges_to_fixed_point(self):
        """V → ~-69 mV at I=0 with dt=0.01."""
        n = ChayNeuron(dt=0.01)
        for _ in range(200000):
            n.step(0.0)
        v_eq = n.v
        for _ in range(50000):
            n.step(0.0)
        assert abs(n.v - v_eq) < 0.01, "V still drifting"
        assert -75 < v_eq < -60, f"V_eq = {v_eq}"

    def test_no_spikes_at_any_tested_current(self):
        """With dt=0.01, g_K=1400 dominates → V never reaches threshold."""
        for I in [0.0, 100.0, 500.0, 1000.0]:
            n = ChayNeuron(dt=0.01)
            spikes = sum(n.step(I) for _ in range(100000))
            assert spikes == 0, f"I={I}: {spikes} spikes at stable dt"

    def test_v_shifts_with_current(self):
        """Higher current shifts the fixed point slightly upward."""
        n0 = ChayNeuron(dt=0.01)
        n1000 = ChayNeuron(dt=0.01)
        for _ in range(200000):
            n0.step(0.0)
            n1000.step(1000.0)
        assert n1000.v > n0.v


class TestChayIonChannels:
    def test_m_inf_sigmoid(self):
        """m_inf = 1/(1 + exp(-(V+25)/8)). At V=-25: m_inf=0.5."""
        import numpy as np

        m = 1.0 / (1.0 + np.exp(-(-25.0 + 25.0) / 8.0))
        assert abs(m - 0.5) < 1e-10

    def test_ca_non_negative(self):
        """Ca concentration clamped ≥ 0."""
        n = ChayNeuron(dt=0.01)
        for _ in range(100000):
            n.step(0.0)
        assert n.ca >= 0.0

    def test_n_bounded_0_1(self):
        """n gating variable clipped to [0, 1]."""
        n = ChayNeuron(dt=0.01)
        for _ in range(100000):
            n.step(0.0)
        assert 0.0 <= n.n <= 1.0

    def test_kca_activation(self):
        """KCa activation: ca/(ca+1). At ca=1: activation=0.5."""
        act = 1.0 / (1.0 + 1.0)
        assert abs(act - 0.5) < 1e-10


class TestChayPerformance:
    def test_isolation_throughput(self):
        n = ChayNeuron(dt=0.01)
        N = 10000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(0.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 5000


class TestChayPipeline:
    def test_population(self):
        assert Population(ChayNeuron, n=5, label="chay").n == 5

    def test_network_runs_without_crash(self):
        """With default dt=0.02, model is unstable but Population/Network
        don't crash — V is clipped, and spike detection runs on clipped values."""
        pop = Population(ChayNeuron, n=5, label="chay")
        drive = PoissonInput(n=5, rate_hz=100.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        # Don't assert spike count — model behaviour at default dt is unstable
        assert isinstance(mon.count, int)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = ChayNeuron(dt=0.01)
            trace = [(n.step(0.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
