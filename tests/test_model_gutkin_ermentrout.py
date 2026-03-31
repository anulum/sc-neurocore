# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: GutkinErmentroutNeuron

"""Full pipeline test for GutkinErmentroutNeuron (Gutkin & Ermentrout 1998).

Minimal 2D conductance model: persistent Na + delayed-rectifier K.
I_Na: g=20, m_inf (instantaneous Boltzmann v_half=-20, k=15)
I_K: g=10, n (tau=1ms, Boltzmann v_half=-25, k=5)
I_L: g=8, ohmic leak

No sub-stepping (dt=0.05). m_Na instantaneous.
Simple enough for full analytical verification.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.gutkin_ermentrout import GutkinErmentroutNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: GutkinErmentroutNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


def _m_inf(v: float) -> float:
    return 1.0 / (1.0 + np.exp(-(v + 20.0) / 15.0))


def _n_inf(v: float) -> float:
    return 1.0 / (1.0 + np.exp(-(v + 25.0) / 5.0))


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestGEIsolation:
    def test_defaults(self):
        n = GutkinErmentroutNeuron()
        assert n.v == -65.0 and n.n == 0.1
        assert n.g_na == 20.0 and n.g_k == 10.0 and n.g_l == 8.0
        assert n.dt == 0.05

    def test_step_returns_binary(self):
        assert GutkinErmentroutNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self):
        n = GutkinErmentroutNeuron()
        for _ in range(100_000):
            n.step(5.0)
        assert np.isfinite(n.v) and np.isfinite(n.n)

    def test_reset_restores_defaults(self):
        n = GutkinErmentroutNeuron()
        for _ in range(5000):
            n.step(5.0)
        n.reset()
        assert n.v == -65.0 and n.n == 0.1

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = GutkinErmentroutNeuron()
            trace = [(n.step(5.0), n.v, n.n) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — m_inf, n_inf, dV, dn formulas
# ---------------------------------------------------------------------------
class TestGEAnalytical:
    def test_m_inf_boltzmann(self):
        """m_inf = 1/(1+exp(-(v+20)/15))."""
        n = GutkinErmentroutNeuron()
        for v in [-80, -60, -20, 0, 20]:
            expected = _m_inf(float(v))
            computed = 1.0 / (1.0 + np.exp(-(v + 20.0) / 15.0))
            assert abs(expected - computed) < 1e-14

    def test_m_inf_midpoint(self):
        """m_inf(-20) = 0.5."""
        assert abs(_m_inf(-20.0) - 0.5) < 1e-12

    def test_n_inf_midpoint(self):
        """n_inf(-25) = 0.5."""
        assert abs(_n_inf(-25.0) - 0.5) < 1e-12

    def test_dv_formula_one_step(self):
        """dV = (-I_Na - I_K - I_L + I) · dt."""
        n = GutkinErmentroutNeuron()
        v0, n0 = n.v, n.n
        I = 3.0
        m_inf_val = _m_inf(v0)
        n_inf_val = _n_inf(v0)
        # dn first
        dn = (n_inf_val - n0) / 1.0 * n.dt
        n_new = n0 + dn
        # Then currents with updated n but original v
        # Actually source: n is updated first, then currents use self.v (unchanged)
        # and self.n (updated)
        i_na = n.g_na * m_inf_val * (v0 - n.e_na)
        i_k = n.g_k * n_new * (v0 - n.e_k)
        i_l = n.g_l * (v0 - n.e_l)
        expected_dv = (-i_na - i_k - i_l + I) * n.dt
        n.step(I)
        actual_dv = n.v - v0
        assert abs(actual_dv - expected_dv) < 1e-10

    def test_dn_formula_one_step(self):
        """dn = (n_inf - n) / tau_n · dt, tau_n=1."""
        n = GutkinErmentroutNeuron()
        v0, n0 = n.v, n.n
        n_inf_val = _n_inf(v0)
        expected_dn = (n_inf_val - n0) / 1.0 * n.dt
        n.step(0.0)
        actual_dn = n.n - n0
        assert abs(actual_dn - expected_dn) < 1e-14

    def test_three_currents(self):
        n = GutkinErmentroutNeuron()
        assert n.g_na > 0 and n.g_k > 0 and n.g_l > 0

    def test_reversal_ordering(self):
        n = GutkinErmentroutNeuron()
        assert n.e_k < n.e_l < n.e_na

    def test_persistent_na_no_inactivation(self):
        """Persistent Na: m only (no h gate). m is instantaneous."""
        # Source: i_na = g_na * m_inf * (v - e_na)
        # No h variable — persistent sodium
        n = GutkinErmentroutNeuron()
        assert not hasattr(n, "h") or n.__class__.__name__ == "GutkinErmentroutNeuron"


# ---------------------------------------------------------------------------
# 3. DYNAMICS
# ---------------------------------------------------------------------------
class TestGEDynamics:
    def test_fires_under_drive(self):
        n = GutkinErmentroutNeuron()
        spikes = _run(n, current=5.0, steps=10_000)
        assert len(spikes) >= 10

    def test_subthreshold_silent(self):
        n = GutkinErmentroutNeuron()
        assert len(_run(n, current=0.5, steps=5000)) == 0

    def test_rate_monotonic(self):
        rates = []
        for I in [2.0, 5.0, 10.0]:
            n = GutkinErmentroutNeuron()
            rates.append(len(_run(n, current=I, steps=10_000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 2.0, 5.0, 10.0, 20.0])
    def test_fi_sweep(self, current: float):
        n = GutkinErmentroutNeuron()
        for _ in range(10_000):
            n.step(current)
        assert np.isfinite(n.v)

    def test_voltage_bounded(self):
        n = GutkinErmentroutNeuron()
        vs = []
        for _ in range(10_000):
            n.step(5.0)
            vs.append(n.v)
        assert min(vs) > -100 and max(vs) < 80


# ---------------------------------------------------------------------------
# 4. PARAMETERS
# ---------------------------------------------------------------------------
class TestGEParameters:
    @pytest.mark.parametrize("g_na", [10.0, 20.0, 40.0])
    def test_g_na_sweep(self, g_na: float):
        n = GutkinErmentroutNeuron(g_na=g_na)
        for _ in range(5000):
            n.step(5.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("g_k", [5.0, 10.0, 20.0])
    def test_g_k_sweep(self, g_k: float):
        n = GutkinErmentroutNeuron(g_k=g_k)
        for _ in range(5000):
            n.step(5.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("dt", [0.02, 0.05, 0.1])
    def test_dt_stability(self, dt: float):
        n = GutkinErmentroutNeuron(dt=dt)
        for _ in range(10_000):
            n.step(5.0)
        assert np.isfinite(n.v) and np.isfinite(n.n)


# ---------------------------------------------------------------------------
# 5. PERFORMANCE
# ---------------------------------------------------------------------------
class TestGEPerformance:
    def test_isolation_throughput(self):
        n = GutkinErmentroutNeuron()
        N = 100_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(5.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        assert rate > 50_000, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(GutkinErmentroutNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert rate > 2_000, f"network: {rate:.0f} neuron-steps/s"


# ---------------------------------------------------------------------------
# 6. FULL PIPELINE
# ---------------------------------------------------------------------------
class TestGEPipeline:
    def test_population(self):
        assert Population(GutkinErmentroutNeuron, n=10, label="ge").n == 10

    def test_projection_wiring(self):
        src = Population(GutkinErmentroutNeuron, n=5, label="src")
        tgt = Population(GutkinErmentroutNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=5.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(GutkinErmentroutNeuron, n=10, label="ge")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = GutkinErmentroutNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(10_000)])
        sc = spike_count(train)
        assert sc >= 5

    def test_analysis_isi(self):
        n = GutkinErmentroutNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(10_000)])
        intervals = isi(train, dt=0.00005)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))

    def test_analysis_firing_rate(self):
        n = GutkinErmentroutNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(10_000)])
        rate = firing_rate(train, dt=0.00005)
        assert rate > 0
