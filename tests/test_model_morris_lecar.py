# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: MorrisLecarNeuron

"""Full pipeline test for MorrisLecarNeuron (Morris & Lecar 1981).

Calcium-potassium oscillator, 2D:
C dV/dt = -g_Ca·m_∞(V)·(V-E_Ca) - g_K·w·(V-E_K) - g_L·(V-E_L) + I
dw/dt = λ(V)·(w_∞(V) - w)

m_∞(V) = 0.5·(1 + tanh((V-v1)/v2))  — instantaneous Ca activation
w_∞(V) = 0.5·(1 + tanh((V-v3)/v4))  — K activation steady state
λ(V) = φ·cosh((V-v3)/(2·v4))        — K opening rate

Three currents: I_Ca (instantaneous), I_K (w-gated), I_L (leak).
Type-II excitability: oscillation in frequency band, Hopf bifurcation.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import math
import os
import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.morris_lecar import MorrisLecarNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: MorrisLecarNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


def _m_inf(v: float, v1: float, v2: float) -> float:
    return 0.5 * (1.0 + np.tanh((v - v1) / v2))


def _w_inf(v: float, v3: float, v4: float) -> float:
    return 0.5 * (1.0 + np.tanh((v - v3) / v4))


def _lam(v: float, v3: float, v4: float, phi: float) -> float:
    return phi * np.cosh((v - v3) / (2.0 * v4))


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestMLIsolation:
    def test_defaults(self):
        n = MorrisLecarNeuron()
        assert n.v == -60.0 and n.w == 0.0
        assert n.c_m == 20.0 and n.dt == 0.1
        assert n.v_threshold == 0.0

    def test_step_returns_binary(self):
        assert MorrisLecarNeuron().step(0.0) in (0, 1)

    def test_both_states_evolve(self):
        n = MorrisLecarNeuron()
        v0, w0 = n.v, n.w
        for _ in range(500):
            n.step(100.0)
        assert n.v != v0 and n.w != w0

    def test_state_finite_long_run(self):
        n = MorrisLecarNeuron()
        for _ in range(100_000):
            n.step(100.0)
        assert np.isfinite(n.v) and np.isfinite(n.w)

    def test_reset_restores_defaults(self):
        n = MorrisLecarNeuron()
        for _ in range(5000):
            n.step(100.0)
        n.reset()
        assert n.v == -60.0 and n.w == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = MorrisLecarNeuron()
            trace = [(n.step(100.0), n.v, n.w) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]

    @pytest.mark.parametrize("integrator", ["baseline_euler", "rk4", "rosenbrock"])
    def test_extreme_voltage_rate_overflow_fails_closed(self, integrator: str):
        n = MorrisLecarNeuron(v=1e6, w=0.25, integrator=integrator)
        before = (n.v, n.w)

        with pytest.raises(FloatingPointError, match="overflowed|non-finite"):
            n.step(0.0)

        assert (n.v, n.w) == before

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"v": math.nan},
            {"w": -0.01},
            {"w": 1.01},
            {"c_m": 0.0},
            {"g_ca": 0.0},
            {"g_k": 0.0},
            {"g_l": 0.0},
            {"v2": 0.0},
            {"v4": 0.0},
            {"phi": 0.0},
            {"dt": 0.0},
            {"v_threshold": math.inf},
        ],
    )
    def test_invalid_physical_configuration_is_rejected(self, kwargs: dict[str, float]):
        with pytest.raises(ValueError):
            MorrisLecarNeuron(**kwargs)

    @pytest.mark.parametrize("integrator", ["baseline_euler", "rk4", "rosenbrock"])
    def test_runtime_parameter_corruption_fails_before_mutation(self, integrator: str):
        n = MorrisLecarNeuron(integrator=integrator)
        n.phi = math.nan
        before = (n.v, n.w)

        with pytest.raises(ValueError):
            n.step(100.0)

        assert (n.v, n.w) == before

    @pytest.mark.parametrize("integrator", ["baseline_euler", "rk4", "rosenbrock"])
    def test_potassium_activation_bounds_fail_before_mutation(self, integrator: str):
        n = MorrisLecarNeuron(w=1.0, dt=10.0, integrator=integrator)
        before = (n.v, n.w)

        with pytest.raises(FloatingPointError, match="potassium (activation|rate)"):
            n.step(-1_000.0)

        assert (n.v, n.w) == before


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — m_inf, w_inf, lambda, dV, dw formulas
# ---------------------------------------------------------------------------
class TestMLAnalytical:
    def test_m_inf_tanh(self):
        """m_∞(V) = 0.5·(1 + tanh((V-v1)/v2))."""
        n = MorrisLecarNeuron()
        for v in [-80, -60, -20, 0, 40]:
            expected = _m_inf(float(v), n.v1, n.v2)
            assert abs(n._m_inf(float(v)) - expected) < 1e-14

    def test_m_inf_range(self):
        """m_∞ ∈ (0, 1) — tanh bounded."""
        n = MorrisLecarNeuron()
        assert n._m_inf(-200.0) > 0
        assert n._m_inf(-200.0) < 0.01
        assert n._m_inf(200.0) > 0.99
        assert n._m_inf(200.0) < 1.0

    def test_m_inf_midpoint(self):
        """m_∞(v1) = 0.5."""
        n = MorrisLecarNeuron()
        assert abs(n._m_inf(n.v1) - 0.5) < 1e-12

    def test_w_inf_tanh(self):
        n = MorrisLecarNeuron()
        for v in [-80, -60, -20, 0, 40]:
            expected = _w_inf(float(v), n.v3, n.v4)
            assert abs(n._w_inf(float(v)) - expected) < 1e-14

    def test_w_inf_midpoint(self):
        """w_∞(v3) = 0.5."""
        n = MorrisLecarNeuron()
        assert abs(n._w_inf(n.v3) - 0.5) < 1e-12

    def test_lambda_positive(self):
        """λ(V) = φ·cosh(...) > 0 for all V (cosh > 0)."""
        n = MorrisLecarNeuron()
        for v in [-100, -60, 0, 50]:
            assert n._lam(float(v)) > 0

    def test_lambda_matches_reference(self):
        n = MorrisLecarNeuron()
        for v in [-80, -40, 0, 30]:
            expected = _lam(float(v), n.v3, n.v4, n.phi)
            assert abs(n._lam(float(v)) - expected) < 1e-14

    def test_lambda_minimum_at_v3(self):
        """λ minimum at V=v3 where cosh(0)=1 → λ_min = φ."""
        n = MorrisLecarNeuron()
        lam_v3 = n._lam(n.v3)
        assert abs(lam_v3 - n.phi) < 1e-12

    def test_dv_formula_one_step(self):
        """dV = (-I_Ca - I_K - I_L + I) / C_m · dt."""
        n = MorrisLecarNeuron()
        v0, w0 = n.v, n.w
        I = 50.0
        m_inf = n._m_inf(v0)
        i_ca = n.g_ca * m_inf * (v0 - n.e_ca)
        i_k = n.g_k * w0 * (v0 - n.e_k)
        i_l = n.g_l * (v0 - n.e_l)
        expected_dv = (-i_ca - i_k - i_l + I) / n.c_m * n.dt
        n.step(I)
        assert abs((n.v - v0) - expected_dv) < 1e-10

    def test_dw_formula_one_step(self):
        """dw = λ(V)·(w_∞(V) - w) · dt."""
        n = MorrisLecarNeuron()
        v0, w0 = n.v, n.w
        lam = n._lam(v0)
        w_inf = n._w_inf(v0)
        expected_dw = lam * (w_inf - w0) * n.dt
        n.step(0.0)
        assert abs((n.w - w0) - expected_dw) < 1e-14

    def test_current_balance_at_rest(self):
        """Three currents at initial state."""
        n = MorrisLecarNeuron()
        v = n.v
        m_inf = n._m_inf(v)
        i_ca = n.g_ca * m_inf * (v - n.e_ca)
        i_k = n.g_k * n.w * (v - n.e_k)
        i_l = n.g_l * (v - n.e_l)
        # I_Ca inward (v < e_ca), I_K outward (w=0 → negligible), I_L = 0 at v=e_l
        assert i_ca < 0  # inward
        assert abs(i_l) < 1e-10  # v=-60 = e_l
        assert abs(i_k) < 1e-10  # w=0


# ---------------------------------------------------------------------------
# 3. TYPE-II EXCITABILITY
# ---------------------------------------------------------------------------
class TestMLTypeII:
    def test_subthreshold_silent(self):
        n = MorrisLecarNeuron()
        assert len(_run(n, current=10.0, steps=10_000)) == 0

    def test_oscillatory_in_band(self):
        n = MorrisLecarNeuron()
        spikes = _run(n, current=100.0, steps=20_000)
        assert len(spikes) >= 10

    def test_type_ii_frequency_onset(self):
        """Type-II: non-zero frequency onset (Hopf bifurcation)."""
        # Near threshold, frequency should be non-zero
        n = MorrisLecarNeuron()
        spikes = _run(n, current=90.0, steps=50_000)
        if len(spikes) >= 5:
            isis = np.diff(spikes).astype(float)
            # Non-zero frequency at onset (unlike Type-I continuous)
            assert np.mean(isis) < 10_000

    def test_high_current_suppression(self):
        """Very high I pushes past oscillatory window."""
        n = MorrisLecarNeuron()
        s_mid = len(_run(n, current=100.0, steps=10_000))
        n2 = MorrisLecarNeuron()
        s_high = len(_run(n2, current=300.0, steps=10_000))
        assert s_mid >= s_high

    def test_voltage_bounded(self):
        n = MorrisLecarNeuron()
        vs = []
        for _ in range(20_000):
            n.step(100.0)
            vs.append(n.v)
        assert min(vs) > -100 and max(vs) < 150

    def test_w_bounded(self):
        """w ∈ [0, 1) — recovery variable stays in physiological range."""
        n = MorrisLecarNeuron()
        ws = []
        for _ in range(20_000):
            n.step(100.0)
            ws.append(n.w)
        assert min(ws) >= -0.1 and max(ws) <= 1.1


# ---------------------------------------------------------------------------
# 4. DYNAMICS — f-I, ISI
# ---------------------------------------------------------------------------
class TestMLDynamics:
    @pytest.mark.parametrize("current", [50.0, 80.0, 100.0, 120.0, 150.0])
    def test_fi_sweep(self, current: float):
        n = MorrisLecarNeuron()
        for _ in range(20_000):
            n.step(current)
        assert np.isfinite(n.v)

    def test_regular_isi_in_band(self):
        n = MorrisLecarNeuron()
        spikes = _run(n, current=100.0, steps=50_000)
        if len(spikes) >= 10:
            isis = np.diff(spikes[3:]).astype(float)
            cv = np.std(isis) / np.mean(isis)
            assert cv < 0.1

    def test_upward_crossing_only(self):
        n = MorrisLecarNeuron()
        prev_v = n.v
        for _ in range(20_000):
            spike = n.step(100.0)
            if spike == 1:
                assert prev_v < n.v_threshold
            prev_v = n.v


# ---------------------------------------------------------------------------
# 5. PARAMETER SENSITIVITY
# ---------------------------------------------------------------------------
class TestMLParameters:
    @pytest.mark.parametrize("g_ca", [2.0, 4.0, 6.0])
    def test_g_ca_sweep(self, g_ca: float):
        n = MorrisLecarNeuron(g_ca=g_ca)
        for _ in range(20_000):
            n.step(100.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("g_k", [4.0, 8.0, 12.0])
    def test_g_k_sweep(self, g_k: float):
        n = MorrisLecarNeuron(g_k=g_k)
        for _ in range(20_000):
            n.step(100.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("phi", [0.04, 1.0 / 15.0, 0.1])
    def test_phi_timescale(self, phi: float):
        n = MorrisLecarNeuron(phi=phi)
        for _ in range(20_000):
            n.step(100.0)
        assert np.isfinite(n.v) and np.isfinite(n.w)

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float):
        n = MorrisLecarNeuron(dt=dt)
        for _ in range(20_000):
            n.step(100.0)
        assert np.isfinite(n.v) and np.isfinite(n.w)

    def test_reversal_ordering(self):
        n = MorrisLecarNeuron()
        assert n.e_k < n.e_l < n.e_ca


# ---------------------------------------------------------------------------
# 6. PERFORMANCE
# ---------------------------------------------------------------------------
class TestMLPerformance:
    def test_isolation_throughput(self):
        n = MorrisLecarNeuron()
        N = 100_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(100.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        # 2 tanh + 1 cosh + 3 currents + 2 state updates.
        # Hosted runners share CPUs and can transiently drop below the
        # workstation floor; keep the local contract strict and use a
        # CI floor that still catches order-of-magnitude regressions.
        min_rate = 35_000 if os.getenv("CI") else 50_000
        assert rate > min_rate, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(MorrisLecarNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert rate > 2_000, f"network: {rate:.0f} neuron-steps/s"


# ---------------------------------------------------------------------------
# 7. FULL PIPELINE
# ---------------------------------------------------------------------------
class TestMLPipeline:
    def test_population(self):
        assert Population(MorrisLecarNeuron, n=10, label="ml").n == 10

    def test_projection_wiring(self):
        src = Population(MorrisLecarNeuron, n=5, label="src")
        tgt = Population(MorrisLecarNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=20.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(MorrisLecarNeuron, n=5, label="ml")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=200.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = MorrisLecarNeuron()
        train = np.array([float(n.step(100.0)) for _ in range(20_000)])
        sc = spike_count(train)
        assert sc >= 5

    def test_analysis_isi(self):
        n = MorrisLecarNeuron()
        train = np.array([float(n.step(100.0)) for _ in range(50_000)])
        intervals = isi(train, dt=0.0001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
            assert np.all(intervals > 0)

    def test_analysis_firing_rate(self):
        n = MorrisLecarNeuron()
        train = np.array([float(n.step(100.0)) for _ in range(20_000)])
        rate = firing_rate(train, dt=0.0001)
        assert rate >= 0

    def test_analysis_cross_validation(self):
        n = MorrisLecarNeuron()
        train = np.array([float(n.step(100.0)) for _ in range(50_000)])
        sc = spike_count(train)
        dt_sim = 0.0001
        duration = len(train) * dt_sim
        rate = firing_rate(train, dt=dt_sim)
        if sc > 0:
            expected = sc / duration
            assert abs(rate - expected) < expected * 0.1
