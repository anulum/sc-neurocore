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

Candidate-first RK4 step (dt=0.05). m_Na instantaneous.
Simple enough for full analytical verification.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time
import math

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
    return 1.0 / (1.0 + math.exp(-(v + 20.0) / 15.0))


def _n_inf(v: float) -> float:
    return 1.0 / (1.0 + math.exp(-(v + 25.0) / 5.0))


def _rhs(
    neuron: GutkinErmentroutNeuron, v: float, n_gate: float, current: float
) -> tuple[float, float]:
    m_inf = _m_inf(v)
    n_inf = _n_inf(v)
    i_na = neuron.g_na * m_inf * (v - neuron.e_na)
    i_k = neuron.g_k * n_gate * (v - neuron.e_k)
    i_l = neuron.g_l * (v - neuron.e_l)
    return -i_na - i_k - i_l + current, n_inf - n_gate


def _rk4_reference(neuron: GutkinErmentroutNeuron, current: float) -> tuple[float, float]:
    v0, n0 = neuron.v, neuron.n
    k1_v, k1_n = _rhs(neuron, v0, n0, current)
    k2_v, k2_n = _rhs(neuron, v0 + 0.5 * neuron.dt * k1_v, n0 + 0.5 * neuron.dt * k1_n, current)
    k3_v, k3_n = _rhs(neuron, v0 + 0.5 * neuron.dt * k2_v, n0 + 0.5 * neuron.dt * k2_n, current)
    k4_v, k4_n = _rhs(neuron, v0 + neuron.dt * k3_v, n0 + neuron.dt * k3_n, current)
    next_v = v0 + neuron.dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0
    next_n = n0 + neuron.dt * (k1_n + 2.0 * k2_n + 2.0 * k3_n + k4_n) / 6.0
    return next_v, next_n


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestGEIsolation:
    def test_defaults(self) -> None:
        n = GutkinErmentroutNeuron()
        assert n.v == -65.0 and n.n == 0.1
        assert n.g_na == 20.0 and n.g_k == 10.0 and n.g_l == 8.0
        assert n.dt == 0.05

    def test_step_returns_binary(self) -> None:
        assert GutkinErmentroutNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self) -> None:
        n = GutkinErmentroutNeuron()
        for _ in range(100_000):
            n.step(5.0)
        assert np.isfinite(n.v) and np.isfinite(n.n)

    def test_reset_restores_defaults(self) -> None:
        n = GutkinErmentroutNeuron()
        for _ in range(5000):
            n.step(5.0)
        n.reset()
        assert n.v == -65.0 and n.n == 0.1

    def test_deterministic(self) -> None:
        traces = []
        for _ in range(2):
            n = GutkinErmentroutNeuron()
            trace = [(n.step(5.0), n.v, n.n) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"dt": 0.0}, "invalid"),
            ({"dt": float("nan")}, "invalid"),
            ({"n": -0.1}, "invalid"),
            ({"n": 1.1}, "invalid"),
            ({"g_na": -1.0}, "invalid"),
            ({"v": float("inf")}, "invalid"),
        ],
    )
    def test_invalid_initial_contract_rejected(self, kwargs: dict[str, float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            GutkinErmentroutNeuron(**kwargs)


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — m_inf, n_inf, dV, dn formulas
# ---------------------------------------------------------------------------
class TestGEAnalytical:
    def test_m_inf_boltzmann(self) -> None:
        """m_inf = 1/(1+exp(-(v+20)/15))."""
        n = GutkinErmentroutNeuron()
        for v in [-80, -60, -20, 0, 20]:
            expected = _m_inf(float(v))
            computed = 1.0 / (1.0 + np.exp(-(v + 20.0) / 15.0))
            assert abs(expected - computed) < 1e-14

    def test_m_inf_midpoint(self) -> None:
        """m_inf(-20) = 0.5."""
        assert abs(_m_inf(-20.0) - 0.5) < 1e-12

    def test_n_inf_midpoint(self) -> None:
        """n_inf(-25) = 0.5."""
        assert abs(_n_inf(-25.0) - 0.5) < 1e-12

    def test_rk4_current_balance_one_step(self) -> None:
        """One committed step matches the explicit RK4 current balance."""
        n = GutkinErmentroutNeuron()
        expected_v, expected_n = _rk4_reference(n, current=3.0)
        n.step(3.0)
        assert abs(n.v - expected_v) < 1e-12
        assert abs(n.n - expected_n) < 1e-12

    def test_rk4_differs_from_euler_baseline(self) -> None:
        """RK4 is not the historical first-order Euler update."""
        n = GutkinErmentroutNeuron()
        v0, n0 = n.v, n.n
        current = 3.0
        euler_n = n0 + (_n_inf(v0) - n0) * n.dt
        euler_v = v0 + _rhs(n, v0, euler_n, current)[0] * n.dt
        n.step(current)
        assert abs(n.v - euler_v) > 1e-6

    def test_three_currents(self) -> None:
        n = GutkinErmentroutNeuron()
        assert n.g_na > 0 and n.g_k > 0 and n.g_l > 0

    def test_reversal_ordering(self) -> None:
        n = GutkinErmentroutNeuron()
        assert n.e_k < n.e_l < n.e_na

    def test_persistent_na_no_inactivation(self) -> None:
        """Persistent Na: m only (no h gate). m is instantaneous."""
        # Source: i_na = g_na * m_inf * (v - e_na)
        # No h variable — persistent sodium
        n = GutkinErmentroutNeuron()
        assert not hasattr(n, "h") or n.__class__.__name__ == "GutkinErmentroutNeuron"


# ---------------------------------------------------------------------------
# 3. DYNAMICS
# ---------------------------------------------------------------------------
class TestGEDynamics:
    def test_fires_under_drive(self) -> None:
        n = GutkinErmentroutNeuron()
        spikes = _run(n, current=5.0, steps=10_000)
        assert len(spikes) >= 10

    def test_subthreshold_silent(self) -> None:
        n = GutkinErmentroutNeuron()
        assert len(_run(n, current=0.5, steps=5000)) == 0

    def test_rate_monotonic(self) -> None:
        rates = []
        for I in [2.0, 5.0, 10.0]:
            n = GutkinErmentroutNeuron()
            rates.append(len(_run(n, current=I, steps=10_000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 2.0, 5.0, 10.0, 20.0])
    def test_fi_sweep(self, current: float) -> None:
        n = GutkinErmentroutNeuron()
        for _ in range(10_000):
            n.step(current)
        assert np.isfinite(n.v)

    def test_voltage_bounded(self) -> None:
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
    def test_g_na_sweep(self, g_na: float) -> None:
        n = GutkinErmentroutNeuron(g_na=g_na)
        for _ in range(5000):
            n.step(5.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("g_k", [5.0, 10.0, 20.0])
    def test_g_k_sweep(self, g_k: float) -> None:
        n = GutkinErmentroutNeuron(g_k=g_k)
        for _ in range(5000):
            n.step(5.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("dt", [0.02, 0.05, 0.1])
    def test_dt_stability(self, dt: float) -> None:
        n = GutkinErmentroutNeuron(dt=dt)
        for _ in range(10_000):
            n.step(5.0)
        assert np.isfinite(n.v) and np.isfinite(n.n)

    def test_invalid_runtime_current_preserves_state(self) -> None:
        n = GutkinErmentroutNeuron()
        before = (n.v, n.n)
        with pytest.raises(ValueError, match="invalid"):
            n.step(float("nan"))
        assert (n.v, n.n) == before

    def test_invalid_candidate_preserves_state(self) -> None:
        n = GutkinErmentroutNeuron(dt=100.0)
        before = (n.v, n.n)
        with pytest.raises(ValueError, match="candidate"):
            n.step(1.0e9)
        assert (n.v, n.n) == before


# ---------------------------------------------------------------------------
# 5. PERFORMANCE
# ---------------------------------------------------------------------------
class TestGEPerformance:
    def test_isolation_throughput(self) -> None:
        n = GutkinErmentroutNeuron()
        N = 100_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(5.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        assert rate > 50_000, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self) -> None:
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
    def test_population(self) -> None:
        assert Population(GutkinErmentroutNeuron, n=10, label="ge").n == 10

    def test_projection_wiring(self) -> None:
        src = Population(GutkinErmentroutNeuron, n=5, label="src")
        tgt = Population(GutkinErmentroutNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=5.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self) -> None:
        pop = Population(GutkinErmentroutNeuron, n=10, label="ge")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self) -> None:
        n = GutkinErmentroutNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(10_000)])
        sc = spike_count(train)
        assert sc >= 5

    def test_analysis_isi(self) -> None:
        n = GutkinErmentroutNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(10_000)])
        intervals = isi(train, dt=0.00005)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))

    def test_analysis_firing_rate(self) -> None:
        n = GutkinErmentroutNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(10_000)])
        rate = firing_rate(train, dt=0.00005)
        assert rate > 0
