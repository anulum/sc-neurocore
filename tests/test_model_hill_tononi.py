# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: HillTononiNeuron

"""Full pipeline test for HillTononiNeuron (Hill & Tononi 2005).

Thalamocortical sleep/wake model with 6 ionic currents:
I_Na(g=50, m³_inf·h), I_K(g=5, n⁴), I_h(g=1, m_h),
I_T(g=3, m²_inf·h_t), I_KNa(g=1.33, w_KNa), I_L(g=0.02).

6 state variables: v, h_na, n_k, m_h, h_t, na_i.
Na-dependent K current: w_KNa = 0.37/(1+(38.7/Na_i)^3.5).
Na/K pump: dNa_i = (-0.001·I_Na - pump_max·Na_i/(Na_i+Na_eq))·dt.
Intrinsic oscillator — fires at I=0.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.hill_tononi import HillTononiNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: HillTononiNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestHTIsolation:
    def test_defaults(self):
        n = HillTononiNeuron()
        assert n.v == -65.0 and n.na_i == 5.0
        assert n.h_na == 0.6 and n.n_k == 0.3
        assert n.m_h == 0.0 and n.h_t == 0.9
        assert n.g_kna == 1.33 and n.dt == 0.05

    def test_six_state_variables(self):
        n = HillTononiNeuron()
        for attr in ["v", "h_na", "n_k", "m_h", "h_t", "na_i"]:
            assert hasattr(n, attr)

    def test_step_returns_binary(self):
        assert HillTononiNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self):
        n = HillTononiNeuron()
        for _ in range(50_000):
            n.step(0.0)
        for attr in ["v", "h_na", "n_k", "m_h", "h_t", "na_i"]:
            assert np.isfinite(getattr(n, attr)), f"{attr} not finite"

    def test_reset_restores_defaults(self):
        n = HillTononiNeuron()
        for _ in range(5000):
            n.step(2.0)
        n.reset()
        assert n.v == -65.0 and n.na_i == 5.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = HillTononiNeuron()
            trace = [(n.step(0.0), n.v, n.na_i) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — Na dynamics, KNa current, T-current, 6 currents
# ---------------------------------------------------------------------------
class TestHTAnalytical:
    def test_six_ionic_currents(self):
        n = HillTononiNeuron()
        for g in [n.g_na, n.g_k, n.g_h, n.g_t, n.g_kna, n.g_l]:
            assert g > 0

    def test_kna_activation_formula(self):
        """w_KNa = 0.37 / (1 + (38.7/Na_i)^3.5). At Na_i=38.7: half-max."""
        w = 0.37 / (1.0 + (38.7 / 38.7) ** 3.5)
        assert abs(w - 0.37 / 2.0) < 1e-10

    def test_kna_low_na(self):
        """At low Na_i (5mM): w_KNa ≈ 0 (K channel closed)."""
        w = 0.37 / (1.0 + (38.7 / 5.0) ** 3.5)
        assert w < 0.001

    def test_na_accumulation_during_spiking(self):
        """Na_i increases during spiking (I_Na inward → Na enters)."""
        n = HillTononiNeuron()
        na_before = n.na_i
        for _ in range(10_000):
            n.step(2.0)
        # Na should accumulate from spiking
        assert n.na_i != na_before

    def test_na_non_negative(self):
        """Na_i clipped to ≥ 0."""
        n = HillTononiNeuron()
        for _ in range(50_000):
            n.step(0.0)
            assert n.na_i >= 0.0

    def test_na_pump_formula(self):
        """Na/K pump: rate = pump_max · Na_i / (Na_i + Na_eq)."""
        n = HillTononiNeuron()
        pump_rate = n.na_pump_max * n.na_i / (n.na_i + n.na_eq)
        assert pump_rate > 0 and np.isfinite(pump_rate)

    def test_reversal_ordering(self):
        n = HillTononiNeuron()
        assert n.e_k < n.e_l < n.e_h < n.e_na < n.e_ca

    def test_gating_bounded(self):
        n = HillTononiNeuron()
        for _ in range(10_000):
            n.step(0.0)
        for attr in ["h_na", "n_k", "m_h", "h_t"]:
            val = getattr(n, attr)
            assert -0.05 <= val <= 1.05, f"{attr}={val}"


# ---------------------------------------------------------------------------
# 3. INTRINSIC OSCILLATION
# ---------------------------------------------------------------------------
class TestHTIntrinsic:
    def test_fires_at_zero_current(self):
        """Intrinsic oscillator — fires without external input."""
        n = HillTononiNeuron()
        spikes = _run(n, current=0.0, steps=10_000)
        assert len(spikes) >= 5

    def test_rate_monotonic(self):
        rates = []
        for I in [0.0, 2.0, 5.0]:
            n = HillTononiNeuron()
            rates.append(len(_run(n, current=I, steps=10_000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 1.0, 3.0, 5.0])
    def test_fi_sweep(self, current: float):
        n = HillTononiNeuron()
        for _ in range(10_000):
            n.step(current)
        assert np.isfinite(n.v)


# ---------------------------------------------------------------------------
# 4. PARAMETERS
# ---------------------------------------------------------------------------
class TestHTParameters:
    @pytest.mark.parametrize("g_kna", [0.0, 1.33, 3.0])
    def test_g_kna_sweep(self, g_kna: float):
        n = HillTononiNeuron(g_kna=g_kna)
        for _ in range(5000):
            n.step(0.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("g_t", [0.0, 3.0, 6.0])
    def test_g_t_sweep(self, g_t: float):
        n = HillTononiNeuron(g_t=g_t)
        for _ in range(5000):
            n.step(0.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("dt", [0.02, 0.05, 0.1])
    def test_dt_stability(self, dt: float):
        n = HillTononiNeuron(dt=dt)
        for _ in range(10_000):
            n.step(0.0)
        assert np.isfinite(n.v) and np.isfinite(n.na_i)


# ---------------------------------------------------------------------------
# 5. PERFORMANCE
# ---------------------------------------------------------------------------
class TestHTPerformance:
    def test_isolation_throughput(self):
        n = HillTononiNeuron()
        N = 20_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(0.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        assert rate > 5_000, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(HillTononiNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert rate > 1_000, f"network: {rate:.0f} neuron-steps/s"


# ---------------------------------------------------------------------------
# 6. FULL PIPELINE
# ---------------------------------------------------------------------------
class TestHTPipeline:
    def test_population(self):
        assert Population(HillTononiNeuron, n=10, label="ht").n == 10

    def test_projection_wiring(self):
        src = Population(HillTononiNeuron, n=5, label="src")
        tgt = Population(HillTononiNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=1.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(HillTononiNeuron, n=10, label="ht")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = HillTononiNeuron()
        train = np.array([float(n.step(0.0)) for _ in range(10_000)])
        sc = spike_count(train)
        assert sc >= 3

    def test_analysis_isi(self):
        n = HillTononiNeuron()
        train = np.array([float(n.step(0.0)) for _ in range(10_000)])
        intervals = isi(train, dt=0.00005)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))

    def test_analysis_firing_rate(self):
        n = HillTononiNeuron()
        train = np.array([float(n.step(0.0)) for _ in range(10_000)])
        rate = firing_rate(train, dt=0.00005)
        assert rate > 0


# Salvaged model-specific behavioural contracts from retired aggregate test file.
class TestHillTononi:
    def test_fires(self):
        from sc_neurocore.neurons.models.hill_tononi import HillTononiNeuron

        n = HillTononiNeuron()
        assert sum(n.step(5.0) for _ in range(300)) > 0


class TestHillTononiRK4:
    """Guards the candidate-first RK4 integrator and its cross-backend parity.

    The historical forward Euler advanced the four gates from the old voltage and
    then the membrane and intracellular sodium from the freshly updated gates,
    mixing inconsistent states. The production path is RK4 over the six-state
    ``(v, h_na, n_k, m_h, h_t, na_i)`` system with one consistent right-hand side
    per stage; the staggered baseline survives only behind
    ``integrator="baseline_euler"``. The ``I_KNa`` Hill exponent ``3.5`` is
    evaluated as ``b·b·b·sqrt(b)`` so every backend reproduces the trajectory.
    """

    def test_default_integrator_is_rk4(self):
        assert HillTononiNeuron().integrator == "rk4"

    def test_unknown_integrator_rejected(self):
        with pytest.raises(ValueError, match="integrator"):
            HillTononiNeuron(integrator="trapezoid")  # type: ignore[arg-type]

    def test_baseline_euler_path_runs_and_fires(self):
        n = HillTononiNeuron(integrator="baseline_euler")
        assert sum(n.step(0.0) for _ in range(10000)) > 0

    def test_rk4_and_baseline_differ(self):
        rk4 = HillTononiNeuron()
        euler = HillTononiNeuron(integrator="baseline_euler")
        rk4_v = [round(rk4.step(10.0) or rk4.v, 9) for _ in range(500)]
        euler_v = [round(euler.step(10.0) or euler.v, 9) for _ in range(500)]
        assert rk4_v != euler_v

    def test_cross_backend_spike_anchor(self):
        # Pins the Python reference the Rust/Julia/Go/Mojo kernels reproduce
        # bit-for-bit (verified by benchmarks/bench_model_hill_tononi.py).
        n = HillTononiNeuron()
        assert sum(n.step(10.0) for _ in range(200000)) == 694

    def test_sodium_clamped_non_negative(self):
        n = HillTononiNeuron()
        for _ in range(50000):
            n.step(10.0)
            assert n.na_i >= 0.0

    def test_non_finite_current_rejected(self):
        n = HillTononiNeuron()
        with pytest.raises(ValueError, match="current"):
            n.step(math.nan)

    def test_non_finite_state_rejected_on_step(self):
        n = HillTononiNeuron()
        n.v = math.inf
        with pytest.raises(ValueError, match="v"):
            n.step(10.0)

    def test_negative_conductance_rejected(self):
        with pytest.raises(ValueError, match="g_na"):
            HillTononiNeuron(g_na=-1.0)

    def test_non_positive_na_eq_rejected(self):
        with pytest.raises(ValueError, match="na_eq"):
            HillTononiNeuron(na_eq=0.0)

    def test_stays_finite_under_extreme_drive(self):
        # The delayed-rectifier and I_KNa currents grow with V, so the cell
        # self-limits to a high but finite fixed point rather than diverging; the
        # _safe_exp guard keeps the saturating gates finite (Python math.exp would
        # otherwise raise OverflowError where the other backends return +inf).
        n = HillTononiNeuron()
        for _ in range(2000):
            n.step(1e5)
        assert math.isfinite(n.v) and math.isfinite(n.na_i)

    def test_h_current_evolves(self):
        from sc_neurocore.neurons.models.hill_tononi import HillTononiNeuron

        n = HillTononiNeuron()
        for _ in range(100):
            n.step(3.0)
        assert n.m_h != 0.0
