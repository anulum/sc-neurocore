# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: LapicqueNeuron

"""Full pipeline test for LapicqueNeuron (Lapicque 1907).

Classical RC integrate-and-fire — the original IF model:
τ · dV/dt = -(V - V_rest) + R·I
Spike: V → V_reset when V ≥ V_threshold.

Steady state: V_ss = V_rest + R·I. Fires only if V_ss ≥ V_threshold,
i.e. I ≥ (V_threshold - V_rest) / R = rheobase.
Exact constant-current flow:
V(t + dt) = V_ss + (V(t) - V_ss) · exp(-dt / τ).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.lapicque import LapicqueNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: LapicqueNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestLapicqueIsolation:
    def test_defaults(self):
        n = LapicqueNeuron()
        assert n.v == 0.0 and n.v_rest == 0.0
        assert n.v_threshold == 1.0 and n.v_reset == 0.0
        assert n.tau == 20.0 and n.resistance == 1.0 and n.dt == 1.0

    def test_step_returns_binary(self):
        assert LapicqueNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self):
        n = LapicqueNeuron()
        for _ in range(100_000):
            n.step(20.0)
        assert np.isfinite(n.v)

    def test_reset_restores_default(self):
        n = LapicqueNeuron()
        for _ in range(100):
            n.step(20.0)
        n.reset()
        assert n.v == n.v_rest

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = LapicqueNeuron()
            trace = [(n.step(20.0), n.v) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestLapicqueValidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("v", np.nan),
            ("v_rest", np.inf),
            ("v_reset", -np.inf),
            ("v_threshold", np.nan),
        ],
    )
    def test_rejects_non_finite_voltage_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            LapicqueNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["tau", "resistance", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_rc_scale_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            LapicqueNeuron(**{field: value})

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"v_threshold": 0.0, "v_rest": 0.0},
            {"v_threshold": -1.0, "v_rest": 0.0},
            {"v_threshold": 0.0, "v_reset": 0.0},
            {"v_threshold": -1.0, "v_reset": 0.0},
        ],
    )
    def test_rejects_invalid_threshold_geometry(self, kwargs):
        with pytest.raises(ValueError, match="v_threshold"):
            LapicqueNeuron(**kwargs)

    def test_rejects_initial_voltage_at_or_above_threshold(self):
        with pytest.raises(ValueError, match="v must be below v_threshold"):
            LapicqueNeuron(v=1.0)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("tau", 0.0, "tau"),
            ("resistance", -1.0, "resistance"),
            ("dt", np.nan, "dt"),
            ("v_threshold", 0.0, "v_threshold"),
            ("v", 1.0, "v must be below v_threshold"),
        ],
    )
    def test_rejects_corrupted_runtime_state_before_integration(
        self, field: str, value: float, message: str
    ):
        n = LapicqueNeuron(v=0.25)
        setattr(n, field, value)
        before = n.v
        with pytest.raises(ValueError, match=message):
            n.step(1.0)
        assert n.v == before

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = LapicqueNeuron(v=0.25)
        before = n.v
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert n.v == before

    def test_rejects_non_finite_voltage_candidate_before_state_mutation(self):
        n = LapicqueNeuron(v=0.25, v_threshold=1.0e308, resistance=1.0e308)
        before = n.v
        with pytest.raises(ValueError, match="voltage candidate"):
            n.step(1.0e308)
        assert n.v == before


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — exact-flow formula, steady state, rheobase
# ---------------------------------------------------------------------------
class TestLapicqueAnalytical:
    def test_exact_flow_formula(self):
        """V_next = V_ss + (V - V_ss) · exp(-dt / τ)."""
        n = LapicqueNeuron()
        v0 = n.v
        I = 0.5  # subthreshold
        v_ss = n.v_rest + n.resistance * I
        expected = v_ss + (v0 - v_ss) * np.exp(-n.dt / n.tau)
        n.step(I)
        assert abs(n.v - expected) < 1e-14

    def test_exact_flow_separates_from_forward_euler_for_large_dt(self):
        n = LapicqueNeuron(v=0.25, dt=5.0)
        v0 = n.v
        current = 0.5
        euler = v0 + (-(v0 - n.v_rest) + n.resistance * current) / n.tau * n.dt
        v_ss = n.v_rest + n.resistance * current
        expected = v_ss + (v0 - v_ss) * np.exp(-n.dt / n.tau)
        spike = n.step(current)
        assert spike == 0
        assert abs(n.v - expected) < 1e-14
        assert abs(n.v - euler) > 1e-4

    def test_steady_state(self):
        """V_ss = V_rest + R·I (at equilibrium dV=0)."""
        n = LapicqueNeuron()
        I = 0.5  # subthreshold
        for _ in range(10_000):
            n.step(I)
        expected_ss = n.v_rest + n.resistance * I
        assert abs(n.v - expected_ss) < 0.01

    def test_rheobase(self):
        """Rheobase = (V_threshold - V_rest) / R. Below: silent."""
        n = LapicqueNeuron()
        rheobase = (n.v_threshold - n.v_rest) / n.resistance
        # Below rheobase: no spikes
        assert len(_run(n, current=rheobase * 0.9, steps=5000)) == 0

    def test_above_rheobase_fires(self):
        n = LapicqueNeuron()
        rheobase = (n.v_threshold - n.v_rest) / n.resistance
        assert len(_run(n, current=rheobase * 1.5, steps=5000)) >= 10

    def test_spike_resets_voltage(self):
        n = LapicqueNeuron()
        for _ in range(10_000):
            if n.step(20.0) == 1:
                assert n.v == n.v_reset
                break

    def test_resistance_scales_input(self):
        """Higher R → more effective current."""
        n1 = LapicqueNeuron(resistance=0.5, v_threshold=100.0)
        n2 = LapicqueNeuron(resistance=2.0, v_threshold=100.0)
        for _ in range(100):
            n1.step(10.0)
            n2.step(10.0)
        assert n2.v > n1.v


# ---------------------------------------------------------------------------
# 3. DYNAMICS
# ---------------------------------------------------------------------------
class TestLapicqueDynamics:
    def test_fires_under_drive(self):
        n = LapicqueNeuron()
        assert len(_run(n, current=20.0, steps=5000)) >= 100

    def test_subthreshold_silent(self):
        n = LapicqueNeuron()
        assert len(_run(n, current=0.5, steps=5000)) == 0

    def test_rate_monotonic(self):
        rates = []
        for I in [10.0, 20.0, 50.0]:
            n = LapicqueNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 5.0, 10.0, 20.0, 50.0])
    def test_fi_sweep(self, current: float):
        n = LapicqueNeuron()
        for _ in range(5000):
            n.step(current)
        assert np.isfinite(n.v)


# ---------------------------------------------------------------------------
# 4. PARAMETERS
# ---------------------------------------------------------------------------
class TestLapicqueParameters:
    @pytest.mark.parametrize("tau", [5.0, 20.0, 50.0])
    def test_tau_sweep(self, tau: float):
        n = LapicqueNeuron(tau=tau)
        for _ in range(5000):
            n.step(20.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("resistance", [0.5, 1.0, 2.0])
    def test_resistance_sweep(self, resistance: float):
        n = LapicqueNeuron(resistance=resistance)
        spikes = len(_run(n, current=20.0, steps=5000))
        assert isinstance(spikes, int)

    @pytest.mark.parametrize("dt", [0.1, 1.0, 2.0])
    def test_dt_stability(self, dt: float):
        n = LapicqueNeuron(dt=dt)
        for _ in range(5000):
            n.step(20.0)
        assert np.isfinite(n.v)


# ---------------------------------------------------------------------------
# 5. PERFORMANCE
# ---------------------------------------------------------------------------
class TestLapicquePerformance:
    def test_isolation_throughput(self):
        n = LapicqueNeuron()
        N = 500_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(20.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        min_rate = 100_000 if os.environ.get("CI") else 160_000
        assert np.isfinite(n.v)
        assert rate > min_rate, f"isolation: {rate:.0f} steps/s, minimum={min_rate}"

    def test_network_throughput(self):
        pop = Population(LapicqueNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert rate > 5_000, f"network: {rate:.0f} neuron-steps/s"


# ---------------------------------------------------------------------------
# 6. FULL PIPELINE
# ---------------------------------------------------------------------------
class TestLapicquePipeline:
    def test_population(self):
        assert Population(LapicqueNeuron, n=10, label="lap").n == 10

    def test_projection_wiring(self):
        src = Population(LapicqueNeuron, n=5, label="src")
        tgt = Population(LapicqueNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=5.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(LapicqueNeuron, n=10, label="lap")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = LapicqueNeuron()
        train = np.array([float(n.step(20.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 50

    def test_analysis_isi(self):
        n = LapicqueNeuron()
        train = np.array([float(n.step(20.0)) for _ in range(5000)])
        intervals = isi(train, dt=0.001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))

    def test_analysis_firing_rate(self):
        n = LapicqueNeuron()
        train = np.array([float(n.step(20.0)) for _ in range(5000)])
        rate = firing_rate(train, dt=0.001)
        assert rate > 0

    def test_analysis_cross_validation(self):
        n = LapicqueNeuron()
        train = np.array([float(n.step(20.0)) for _ in range(5000)])
        sc = spike_count(train)
        dt_sim = 0.001
        duration = len(train) * dt_sim
        rate = firing_rate(train, dt=dt_sim)
        if sc > 0:
            expected = sc / duration
            assert abs(rate - expected) < expected * 0.1


class TestLapicqueNeuronSimulate:
    """Engineering-verification surface for ``LapicqueNeuron.simulate``."""

    def test_simulate_python_returns_finite_trace(self) -> None:
        n = LapicqueNeuron()
        trace, spikes = n.simulate(1000, current=2.0, backend="python")
        assert trace.shape == (1000,)
        assert np.all(np.isfinite(trace))
        assert spikes >= 1

    def test_simulate_rust_matches_or_ulp_python(self) -> None:
        pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")
        py = LapicqueNeuron()
        rs = LapicqueNeuron()
        tr_py, sp_py = py.simulate(1000, current=2.0, backend="python")
        tr_rs, sp_rs = rs.simulate(1000, current=2.0, backend="rust")
        assert sp_py == sp_rs
        max_diff = float(np.max(np.abs(tr_py - tr_rs)))
        assert max_diff < 1e-9

    def test_simulate_rust_rejects_non_default(self) -> None:
        pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")
        # force non-default via a constructor override that every model accepts
        try:
            n = (
                LapicqueNeuron(dt=0.02)
                if "dt" in LapicqueNeuron.__dataclass_fields__
                else LapicqueNeuron()
            )
            if "dt" not in LapicqueNeuron.__dataclass_fields__:
                pytest.skip("no dt field")
        except TypeError:
            pytest.skip("cannot override defaults")
        with pytest.raises(RuntimeError, match="factory-default"):
            n.simulate(10, current=0.0, backend="rust")
