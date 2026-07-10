# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: PerfectIntegratorNeuron

"""Full pipeline test for PerfectIntegratorNeuron (Lapicque 1907, no leak).

dV/dt = I / C — voltage accumulates without decay.
Analytically: ISI = C·θ / (I·dt) steps, firing rate f = I / (C·θ)."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.perfect_integrator import PerfectIntegratorNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_spike_times(neuron: PerfectIntegratorNeuron, current: float, steps: int) -> list[int]:
    """Run neuron and return list of step indices where spikes occurred."""
    return [t for t in range(steps) if neuron.step(current) == 1]


def _analytical_isi_steps(
    current: float, c_m: float, threshold: float, v_reset: float, dt: float
) -> float:
    """Exact ISI in steps: voltage ramp from v_reset to threshold.

    Each step adds dV = I/C * dt.  Steps to threshold = (θ - v_reset) / dV.
    """
    dv_per_step = current / c_m * dt
    if dv_per_step <= 0:
        return float("inf")
    return (threshold - v_reset) / dv_per_step


# ---------------------------------------------------------------------------
# 1. Isolation tests — dynamics, invariants, edge cases
# ---------------------------------------------------------------------------


class TestPerfectIntegratorIsolation:
    """Core single-neuron dynamics."""

    def test_construction_defaults(self):
        n = PerfectIntegratorNeuron()
        assert n.v == 0.0
        assert n.c_m == 1.0
        assert n.v_threshold == 1.0
        assert n.v_reset == 0.0
        assert n.dt == 0.1

    def test_step_returns_binary(self):
        assert PerfectIntegratorNeuron().step(0.0) in (0, 1)

    def test_zero_input_no_drift(self):
        """With I=0, voltage must stay exactly at initial value."""
        n = PerfectIntegratorNeuron(v=0.3)
        for _ in range(1000):
            n.step(0.0)
        assert n.v == 0.3

    def test_linear_voltage_ramp(self):
        """Voltage should increase linearly: V(t) = V₀ + (I/C)·dt·t."""
        n = PerfectIntegratorNeuron()
        I, C, dt = 3.0, n.c_m, n.dt
        dv = I / C * dt  # 0.3 per step
        for t in range(1, 4):
            n.step(I)
            expected = dv * t
            assert abs(n.v - expected) < 1e-12, f"step {t}: {n.v} != {expected}"

    def test_no_leak_invariant(self):
        """Key property: voltage is unchanged by zero-input steps (no decay)."""
        n = PerfectIntegratorNeuron()
        n.step(2.0)
        v_charged = n.v
        for _ in range(500):
            n.step(0.0)
        assert n.v == v_charged, "Voltage decayed — leak detected in integrator"


class TestPerfectIntegratorValidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("v", np.nan),
            ("v_threshold", np.inf),
            ("v_reset", -np.inf),
        ],
    )
    def test_rejects_non_finite_voltage_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            PerfectIntegratorNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["c_m", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_scale_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            PerfectIntegratorNeuron(**{field: value})

    @pytest.mark.parametrize(
        ("v_threshold", "v_reset"),
        [
            (0.0, 0.0),
            (-1.0, 0.0),
        ],
    )
    def test_rejects_non_positive_threshold_excursion(self, v_threshold: float, v_reset: float):
        with pytest.raises(ValueError, match="v_threshold"):
            PerfectIntegratorNeuron(v_threshold=v_threshold, v_reset=v_reset)

    def test_rejects_initial_voltage_at_or_above_threshold(self):
        with pytest.raises(ValueError, match="v must be below v_threshold"):
            PerfectIntegratorNeuron(v=1.0)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = PerfectIntegratorNeuron(v=0.25)
        before = n.v
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert n.v == before

    def test_rejects_non_finite_voltage_increment_before_state_mutation(self):
        n = PerfectIntegratorNeuron(v=0.25, v_threshold=1.0e308, c_m=1.0e-308)
        before = n.v
        with pytest.raises(ValueError, match="voltage increment"):
            n.step(1.0e308)
        assert n.v == before

    @pytest.mark.parametrize("field", ["v", "c_m", "dt", "v_threshold", "v_reset"])
    def test_rejects_corrupted_runtime_state_before_voltage_mutation(self, field: str):
        n = PerfectIntegratorNeuron(v=0.25)
        before = n.v
        setattr(n, field, np.nan)
        with pytest.raises(ValueError, match="runtime"):
            n.step(1.0)
        if field != "v":
            assert n.v == before

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("c_m", 0.0),
            ("dt", 0.0),
            ("v_threshold", 0.0),
        ],
    )
    def test_rejects_invalid_runtime_geometry_before_voltage_mutation(
        self, field: str, value: float
    ):
        n = PerfectIntegratorNeuron(v=0.25)
        before = n.v
        setattr(n, field, value)
        with pytest.raises(ValueError, match="runtime"):
            n.step(1.0)
        assert n.v == before


class TestPerfectIntegratorThreshold:
    """Threshold, reset, and spike timing."""

    def test_exact_threshold_fires(self):
        """When V reaches exactly threshold, a spike must occur."""
        # Use I=10 so dV=1.0/step → hits threshold=1.0 exactly at step 1
        n = PerfectIntegratorNeuron(dt=0.1, c_m=1.0, v_threshold=1.0)
        s = n.step(10.0)
        assert s == 1

    def test_reset_to_v_reset(self):
        n = PerfectIntegratorNeuron()
        n.step(10.0)  # spike
        assert n.v == n.v_reset

    def test_custom_reset_potential(self):
        n = PerfectIntegratorNeuron(v_reset=-0.5, v_threshold=1.0)
        n.step(10.0)  # spike at v=1.0
        assert n.v == -0.5

    def test_superthreshold_instant_spike(self):
        """Very large current → spike on first step."""
        n = PerfectIntegratorNeuron()
        assert n.step(100.0) == 1


class TestPerfectIntegratorAnalyticalFI:
    """f–I curve: firing rate = I·dt / (C·(θ - V_reset)) for constant input.

    This is the exact analytical result for a perfect integrator.
    We verify the simulation matches to within ±1 spike (quantisation).
    """

    @pytest.mark.parametrize("current", [2.0, 5.0, 10.0, 20.0, 50.0])
    def test_fi_curve_analytical(self, current: float):
        n = PerfectIntegratorNeuron()
        steps = 10000
        spikes = sum(n.step(current) for _ in range(steps))
        isi_analytical = _analytical_isi_steps(
            current,
            n.c_m,
            n.v_threshold,
            n.v_reset,
            n.dt,
        )
        # Max 1 spike per step (discrete time clamp)
        expected_spikes = min(steps, steps / isi_analytical)
        # Allow ±1 spike for boundary quantisation
        assert abs(spikes - expected_spikes) <= 1, (
            f"I={current}: got {spikes}, expected {expected_spikes:.1f}"
        )

    def test_fi_linearity(self):
        """f(2I) / f(I) ≈ 2 — perfect integrator has exactly linear f-I."""
        steps = 5000
        n1 = PerfectIntegratorNeuron()
        n2 = PerfectIntegratorNeuron()
        s1 = sum(n1.step(3.0) for _ in range(steps))
        s2 = sum(n2.step(6.0) for _ in range(steps))
        ratio = s2 / s1 if s1 > 0 else float("inf")
        assert 1.95 <= ratio <= 2.05, f"ratio {ratio} deviates from 2.0"

    def test_fi_threshold_dependence(self):
        """Doubling threshold halves the rate (same current)."""
        steps = 5000
        I = 5.0
        n1 = PerfectIntegratorNeuron(v_threshold=1.0)
        n2 = PerfectIntegratorNeuron(v_threshold=2.0)
        s1 = sum(n1.step(I) for _ in range(steps))
        s2 = sum(n2.step(I) for _ in range(steps))
        ratio = s1 / s2 if s2 > 0 else float("inf")
        assert 1.95 <= ratio <= 2.05

    def test_fi_capacitance_dependence(self):
        """Doubling C_m halves the rate."""
        steps = 5000
        I = 5.0
        n1 = PerfectIntegratorNeuron(c_m=1.0)
        n2 = PerfectIntegratorNeuron(c_m=2.0)
        s1 = sum(n1.step(I) for _ in range(steps))
        s2 = sum(n2.step(I) for _ in range(steps))
        ratio = s1 / s2 if s2 > 0 else float("inf")
        assert 1.95 <= ratio <= 2.05


class TestPerfectIntegratorISI:
    """Inter-spike interval analysis — should be perfectly regular."""

    def test_constant_isi(self):
        """All ISIs identical (deterministic, no adaptation)."""
        n = PerfectIntegratorNeuron()
        times = _collect_spike_times(n, current=5.0, steps=2000)
        assert len(times) >= 10, "Not enough spikes to analyse ISI"
        isis = np.diff(times)
        # All ISIs should be identical (±0 for deterministic model)
        assert np.all(isis == isis[0]), f"ISI variability detected: unique ISIs = {np.unique(isis)}"

    def test_isi_matches_analytical(self):
        """Measured ISI matches C·(θ-V_reset) / (I·dt)."""
        n = PerfectIntegratorNeuron(c_m=2.0, v_threshold=3.0, v_reset=0.5)
        I = 10.0
        times = _collect_spike_times(n, current=I, steps=5000)
        assert len(times) >= 5
        measured_isi = np.median(np.diff(times))
        expected_isi = _analytical_isi_steps(
            I,
            n.c_m,
            n.v_threshold,
            n.v_reset,
            n.dt,
        )
        # Allow 1 step tolerance for floating-point rounding
        assert abs(measured_isi - round(expected_isi)) <= 1

    def test_cv_isi_zero(self):
        """Coefficient of variation of ISI = 0 (no jitter)."""
        n = PerfectIntegratorNeuron()
        times = _collect_spike_times(n, current=5.0, steps=5000)
        isis = np.diff(times).astype(float)
        cv = np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else 0.0
        assert cv == 0.0, f"CV(ISI) = {cv}, expected 0.0"


class TestPerfectIntegratorEdgeCases:
    """Numerical edge cases and parameter boundaries."""

    def test_negative_current_no_spike(self):
        """Negative input drives V below reset — never reaches threshold."""
        n = PerfectIntegratorNeuron()
        spikes = sum(n.step(-5.0) for _ in range(1000))
        assert spikes == 0
        assert n.v < 0.0

    def test_large_negative_current_finite(self):
        """Even extreme negative current keeps V finite."""
        n = PerfectIntegratorNeuron()
        for _ in range(100000):
            n.step(-1e6)
        assert np.isfinite(n.v)

    def test_very_small_dt(self):
        """Fine time resolution: more steps to spike, same total time."""
        n = PerfectIntegratorNeuron(dt=0.001)
        I = 5.0
        # Analytical: steps = θ / (I/C * dt) = 1.0 / 0.005 = 200
        times = _collect_spike_times(n, current=I, steps=500)
        assert len(times) >= 1
        assert abs(times[0] - 200) <= 1

    def test_very_large_dt(self):
        """Coarse dt: spike on first step if dV >= θ."""
        n = PerfectIntegratorNeuron(dt=1.0)
        # dV = 5.0 * 1.0 = 5.0 >= 1.0
        assert n.step(5.0) == 1

    def test_threshold_must_exceed_reset(self):
        """Zero threshold excursion is a degenerate no-distance ISI."""
        with pytest.raises(ValueError, match="v_threshold"):
            PerfectIntegratorNeuron(v_threshold=0.0, v_reset=0.0)

    def test_floating_point_accumulation(self):
        """Document fp rounding: 10 additions of 0.1 ≠ 1.0 exactly.

        With I=1.0, dt=0.1, C=1.0: dV=0.1/step. After 10 steps,
        V = 0.99999... due to IEEE 754 rounding of 0.1.
        Spike is delayed to step 11.
        """
        n = PerfectIntegratorNeuron()
        times = _collect_spike_times(n, current=1.0, steps=15)
        assert len(times) >= 1
        # Spike at step 10 or 11 depending on fp accumulation
        assert times[0] in (9, 10), f"First spike at step {times[0]}"

    def test_alternating_current(self):
        """Alternating +/- current: voltage oscillates around 0, no spikes."""
        n = PerfectIntegratorNeuron()
        spikes = 0
        for t in range(10000):
            sign = 1.0 if t % 2 == 0 else -1.0
            spikes += n.step(sign * 3.0)
        assert spikes == 0
        assert abs(n.v) < 1e-10

    def test_reset_method(self):
        n = PerfectIntegratorNeuron()
        for _ in range(50):
            n.step(3.0)
        n.reset()
        assert n.v == n.v_reset

    def test_deterministic_reproducibility(self):
        """Exact bit-for-bit reproducibility across runs."""
        runs = []
        for _ in range(3):
            n = PerfectIntegratorNeuron()
            trace = [(n.step(3.5), n.v) for _ in range(200)]
            runs.append(trace)
        assert runs[0] == runs[1] == runs[2]


class TestPerfectIntegratorParameterSweep:
    """Systematic parameter sweeps verifying scaling laws."""

    @pytest.mark.parametrize("c_m", [0.5, 1.0, 2.0, 5.0])
    def test_rate_inversely_proportional_to_capacitance(self, c_m: float):
        """f ∝ 1/C — verify across parameter range."""
        n = PerfectIntegratorNeuron(c_m=c_m)
        I = 10.0
        steps = 5000
        spikes = sum(n.step(I) for _ in range(steps))
        isi = _analytical_isi_steps(I, c_m, n.v_threshold, n.v_reset, n.dt)
        expected = min(steps, steps / isi)
        assert abs(spikes - expected) <= 1

    @pytest.mark.parametrize("threshold", [0.5, 1.0, 2.0, 5.0])
    def test_rate_inversely_proportional_to_threshold(self, threshold: float):
        """f ∝ 1/θ."""
        n = PerfectIntegratorNeuron(v_threshold=threshold)
        I = 10.0
        steps = 5000
        spikes = sum(n.step(I) for _ in range(steps))
        isi = _analytical_isi_steps(I, n.c_m, threshold, n.v_reset, n.dt)
        expected = min(steps, steps / isi)
        assert abs(spikes - expected) <= 1


# ---------------------------------------------------------------------------
# 2. Network integration
# ---------------------------------------------------------------------------


class TestPerfectIntegratorNetwork:
    def test_population_construction(self):
        pop = Population(PerfectIntegratorNeuron, n=10, label="pi")
        assert pop.n == 10

    def test_network_produces_spikes(self):
        pop = Population(PerfectIntegratorNeuron, n=20, label="pi")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_two_populations_different_drive(self):
        """Stronger drive → more spikes across a population."""
        pop_weak = Population(PerfectIntegratorNeuron, n=10, label="weak")
        pop_high_drive = Population(PerfectIntegratorNeuron, n=10, label="high_drive")
        drive_weak = PoissonInput(n=10, rate_hz=100.0, weight=2.0, dt=0.001, seed=42)
        drive_high_drive = PoissonInput(n=10, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon_weak = SpikeMonitor(pop_weak)
        mon_high_drive = SpikeMonitor(pop_high_drive)
        net_weak = Network(pop_weak, drive_weak, mon_weak)
        net_high_drive = Network(pop_high_drive, drive_high_drive, mon_high_drive)
        net_weak.run(duration=0.5, dt=0.001, backend="python")
        net_high_drive.run(duration=0.5, dt=0.001, backend="python")
        assert mon_high_drive.count > mon_weak.count


# ---------------------------------------------------------------------------
# 3. Analysis pipeline
# ---------------------------------------------------------------------------


class TestPerfectIntegratorAnalysis:
    def test_spike_count_matches_manual(self):
        n = PerfectIntegratorNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(500)])
        manual_count = int(train.sum())
        assert spike_count(train) == manual_count

    def test_spike_count_long_run(self):
        """Long run spike count matches analytical prediction."""
        n = PerfectIntegratorNeuron()
        I = 5.0
        steps = 10000
        train = np.array([float(n.step(I)) for _ in range(steps)])
        analytical = steps * I * n.dt / (n.c_m * n.v_threshold)
        assert abs(spike_count(train) - analytical) <= 1


class TestPerfectIntegratorSimulate:
    """Engineering-verification surface for ``PerfectIntegratorNeuron.simulate``."""

    def test_simulate_python_returns_finite_trace(self) -> None:
        n = PerfectIntegratorNeuron()
        trace, spikes = n.simulate(1000, current=1.0, backend="python")
        assert trace.shape == (1000,)
        assert np.all(np.isfinite(trace))
        assert spikes == 90
        assert n.v == float(trace[-1])

    def test_simulate_rust_matches_python(self) -> None:
        pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")
        py = PerfectIntegratorNeuron()
        rs = PerfectIntegratorNeuron()
        tr_py, sp_py = py.simulate(1000, current=1.0, backend="python")
        tr_rs, sp_rs = rs.simulate(1000, current=1.0, backend="rust")
        assert sp_py == sp_rs
        assert np.array_equal(tr_py, tr_rs)

    def test_simulate_rust_rejects_non_default(self) -> None:
        pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")
        n = PerfectIntegratorNeuron(c_m=2.0)
        with pytest.raises(RuntimeError, match="factory-default"):
            n.simulate(10, current=0.0, backend="rust")
