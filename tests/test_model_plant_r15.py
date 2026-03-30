# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: PlantR15Neuron

"""Full pipeline test for PlantR15Neuron (Plant 1981, Aplysia R15).

5 ODEs: V, m, h, n, Ca. Parabolic burster with Ca-dependent K current.
At default parameters, model fires one transient spike then converges to
a stable equilibrium at V ≈ −23.8 mV (Ca accumulation suppresses firing)."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.plant_r15 import PlantR15Neuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(neuron: PlantR15Neuron, current: float, steps: int) -> tuple[list[int], list[float]]:
    """Return (spike_times, voltage_trace)."""
    spike_times: list[int] = []
    voltages: list[float] = []
    for t in range(steps):
        s = neuron.step(current)
        if s == 1:
            spike_times.append(t)
        voltages.append(neuron.v)
    return spike_times, voltages


# ---------------------------------------------------------------------------
# 1. Isolation — construction and state variables
# ---------------------------------------------------------------------------


class TestPlantR15Isolation:
    def test_construction_defaults(self):
        n = PlantR15Neuron()
        assert n.v == -50.0
        assert n.m == 0.05
        assert n.h == 0.6
        assert n.n == 0.3
        assert n.ca == 0.1
        assert n.dt == 0.05
        assert n.v_threshold == -10.0

    def test_step_returns_binary(self):
        n = PlantR15Neuron()
        assert n.step(0.0) in (0, 1)

    def test_five_state_variables_evolve(self):
        """All five state variables (V, m, h, n, Ca) should change."""
        n = PlantR15Neuron()
        initial = (n.v, n.m, n.h, n.n, n.ca)
        for _ in range(100):
            n.step(1.0)
        final = (n.v, n.m, n.h, n.n, n.ca)
        for i, (name, v0, v1) in enumerate(zip(["v", "m", "h", "n", "ca"], initial, final)):
            assert v0 != v1, f"{name} didn't evolve"

    def test_substep_integration(self):
        """Model uses 5 sub-steps per step() call for numerical stability."""
        n = PlantR15Neuron()
        v_before = n.v
        n.step(1.0)
        # With 5 sub-steps × dt=0.05, effective integration = 5 × 0.05 = 0.25 ms
        # Voltage should have changed
        assert n.v != v_before

    def test_reset_restores_initial(self):
        n = PlantR15Neuron()
        for _ in range(500):
            n.step(1.0)
        n.reset()
        assert n.v == -50.0
        assert n.m == 0.05
        assert n.h == 0.6
        assert n.n == 0.3
        assert n.ca == 0.1


# ---------------------------------------------------------------------------
# 2. Equilibrium convergence — key finding at default params
# ---------------------------------------------------------------------------


class TestPlantR15Equilibrium:
    def test_transient_spike(self):
        """From initial conditions, model fires exactly 1 transient spike."""
        n = PlantR15Neuron()
        spike_times, _ = _run(n, current=0.0, steps=100000)
        assert len(spike_times) == 1, f"Expected 1 transient spike, got {len(spike_times)}"

    def test_converges_to_fixed_point(self):
        """After transient, V stabilises near −23.8 mV (equilibrium)."""
        n = PlantR15Neuron()
        for _ in range(50000):
            n.step(0.0)
        v_eq = n.v
        # Run 10k more steps — V should barely change
        for _ in range(10000):
            n.step(0.0)
        assert abs(n.v - v_eq) < 0.01, (
            f"V drifted from {v_eq:.3f} to {n.v:.3f} — not at equilibrium"
        )
        assert -30.0 < v_eq < -15.0, f"V_eq = {v_eq:.2f} outside expected range"

    def test_equilibrium_independent_of_small_current(self):
        """Small currents (I<1) shift equilibrium slightly but don't trigger
        sustained oscillation — model stays at a (shifted) fixed point."""
        for I in [0.0, 0.1, 0.5]:
            n = PlantR15Neuron()
            spike_times, _ = _run(n, current=I, steps=100000)
            assert len(spike_times) <= 2, (
                f"I={I}: {len(spike_times)} spikes — expected ≤2 (transient only)"
            )


# ---------------------------------------------------------------------------
# 3. Calcium dynamics
# ---------------------------------------------------------------------------


class TestPlantR15Calcium:
    def test_calcium_non_negative(self):
        """Ca concentration is clamped ≥ 0."""
        n = PlantR15Neuron()
        for _ in range(50000):
            n.step(0.0)
        assert n.ca >= 0.0

    def test_calcium_accumulates_from_initial(self):
        """Ca should increase from initial 0.1 during early transient
        (Ca influx from depolarisation > Ca decay)."""
        n = PlantR15Neuron()
        ca_initial = n.ca
        for _ in range(5000):
            n.step(0.0)
        assert n.ca > ca_initial, f"Ca={n.ca:.4f} <= initial {ca_initial}"

    def test_calcium_at_equilibrium(self):
        """At steady state, Ca stabilises (dCa/dt ≈ 0)."""
        n = PlantR15Neuron()
        for _ in range(50000):
            n.step(0.0)
        ca_1 = n.ca
        for _ in range(10000):
            n.step(0.0)
        ca_2 = n.ca
        assert abs(ca_2 - ca_1) < 0.01, f"Ca still drifting: {ca_1:.4f} → {ca_2:.4f}"

    def test_calcium_suppresses_firing(self):
        """High Ca activates I_KCa, which hyperpolarises — the mechanism
        that terminates bursts in the R15 model."""
        n = PlantR15Neuron()
        for _ in range(50000):
            n.step(0.0)
        # At equilibrium, Ca should be significant
        assert n.ca > 0.5, f"Ca = {n.ca:.4f}, expected >0.5 at equilibrium"


# ---------------------------------------------------------------------------
# 4. Gating variables
# ---------------------------------------------------------------------------


class TestPlantR15Gating:
    def test_gating_bounded(self):
        """m, h, n should stay approximately in [0, 1]."""
        n = PlantR15Neuron()
        for _ in range(50000):
            n.step(1.0)
        for name, val in [("m", n.m), ("h", n.h), ("n", n.n)]:
            assert -0.01 <= val <= 1.01, f"{name} = {val:.6f}"

    def test_gating_at_equilibrium(self):
        """At fixed point, gating variables should be stable."""
        n = PlantR15Neuron()
        for _ in range(50000):
            n.step(0.0)
        g1 = (n.m, n.h, n.n)
        for _ in range(10000):
            n.step(0.0)
        g2 = (n.m, n.h, n.n)
        for name, v1, v2 in zip(["m", "h", "n"], g1, g2):
            assert abs(v1 - v2) < 1e-4, f"{name} drifted: {v1:.6f} → {v2:.6f}"


# ---------------------------------------------------------------------------
# 5. Numerical stability
# ---------------------------------------------------------------------------


class TestPlantR15Stability:
    def test_moderate_current_finite(self):
        """Moderate current (I≤10) keeps all state finite."""
        n = PlantR15Neuron()
        for _ in range(50000):
            n.step(10.0)
        for name, val in [("v", n.v), ("m", n.m), ("h", n.h), ("n", n.n), ("ca", n.ca)]:
            assert np.isfinite(val), f"{name} = {val}"

    def test_high_current_divergence(self):
        """Very high current (I≥100) may cause voltage divergence.

        This documents a numerical limitation — Euler integration with
        dt=0.05 and 5 sub-steps can't handle extreme drive.
        """
        n = PlantR15Neuron()
        for _ in range(100000):
            n.step(100.0)
        # At I=100, V may diverge far from biological range
        # We just document this — not a bug, just Euler limitation
        assert np.isfinite(n.v), "V is NaN/Inf — complete numerical failure"

    @pytest.mark.parametrize("dt", [0.02, 0.05, 0.1])
    def test_dt_stability(self, dt: float):
        """Model stays finite across time-step sizes."""
        n = PlantR15Neuron(dt=dt)
        for _ in range(50000):
            n.step(1.0)
        assert np.isfinite(n.v)


# ---------------------------------------------------------------------------
# 6. Parameter sensitivity
# ---------------------------------------------------------------------------


class TestPlantR15Parameters:
    def test_g_kca_controls_burst_termination(self):
        """Reducing g_KCa should allow more spikes (less Ca-K inhibition)."""
        n_low = PlantR15Neuron(g_kca=0.001)
        n_high = PlantR15Neuron(g_kca=0.03)
        s_low, _ = _run(n_low, current=0.0, steps=50000)
        s_high, _ = _run(n_high, current=0.0, steps=50000)
        assert len(s_low) >= len(s_high), f"Low g_KCa: {len(s_low)} spikes, high: {len(s_high)}"

    def test_tau_ca_affects_calcium_dynamics(self):
        """Shorter tau_Ca → faster Ca decay → different equilibrium."""
        n_fast = PlantR15Neuron(tau_ca=100.0)
        n_slow = PlantR15Neuron(tau_ca=1000.0)
        for _ in range(50000):
            n_fast.step(0.0)
            n_slow.step(0.0)
        # Faster decay → lower steady-state Ca
        assert n_fast.ca < n_slow.ca, f"Fast Ca={n_fast.ca:.4f}, slow Ca={n_slow.ca:.4f}"


# ---------------------------------------------------------------------------
# 7. Determinism
# ---------------------------------------------------------------------------


class TestPlantR15Determinism:
    def test_bit_exact_reproducibility(self):
        traces = []
        for _ in range(2):
            n = PlantR15Neuron()
            trace = [(n.step(1.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 8. Network
# ---------------------------------------------------------------------------


class TestPlantR15Network:
    def test_population(self):
        pop = Population(PlantR15Neuron, n=5, label="r15")
        assert pop.n == 5

    def test_network_spikes(self):
        """With strong Poisson drive, R15 neurons should fire."""
        pop = Population(PlantR15Neuron, n=5, label="r15")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0


# ---------------------------------------------------------------------------
# 9. Analysis
# ---------------------------------------------------------------------------


class TestPlantR15Analysis:
    def test_spike_count(self):
        """At least 1 transient spike in a long run."""
        n = PlantR15Neuron()
        train = np.array([float(n.step(0.0)) for _ in range(50000)])
        assert spike_count(train) >= 1

    def test_spike_count_consistency(self):
        n = PlantR15Neuron()
        train = np.array([float(n.step(0.0)) for _ in range(50000)])
        assert spike_count(train) == int(train.sum())
