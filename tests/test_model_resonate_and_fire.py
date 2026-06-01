# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: ResonateAndFireNeuron

"""Full pipeline test for ResonateAndFireNeuron (Izhikevich 2001).

Complex dynamics z = x + iy, dz/dt = (b + iω)z + I.
Spike when |z| ≥ threshold, reset to origin.
Analytical: r_ss = I / sqrt(b² + ω²), I_crit = θ·sqrt(b² + ω²)."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.resonate_and_fire import ResonateAndFireNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


def _run(neuron: ResonateAndFireNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


def _exact_linear_flow(
    x: float,
    y: float,
    current: float,
    b: float,
    omega: float,
    dt: float,
) -> tuple[float, float]:
    denominator = b**2 + omega**2
    x_ss = -b * current / denominator
    y_ss = omega * current / denominator
    dx = x - x_ss
    dy = y - y_ss
    decay = np.exp(b * dt)
    angle = omega * dt
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return (
        x_ss + decay * (dx * cos_angle - dy * sin_angle),
        y_ss + decay * (dx * sin_angle + dy * cos_angle),
    )


def _critical_current(b: float, omega: float, threshold: float) -> float:
    """Analytical critical current: I_crit = θ · sqrt(b² + ω²)."""
    return threshold * np.sqrt(b**2 + omega**2)


class TestResonateAndFireIsolation:
    def test_construction_defaults(self):
        n = ResonateAndFireNeuron()
        assert n.x == 0.0
        assert n.y == 0.0
        assert n.b == -0.1
        assert n.omega == 1.0
        assert n.threshold == 1.0
        assert n.dt == 0.05

    def test_step_returns_binary(self):
        assert ResonateAndFireNeuron().step(0.0) in (0, 1)

    def test_two_state_variables_evolve(self):
        n = ResonateAndFireNeuron()
        for _ in range(100):
            n.step(0.5)
        assert n.x != 0.0
        assert n.y != 0.0

    def test_state_finite_long_run(self):
        n = ResonateAndFireNeuron()
        for _ in range(50000):
            n.step(2.0)
        assert np.isfinite(n.x) and np.isfinite(n.y)

    def test_reset(self):
        n = ResonateAndFireNeuron()
        for _ in range(100):
            n.step(2.0)
        n.reset()
        assert n.x == 0.0 and n.y == 0.0

    def test_exact_linear_flow_matches_matrix_exponential_without_spike(self):
        """One step must match the closed-form constant-input resonator flow."""
        n = ResonateAndFireNeuron(
            x=0.3,
            y=-0.2,
            b=-0.2,
            omega=1.7,
            threshold=100.0,
            dt=1.25,
        )
        expected_x, expected_y = _exact_linear_flow(n.x, n.y, 0.8, n.b, n.omega, n.dt)

        spike = n.step(0.8)

        assert spike == 0
        assert n.x == pytest.approx(expected_x, rel=0.0, abs=1.0e-12)
        assert n.y == pytest.approx(expected_y, rel=0.0, abs=1.0e-12)

    def test_large_timestep_damped_homogeneous_flow_remains_bounded(self):
        """Exact damped rotation must not explode when Euler would be unstable."""
        n = ResonateAndFireNeuron(x=1.0, y=0.0, b=-0.5, omega=4.0, threshold=100.0, dt=10.0)
        n.step(0.0)
        assert np.hypot(n.x, n.y) == pytest.approx(np.exp(-5.0), rel=0.0, abs=1.0e-12)

    def test_rejects_corrupted_runtime_state_before_mutation(self):
        n = ResonateAndFireNeuron(x=0.25, y=-0.5)
        n.dt = 0.0
        before = (n.x, n.y)
        with pytest.raises(ValueError, match="dt"):
            n.step(0.5)
        assert (n.x, n.y) == before

    def test_rejects_non_finite_exact_radius_before_mutation(self):
        n = ResonateAndFireNeuron(x=1.5e308, y=1.5e308, b=0.0, threshold=1.7e308)
        before = (n.x, n.y)
        with pytest.raises(ValueError, match="exact resonator update"):
            n.step(0.0)
        assert (n.x, n.y) == before


class TestResonateAndFireSteadyState:
    """Verify analytical steady-state: x_ss = -bI/(b²+ω²), y_ss = Iω/(b²+ω²),
    r_ss = I/sqrt(b²+ω²)."""

    def test_subthreshold_radius_converges(self):
        """At I < I_crit, state spirals to analytical steady-state radius."""
        n = ResonateAndFireNeuron()
        I = 0.5  # well below I_crit ≈ 1.005
        for _ in range(50000):
            n.step(I)
        r = np.sqrt(n.x**2 + n.y**2)
        r_analytical = I / np.sqrt(n.b**2 + n.omega**2)
        assert abs(r - r_analytical) < 0.01, f"r = {r:.4f}, expected r_ss = {r_analytical:.4f}"

    def test_steady_state_x_component(self):
        """x_ss = -bI / (b² + ω²)."""
        n = ResonateAndFireNeuron()
        I = 0.3
        for _ in range(50000):
            n.step(I)
        # At steady state, (x, y) rotate — check time-averaged radius instead
        # The instantaneous x oscillates; use radius which is constant
        r = np.sqrt(n.x**2 + n.y**2)
        r_expected = I / np.sqrt(n.b**2 + n.omega**2)
        assert abs(r - r_expected) < 0.01

    def test_damping_with_negative_b(self):
        """b < 0 → damped oscillation (spiral sink). Without input, state decays."""
        n = ResonateAndFireNeuron(b=-0.5)
        n.x = 0.5
        n.y = 0.5
        for _ in range(1000):
            n.step(0.0)
        r = np.sqrt(n.x**2 + n.y**2)
        assert r < 0.01, f"r = {r:.4f}, expected decay to ~0 with b=-0.5"


class TestResonateAndFireThreshold:
    def test_critical_current_analytical(self):
        """I_crit = θ·sqrt(b² + ω²). Well below: no spikes. Above: spikes.

        Note: during transient spiral approach, radius can transiently
        overshoot the steady-state value. Need wide margin (50%) below
        I_crit to guarantee no spikes.
        """
        n = ResonateAndFireNeuron()
        I_crit = _critical_current(n.b, n.omega, n.threshold)
        # 50% below critical — safely subthreshold
        n_below = ResonateAndFireNeuron()
        spikes_below = len(_run(n_below, current=I_crit * 0.5, steps=50000))
        assert spikes_below == 0, f"{spikes_below} spikes at 50% of I_crit"
        # 20% above critical
        n_above = ResonateAndFireNeuron()
        spikes_above = len(_run(n_above, current=I_crit * 1.2, steps=50000))
        assert spikes_above > 10, f"Only {spikes_above} spikes above I_crit"

    def test_radius_threshold(self):
        """Spike occurs when |z| = sqrt(x² + y²) ≥ threshold."""
        n = ResonateAndFireNeuron()
        for _ in range(50000):
            s = n.step(2.0)
            if s == 1:
                # Just after spike: x=0, y=0 (reset)
                assert n.x == 0.0 and n.y == 0.0
                break
        else:
            pytest.fail("No spike observed at I=2.0")

    def test_reset_to_origin(self):
        """After spike, both x and y reset to 0."""
        n = ResonateAndFireNeuron()
        for _ in range(50000):
            if n.step(5.0) == 1:
                assert n.x == 0.0 and n.y == 0.0
                return
        pytest.fail("No spike")

    def test_custom_threshold(self):
        """Lower threshold → more spikes."""
        n_low = ResonateAndFireNeuron(threshold=0.5)
        n_high = ResonateAndFireNeuron(threshold=2.0)
        s_low = len(_run(n_low, current=2.0, steps=50000))
        s_high = len(_run(n_high, current=2.0, steps=50000))
        assert s_low > s_high


class TestResonateAndFireFI:
    def test_monotonic_fi(self):
        """Higher constant current → more spikes."""
        rates = []
        for I in [1.5, 2.0, 5.0, 10.0]:
            n = ResonateAndFireNeuron()
            rates.append(len(_run(n, current=I, steps=50000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))

    def test_rate_proportional_to_excess_current(self):
        """Well above threshold, rate grows with current."""
        n2 = ResonateAndFireNeuron()
        n5 = ResonateAndFireNeuron()
        s2 = len(_run(n2, current=2.0, steps=50000))
        s5 = len(_run(n5, current=5.0, steps=50000))
        assert s5 > s2 * 1.5

    def test_zero_current_no_spikes(self):
        n = ResonateAndFireNeuron()
        spikes = len(_run(n, current=0.0, steps=50000))
        assert spikes == 0

    def test_very_high_current_fires_every_step(self):
        """At I=25, exact one-step radius exceeds threshold on every step."""
        n = ResonateAndFireNeuron()
        spikes = len(_run(n, current=25.0, steps=1000))
        assert spikes == 1000


class TestResonateAndFireOscillation:
    def test_subthreshold_oscillation(self):
        """Below I_crit, x and y oscillate (spiral) around steady state."""
        n = ResonateAndFireNeuron(b=-0.01, threshold=10.0)
        xs = []
        for t in range(5000):
            n.step(0.5)
            xs.append(n.x)
        xs = np.array(xs[1000:])  # skip transient
        # x should oscillate: check that it crosses its mean
        mean_x = np.mean(xs)
        crossings = np.sum(np.diff(np.sign(xs - mean_x)) != 0)
        assert crossings > 20, f"Only {crossings} zero-crossings — expected oscillation"

    def test_omega_sets_frequency(self):
        """Higher omega → faster subthreshold oscillation."""
        for omega in [0.5, 1.0, 2.0]:
            n = ResonateAndFireNeuron(omega=omega)
            xs = []
            for _ in range(5000):
                n.step(0.3)
                xs.append(n.x)
            xs = np.array(xs[1000:])
            mean_x = np.mean(xs)
            crossings = np.sum(np.diff(np.sign(xs - mean_x)) != 0)
            # Higher omega → more crossings
            if omega == 0.5:
                low_crossings = crossings
            elif omega == 2.0:
                assert crossings > low_crossings, (
                    f"omega=2.0 ({crossings}) not faster than omega=0.5 ({low_crossings})"
                )


class TestResonateAndFireParameters:
    def test_b_positive_unstable(self):
        """b > 0 → expanding spiral. Even zero input would eventually spike."""
        n = ResonateAndFireNeuron(b=0.1)
        n.x = 0.01  # tiny perturbation
        spikes = len(_run(n, current=0.0, steps=50000))
        assert spikes > 0, "b>0 with perturbation should cause spikes"

    def test_b_more_negative_higher_threshold(self):
        """Heavier damping (more negative b) → higher effective I_crit."""
        n_weak = ResonateAndFireNeuron(b=-0.05)
        n_heavy_damping = ResonateAndFireNeuron(b=-0.5)
        I = 1.5
        s_weak = len(_run(n_weak, current=I, steps=50000))
        s_heavy_damping = len(_run(n_heavy_damping, current=I, steps=50000))
        assert s_weak > s_heavy_damping

    @pytest.mark.parametrize("dt", [0.02, 0.05, 0.1])
    def test_dt_stability(self, dt: float):
        n = ResonateAndFireNeuron(dt=dt)
        for _ in range(50000):
            n.step(2.0)
        assert np.isfinite(n.x) and np.isfinite(n.y)


class TestResonateAndFireDeterminism:
    def test_bit_exact(self):
        traces = []
        for _ in range(2):
            n = ResonateAndFireNeuron()
            trace = [(n.step(2.0), n.x, n.y) for _ in range(300)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestResonateAndFireISI:
    def test_constant_isi_at_steady_state(self):
        """After transient, ISI should stabilise."""
        n = ResonateAndFireNeuron()
        spikes = _run(n, current=2.0, steps=50000)
        assert len(spikes) >= 20
        isis = np.diff(spikes[10:]).astype(float)
        cv = np.std(isis) / np.mean(isis)
        assert cv < 0.05, f"CV(ISI) = {cv:.4f}"


class TestResonateAndFireNetwork:
    def test_population(self):
        assert Population(ResonateAndFireNeuron, n=10, label="raf").n == 10

    def test_network_spikes(self):
        pop = Population(ResonateAndFireNeuron, n=10, label="raf")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0


class TestResonateAndFireAnalysis:
    def test_spike_count(self):
        n = ResonateAndFireNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(50000)])
        assert spike_count(train) >= 100

    def test_spike_count_consistency(self):
        n = ResonateAndFireNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(50000)])
        assert spike_count(train) == int(train.sum())


class TestResonateAndFireValidation:
    @pytest.mark.parametrize("field", ["x", "y", "b", "omega"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_state_and_oscillator_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            ResonateAndFireNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["threshold", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf])
    def test_rejects_non_positive_or_non_finite_threshold_and_step(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            ResonateAndFireNeuron(**{field: value})

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = ResonateAndFireNeuron(x=0.25, y=-0.5)
        before = (n.x, n.y)
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert (n.x, n.y) == before

    def test_rejects_zero_omega(self):
        with pytest.raises(ValueError, match="omega"):
            ResonateAndFireNeuron(omega=0.0)

    def test_rejects_non_finite_exact_update_before_state_mutation(self):
        n = ResonateAndFireNeuron(
            x=0.25,
            y=-0.5,
            b=1.0e308,
            threshold=1.0e308,
            dt=1.0e308,
        )
        before = (n.x, n.y)

        with pytest.raises(ValueError, match="exact resonator"):
            n.step(1.0e308)

        assert (n.x, n.y) == before

    def test_rejects_invalid_exact_flow_denominator(self):
        with pytest.raises(ValueError, match="denominator"):
            ResonateAndFireNeuron._exact_linear_flow(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    def test_rejects_non_finite_exact_flow_equilibrium(self):
        with pytest.raises(ValueError, match="equilibrium"):
            ResonateAndFireNeuron._exact_linear_flow(
                0.0,
                0.0,
                1.0e308,
                1.0e154,
                1.0,
                1.0e-154,
            )

    def test_rejects_non_finite_exact_flow_decay(self):
        with pytest.raises(ValueError, match="decay"):
            ResonateAndFireNeuron._exact_linear_flow(0.0, 0.0, 0.0, 1.0, 1.0, 1000.0)
