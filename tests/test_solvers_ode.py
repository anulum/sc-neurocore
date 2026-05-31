# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for solvers/

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from sc_neurocore.solvers import (
    EulerSolver,
    HeunSolver,
    RK4Solver,
    DormandPrinceSolver,
    ExponentialEuler,
    ExactLIFSolver,
    StormerVerlet,
    LeapfrogSolver,
    RosenbrockEuler,
    ImplicitEuler,
    TrapezoidalRule,
    get_solver,
)


# ---------------------------------------------------------------------------
# Known test ODEs
# ---------------------------------------------------------------------------


def decay_ode(t, y):
    """dy/dt = -y. Solution: y(t) = y0 * exp(-t)."""
    return -y


def stiff_ode(t, y):
    """dy/dt = -1000*y. Very stiff decay."""
    return -1000.0 * y


def harmonic_oscillator(t, y):
    """Simple harmonic oscillator: dq/dt = p, dp/dt = -q.
    State: [q, p]. Conserves H = (q^2 + p^2) / 2.
    """
    return np.array([y[1], -y[0]])


# ---------------------------------------------------------------------------
# Euler Solver
# ---------------------------------------------------------------------------


class TestEulerSolver:
    def test_decay_direction(self):
        solver = EulerSolver()
        y = np.array([1.0])
        y_new, _ = solver.step(decay_ode, y, t=0.0, dt=0.01)
        assert y_new[0] < 1.0

    def test_convergence_order_1(self):
        """Euler error should be O(h)."""
        y0 = np.array([1.0])
        t_end = 1.0
        exact = math.exp(-t_end)
        errors = []
        for n_steps in [100, 200, 400]:
            dt = t_end / n_steps
            y = y0.copy()
            solver = EulerSolver()
            for _ in range(n_steps):
                y, _ = solver.step(decay_ode, y, 0.0, dt)
            errors.append(abs(y[0] - exact))
        # Error ratio should be ~2 for O(h)
        ratio = errors[0] / errors[1]
        assert 1.5 < ratio < 2.5


class TestHeunSolver:
    def test_convergence_order_2(self):
        y0 = np.array([1.0])
        t_end = 1.0
        exact = math.exp(-t_end)
        errors = []
        for n_steps in [50, 100, 200]:
            dt = t_end / n_steps
            y = y0.copy()
            solver = HeunSolver()
            for i in range(n_steps):
                y, _ = solver.step(decay_ode, y, i * dt, dt)
            errors.append(abs(y[0] - exact))
        ratio = errors[0] / errors[1]
        assert 3.0 < ratio < 5.0  # O(h²) → ratio ~4


class TestRK4Solver:
    def test_convergence_order_4(self):
        y0 = np.array([1.0])
        t_end = 1.0
        exact = math.exp(-t_end)
        errors = []
        for n_steps in [10, 20, 40]:
            dt = t_end / n_steps
            y = y0.copy()
            solver = RK4Solver()
            for i in range(n_steps):
                y, _ = solver.step(decay_ode, y, i * dt, dt)
            errors.append(abs(y[0] - exact))
        ratio = errors[0] / errors[1]
        assert 12.0 < ratio < 20.0  # O(h⁴) → ratio ~16

    def test_accuracy_better_than_euler(self):
        y0 = np.array([1.0])
        n_steps = 20
        dt = 1.0 / n_steps
        exact = math.exp(-1.0)

        y_e = y0.copy()
        euler = EulerSolver()
        for _ in range(n_steps):
            y_e, _ = euler.step(decay_ode, y_e, 0.0, dt)

        y_r = y0.copy()
        rk4 = RK4Solver()
        for i in range(n_steps):
            y_r, _ = rk4.step(decay_ode, y_r, i * dt, dt)

        assert abs(y_r[0] - exact) < abs(y_e[0] - exact)


# ---------------------------------------------------------------------------
# Dormand-Prince Adaptive
# ---------------------------------------------------------------------------


class TestDormandPrinceSolver:
    def test_adaptive_reaches_solution(self):
        solver = DormandPrinceSolver(atol=1e-8, rtol=1e-6)
        ts, ys = solver.integrate(decay_ode, np.array([1.0]), (0.0, 1.0))
        assert abs(ys[-1, 0] - math.exp(-1.0)) < 1e-5

    def test_step_size_adapts(self):
        solver = DormandPrinceSolver()
        ts, ys = solver.integrate(decay_ode, np.array([1.0]), (0.0, 2.0), dt0=0.001)
        dts = np.diff(ts)
        assert dts.max() > dts.min() * 1.5  # step varies

    def test_high_precision(self):
        solver = DormandPrinceSolver(atol=1e-12, rtol=1e-10)
        ts, ys = solver.integrate(decay_ode, np.array([1.0]), (0.0, 1.0))
        assert abs(ys[-1, 0] - math.exp(-1.0)) < 1e-9

    def test_zero_error_uses_max_growth_factor(self):
        solver = DormandPrinceSolver(max_factor=3.0)

        def zero_rhs(t, y):
            return np.zeros_like(y)

        y_new, dt_used, dt_next = solver.step(zero_rhs, np.array([1.0]), 0.0, 0.1)

        np.testing.assert_allclose(y_new, np.array([1.0]))
        assert dt_used == pytest.approx(0.1)
        assert dt_next == pytest.approx(0.3)

    def test_rejects_initial_step_when_error_is_large(self):
        solver = DormandPrinceSolver(atol=1e-12, rtol=1e-12, min_factor=0.2)

        y_new, dt_used, dt_next = solver.step(lambda t, y: y * y, np.array([1.0]), 0.0, 1.0)

        assert dt_used < 1.0
        assert dt_next > 0.0
        assert y_new[0] > 1.0


# ---------------------------------------------------------------------------
# Exponential Euler
# ---------------------------------------------------------------------------


class TestExponentialEuler:
    def test_exact_for_constant_current(self):
        """ExponentialEuler is exact for linear LIF with constant I."""
        tau = 20.0
        v_rest = -65.0
        solver = ExponentialEuler(tau=tau, y_rest=v_rest, r_m=1.0)

        def current_fn(t, y):
            return np.array([10.0])

        y = np.array([v_rest])
        dt = 5.0
        y_new, _ = solver.step(current_fn, y, 0.0, dt)
        expected = v_rest + 10.0 * (1.0 - math.exp(-dt / tau))
        assert abs(y_new[0] - expected) < 1e-10


# ---------------------------------------------------------------------------
# Exact LIF Solver
# ---------------------------------------------------------------------------


class TestExactLIFSolver:
    def test_spike_time_matches_analytical(self):
        solver = ExactLIFSolver(tau=10.0, v_rest=-65.0, v_thresh=-50.0, r_m=1.0)
        # V_inf = -65 + 20 = -45 (above threshold)
        t = solver.next_spike_time(v0=-65.0, current=20.0)
        assert t is not None
        v_at_t = solver.evolve_to_time(-65.0, t, 20.0)
        assert abs(v_at_t - solver.v_thresh) < 1e-8

    def test_subthreshold_no_spike(self):
        solver = ExactLIFSolver(tau=10.0, v_rest=-65.0, v_thresh=-50.0, r_m=1.0)
        t = solver.next_spike_time(v0=-65.0, current=10.0)
        assert t is None  # V_inf = -55, never reaches -50

    def test_already_threshold_spikes_immediately(self):
        solver = ExactLIFSolver(v_thresh=-50.0)

        assert solver.next_spike_time(v0=-50.0, current=20.0) == 0.0

    def test_evolve_to_time_at_zero(self):
        solver = ExactLIFSolver()
        v = solver.evolve_to_time(v0=-60.0, t=0.0, current=0.0)
        assert v == pytest.approx(-60.0)

    def test_firing_rate_suprathreshold(self):
        solver = ExactLIFSolver(tau=10.0, v_rest=-65.0, v_thresh=-50.0, v_reset=-65.0, r_m=1.0)
        rate = solver.firing_rate(current=30.0)
        assert rate > 0

    def test_firing_rate_subthreshold(self):
        solver = ExactLIFSolver(tau=10.0, v_rest=-65.0, v_thresh=-50.0, r_m=1.0)
        rate = solver.firing_rate(current=5.0)
        assert rate == 0.0

    def test_firing_rate_zero_for_immediate_spike_time(self):
        solver = ExactLIFSolver(v_thresh=-50.0, v_reset=-50.0)

        assert solver.firing_rate(current=20.0) == 0.0

    def test_simulate_produces_spikes(self):
        solver = ExactLIFSolver(tau=10.0, v_rest=-65.0, v_thresh=-50.0, v_reset=-65.0, r_m=1.0)
        spikes, _ = solver.simulate(current=30.0, t_end=100.0)
        assert len(spikes) >= 2

    def test_simulate_breaks_when_next_spike_exceeds_window(self):
        solver = ExactLIFSolver(tau=10.0, v_rest=-65.0, v_thresh=-50.0, v_reset=-65.0, r_m=1.0)

        spikes, voltages = solver.simulate(current=20.0, t_end=1.0)

        assert spikes == []
        assert voltages == []


# ---------------------------------------------------------------------------
# Symplectic Solvers
# ---------------------------------------------------------------------------


class TestSymplecticSolvers:
    def _run_oscillator(self, solver, n_steps=10000, dt=0.01):
        y = np.array([1.0, 0.0])  # q=1, p=0
        energies = []
        for _ in range(n_steps):
            y, _ = solver.step(harmonic_oscillator, y, 0.0, dt)
            energies.append(0.5 * (y[0] ** 2 + y[1] ** 2))
        return energies

    def test_verlet_energy_conservation(self):
        energies = self._run_oscillator(StormerVerlet(), n_steps=10000)
        assert abs(energies[0] - energies[-1]) / energies[0] < 0.01

    def test_leapfrog_energy_conservation(self):
        energies = self._run_oscillator(LeapfrogSolver(), n_steps=10000)
        assert abs(energies[0] - energies[-1]) / energies[0] < 0.01

    @pytest.mark.parametrize("solver", [StormerVerlet(), LeapfrogSolver()])
    @pytest.mark.parametrize(
        ("y", "t", "dt", "match"),
        [
            (np.array([1.0]), 0.0, 0.01, "even-length"),
            (np.array([[1.0, 0.0]]), 0.0, 0.01, "1-D"),
            (np.array([1.0, np.nan]), 0.0, 0.01, "finite"),
            (np.array([1.0, 0.0]), True, 0.01, "t"),
            (np.array([1.0, 0.0]), float("nan"), 0.01, "t"),
            (np.array([1.0, 0.0]), 0.0, True, "dt"),
            (np.array([1.0, 0.0]), 0.0, 0.0, "dt"),
            (np.array([1.0, 0.0]), 0.0, float("inf"), "dt"),
        ],
    )
    def test_symplectic_solvers_reject_invalid_state_contracts(self, solver, y, t, dt, match):
        with pytest.raises(ValueError, match=match):
            solver.step(harmonic_oscillator, y, t, dt)

    @pytest.mark.parametrize("solver", [StormerVerlet(), LeapfrogSolver()])
    def test_symplectic_solvers_reject_bad_rhs_shape(self, solver):
        with pytest.raises(ValueError, match="state shape"):
            solver.step(lambda _t, _y: np.array([1.0]), np.array([1.0, 0.0]), 0.0, 0.01)

    @pytest.mark.parametrize("solver", [StormerVerlet(), LeapfrogSolver()])
    def test_symplectic_solvers_reject_nonfinite_rhs(self, solver):
        with pytest.raises(ValueError, match="finite"):
            solver.step(
                lambda _t, _y: np.array([0.0, np.nan]),
                np.array([1.0, 0.0]),
                0.0,
                0.01,
            )

    @pytest.mark.parametrize("solver", [StormerVerlet(), LeapfrogSolver()])
    def test_symplectic_solvers_reject_nonfinite_candidate_state(self, solver):
        def rhs(_t, _y):
            return np.array([1e308, 1e308])

        with pytest.raises(ValueError, match="non-finite state"):
            solver.step(rhs, np.array([1.0, 0.0]), 0.0, 2.0)


# ---------------------------------------------------------------------------
# Implicit / Stiff Solvers
# ---------------------------------------------------------------------------


class TestImplicitSolvers:
    def test_rosenbrock_euler_stable_for_stiff(self):
        solver = RosenbrockEuler()
        y = np.array([1.0])
        dt = 0.01
        for _ in range(10):
            y, _ = solver.step(stiff_ode, y, 0.0, dt)
        assert 0.0 <= y[0] < 1e-8

    def test_rosenbrock_euler_rejects_invalid_parameters(self):
        with pytest.raises(ValueError, match="gamma must be positive"):
            RosenbrockEuler(gamma=0.0)
        with pytest.raises(ValueError, match="jacobian_epsilon must be positive"):
            RosenbrockEuler(jacobian_epsilon=0.0)

    def test_implicit_euler_stable_for_stiff(self):
        solver = ImplicitEuler(max_iterations=50)
        y = np.array([1.0])
        dt = 0.001
        for _ in range(1000):
            y, _ = solver.step(stiff_ode, y, 0.0, dt)
        assert abs(y[0]) < 1e-3  # decayed

    def test_trapezoidal_stable_for_stiff(self):
        solver = TrapezoidalRule(max_iterations=50)
        y = np.array([1.0])
        dt = 0.001
        for _ in range(1000):
            y, _ = solver.step(stiff_ode, y, 0.0, dt)
        assert abs(y[0]) < 1e-3

    def test_trapezoidal_more_accurate_than_implicit_euler(self):
        y0 = np.array([1.0])
        dt = 0.01
        exact = math.exp(-10.0)  # t=10*dt=0.1 for standard decay
        n = 10

        ye = y0.copy()
        ie = ImplicitEuler()
        for _ in range(n):
            ye, _ = ie.step(decay_ode, ye, 0.0, dt)

        yt = y0.copy()
        tr = TrapezoidalRule()
        for _ in range(n):
            yt, _ = tr.step(decay_ode, yt, 0.0, dt)

        assert abs(yt[0] - exact) < abs(ye[0] - exact)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    @pytest.mark.parametrize("name", ["euler", "heun", "rk4"])
    def test_get_solver(self, name):
        solver = get_solver(name)
        y, _ = solver.step(decay_ode, np.array([1.0]), 0.0, 0.01)
        assert y[0] < 1.0

    def test_unknown_solver_raises(self):
        with pytest.raises(ValueError):
            get_solver("nonexistent_solver")

    def test_dp45_with_kwargs(self):
        solver = get_solver("dp45", atol=1e-6, rtol=1e-4)
        assert isinstance(solver, DormandPrinceSolver)

    def test_exponential_euler_with_kwargs(self):
        solver = get_solver("exponential_euler", tau=5.0, y_rest=-60.0)
        assert isinstance(solver, ExponentialEuler)
        assert solver.tau == pytest.approx(5.0)
        assert solver.y_rest == pytest.approx(-60.0)

    @pytest.mark.parametrize("name", ["rosenbrock", "rosenbrock_euler"])
    def test_rosenbrock_aliases(self, name):
        solver = get_solver(name, gamma=0.5)
        assert isinstance(solver, RosenbrockEuler)
        assert solver.gamma == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


class TestSolverBenchmark:
    def test_rk4_throughput(self):
        """100k RK4 steps in < 5s."""
        solver = RK4Solver()
        y = np.array([1.0])
        t0 = time.perf_counter()
        for i in range(100_000):
            y, _ = solver.step(decay_ode, y, 0.0, 1e-5)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"100k RK4 steps took {elapsed:.2f}s"

    def test_exact_lif_throughput(self):
        """1000 exact LIF simulations, 100ms each."""
        solver = ExactLIFSolver(tau=10.0, v_rest=-65.0, v_thresh=-50.0, v_reset=-65.0, r_m=1.0)
        t0 = time.perf_counter()
        for _ in range(1000):
            solver.simulate(current=25.0, t_end=100.0)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"1000 LIF sims took {elapsed:.2f}s"
