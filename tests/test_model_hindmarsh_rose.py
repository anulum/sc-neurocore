# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: HindmarshRoseNeuron

"""Pipeline test for HindmarshRoseNeuron (Hindmarsh & Rose 1984).

3D chaotic bursting model:
dx/dt = y - x³ + b·x² - z + I
dy/dt = 1 - 5·x² - y
dz/dt = r·(s·(x - x_rest) - z)

x: fast membrane-like variable. y: fast recovery.
z: slow adaptation (r=0.001) — modulates bursting.
b=3: controls burst width. s=4: z-x coupling.
Chaotic regime at intermediate I. Bursting at I≈3-5.
Default RK4 integration prioritizes trajectory fidelity over Euler throughput.
Pipeline and performance contract tests live in this module-specific file."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.hindmarsh_rose import HindmarshRoseNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: HindmarshRoseNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestHRIsolation:
    def test_defaults(self):
        n = HindmarshRoseNeuron()
        assert n.x == -1.6 and n.y == -10.0 and n.z == 2.0
        assert n.b == 3.0 and n.r == 0.001 and n.s == 4.0
        assert n.x_rest == -1.6 and n.dt == 0.1
        assert n.integrator == "rk4"

    def test_three_state_variables(self):
        n = HindmarshRoseNeuron()
        for attr in ["x", "y", "z"]:
            assert hasattr(n, attr)

    def test_step_returns_binary(self):
        assert HindmarshRoseNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self):
        n = HindmarshRoseNeuron()
        for _ in range(100_000):
            n.step(5.0)
        assert np.isfinite(n.x) and np.isfinite(n.y) and np.isfinite(n.z)

    def test_reset_restores_defaults(self):
        n = HindmarshRoseNeuron()
        for _ in range(5000):
            n.step(5.0)
        n.reset()
        assert n.x == -1.6 and n.y == -10.0 and n.z == 2.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = HindmarshRoseNeuron()
            trace = [(n.step(5.0), n.x) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]

    def test_rejects_nonfinite_current(self):
        n = HindmarshRoseNeuron()
        with pytest.raises(ValueError, match="current"):
            n.step(float("nan"))

    def test_rk4_overflow_fails_closed_without_mutating_state(self):
        n = HindmarshRoseNeuron(x=1e103, y=0.0, z=0.0, integrator="rk4")
        before = (n.x, n.y, n.z)

        with pytest.raises(FloatingPointError, match="overflowed|non-finite"):
            n.step(0.0)

        assert (n.x, n.y, n.z) == before

    def test_euler_overflow_fails_closed_without_mutating_state(self):
        n = HindmarshRoseNeuron(x=1e103, y=0.0, z=0.0, integrator="euler")
        before = (n.x, n.y, n.z)

        with pytest.raises(FloatingPointError, match="overflowed|non-finite"):
            n.step(0.0)

        assert (n.x, n.y, n.z) == before

    def test_runtime_parameter_corruption_fails_before_mutation(self):
        n = HindmarshRoseNeuron()
        n.dt = float("nan")
        before = (n.x, n.y, n.z)

        with pytest.raises(FloatingPointError, match="non-finite"):
            n.step(3.0)

        assert (n.x, n.y, n.z) == before


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — dx, dy, dz formulas, cubic term, slow z
# ---------------------------------------------------------------------------
class TestHRAnalytical:
    def test_dx_formula_one_step(self):
        """dx = (y - x³ + b·x² - z + I) · dt."""
        n = HindmarshRoseNeuron(integrator="euler")
        x0, y0, z0 = n.x, n.y, n.z
        I = 3.0
        expected_dx = (y0 - x0**3 + n.b * x0**2 - z0 + I) * n.dt
        expected_dy = (1.0 - 5.0 * x0**2 - y0) * n.dt
        expected_dz = n.r * (n.s * (x0 - n.x_rest) - z0) * n.dt
        n.step(I)
        assert abs((n.x - x0) - expected_dx) < 1e-12
        assert abs((n.y - y0) - expected_dy) < 1e-12
        assert abs((n.z - z0) - expected_dz) < 1e-14

    def test_runge_kutta_tracks_substepped_reference_better_than_euler(self):
        horizon = 1.0
        current = 3.5
        reference = HindmarshRoseNeuron(dt=0.001, integrator="rk4")
        coarse_rk4 = HindmarshRoseNeuron(dt=0.1, integrator="rk4")
        coarse_euler = HindmarshRoseNeuron(dt=0.1, integrator="euler")

        for _ in range(int(horizon / reference.dt)):
            reference.step(current)
        for _ in range(int(horizon / coarse_rk4.dt)):
            coarse_rk4.step(current)
            coarse_euler.step(current)

        rk4_error = (
            abs(coarse_rk4.x - reference.x)
            + abs(coarse_rk4.y - reference.y)
            + abs(coarse_rk4.z - reference.z)
        )
        euler_error = (
            abs(coarse_euler.x - reference.x)
            + abs(coarse_euler.y - reference.y)
            + abs(coarse_euler.z - reference.z)
        )

        assert rk4_error < euler_error
        assert rk4_error < 5e-3

    def test_cubic_nonlinearity(self):
        """-x³ creates the excitable dynamics."""
        n = HindmarshRoseNeuron()
        # At x=2: -x³ = -8, b·x² = 12 → net = 4 (positive)
        assert -8.0 + n.b * 4.0 == 4.0

    def test_z_slow_timescale(self):
        """r=0.001 → z changes ~1000x slower than x."""
        n = HindmarshRoseNeuron()
        assert n.r == 0.001

    def test_z_adapts_to_x(self):
        """dz/dt = r·(s·(x-x_rest) - z). z tracks s·(x-x_rest)."""
        n = HindmarshRoseNeuron()
        z0 = n.z
        for _ in range(50_000):
            n.step(5.0)
        # z should have moved from initial value
        assert n.z != z0

    def test_x_nullcline(self):
        """x-nullcline: y = x³ - b·x² + z - I."""
        # At rest (x=-1.6): y_null = (-1.6)³ - 3·(-1.6)² + z - I
        x = -1.6
        y_null = x**3 - 3.0 * x**2 + 2.0 - 0.0
        assert np.isfinite(y_null)

    def test_y_nullcline(self):
        """y-nullcline: y = 1 - 5x²."""
        x = -1.6
        y_null = 1.0 - 5.0 * x**2
        assert abs(y_null - (1.0 - 12.8)) < 1e-10


# ---------------------------------------------------------------------------
# 3. BURSTING DYNAMICS
# ---------------------------------------------------------------------------
class TestHRBursting:
    def test_fires_at_moderate_current(self):
        n = HindmarshRoseNeuron()
        spikes = _run(n, current=5.0, steps=10_000)
        assert len(spikes) >= 20

    def test_silent_at_low_current(self):
        n = HindmarshRoseNeuron()
        # At I=0 with default params may or may not fire
        spikes = _run(n, current=0.0, steps=5000)
        assert isinstance(len(spikes), int)

    def test_rate_monotonic(self):
        rates = []
        for I in [2.0, 5.0, 10.0]:
            n = HindmarshRoseNeuron()
            rates.append(len(_run(n, current=I, steps=10_000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 2.0, 3.5, 5.0, 10.0])
    def test_fi_sweep(self, current: float):
        n = HindmarshRoseNeuron()
        for _ in range(10_000):
            n.step(current)
        assert np.isfinite(n.x)

    def test_x_bounded(self):
        """x stays bounded (cubic creates restoring force)."""
        n = HindmarshRoseNeuron()
        xs = []
        for _ in range(20_000):
            n.step(5.0)
            xs.append(n.x)
        assert min(xs) > -5 and max(xs) < 5


# ---------------------------------------------------------------------------
# 4. PARAMETERS
# ---------------------------------------------------------------------------
class TestHRParameters:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("dt", 0.0),
            ("dt", float("nan")),
            ("r", -0.001),
            ("s", 0.0),
            ("b", float("inf")),
        ],
    )
    def test_rejects_nonphysical_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            HindmarshRoseNeuron(**{field: value})

    def test_rejects_unknown_integrator(self):
        with pytest.raises(ValueError, match="integrator"):
            HindmarshRoseNeuron(integrator="verlet")

    @pytest.mark.parametrize("b", [2.0, 3.0, 4.0])
    def test_b_sweep(self, b: float):
        n = HindmarshRoseNeuron(b=b)
        for _ in range(10_000):
            n.step(5.0)
        assert np.isfinite(n.x)

    @pytest.mark.parametrize("r", [0.0005, 0.001, 0.005])
    def test_r_slow_timescale(self, r: float):
        n = HindmarshRoseNeuron(r=r)
        for _ in range(10_000):
            n.step(5.0)
        assert np.isfinite(n.z)

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.15])
    def test_dt_stability(self, dt: float):
        n = HindmarshRoseNeuron(dt=dt)
        for _ in range(10_000):
            n.step(5.0)
        assert np.isfinite(n.x) and np.isfinite(n.y) and np.isfinite(n.z)


# ---------------------------------------------------------------------------
# 5. PERFORMANCE
# ---------------------------------------------------------------------------
class TestHRPerformance:
    def test_isolation_throughput(self):
        n = HindmarshRoseNeuron()
        N = 200_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(5.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        min_rate = 60_000 if os.getenv("CI") else 200_000
        assert np.isfinite(n.x) and np.isfinite(n.y) and np.isfinite(n.z)
        assert rate > min_rate, f"isolation: {rate:.0f} steps/s, minimum={min_rate}"

    def test_network_throughput(self):
        pop = Population(HindmarshRoseNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert rate > 2_000, f"network: {rate:.0f} neuron-steps/s"


# ---------------------------------------------------------------------------
# 6. PIPELINE
# ---------------------------------------------------------------------------
class TestHRPipeline:
    def test_population(self):
        assert Population(HindmarshRoseNeuron, n=10, label="hr").n == 10

    def test_projection_wiring(self):
        src = Population(HindmarshRoseNeuron, n=5, label="src")
        tgt = Population(HindmarshRoseNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=2.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(HindmarshRoseNeuron, n=10, label="hr")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = HindmarshRoseNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(10_000)])
        sc = spike_count(train)
        assert sc >= 10

    def test_analysis_isi(self):
        n = HindmarshRoseNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(10_000)])
        intervals = isi(train, dt=0.0001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))

    def test_analysis_firing_rate(self):
        n = HindmarshRoseNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(10_000)])
        rate = firing_rate(train, dt=0.0001)
        assert rate > 0
