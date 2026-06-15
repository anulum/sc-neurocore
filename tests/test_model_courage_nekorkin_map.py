# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: CourageNekorkinMapNeuron

"""Full pipeline test for CourageNekorkinMapNeuron (Courbage-Nekorkin-Vdovin 2007).

Canonical discontinuous two-dimensional spiking map (Chaos 17:043109;
arXiv:0712.2097, eqs. 3-5):

    x(n+1) = x(n) + F(x(n)) - y(n) - beta*H(x(n) - d) + I
    y(n+1) = y(n) + eps*(x(n) - J)

    F(x) = -m0*x        for x <= Jmin
           m1*(x - a)   for Jmin < x < Jmax
           -m0*(x - 1)  for x >= Jmax
    H(z) = 1 for z >= 0, else 0
    Jmin = a*m1/(m0 + m1), Jmax = (m0 + a*m1)/(m0 + m1)

The default parameters (m0=0.0864, m1=0.65, a=0.2 from figure 1; d=0.235, J=0.2,
beta=0.085, eps=0.02 inside the B^+ invariant-region triangle) place the model in
the published chaotic spiking-bursting regime. The map has NO clip: it stays
bounded by its own invariant attractor.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.courage_nekorkin_map import CourageNekorkinMapNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: CourageNekorkinMapNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


# Default breakpoints for assertions.
def _breakpoints(m0=0.0864, m1=0.65, a=0.2):
    am1 = a * m1
    den = m0 + m1
    return am1 / den, (m0 + am1) / den


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestCNIsolation:
    def test_defaults(self):
        n = CourageNekorkinMapNeuron()
        assert n.x == 0.0 and n.y == 0.0
        assert n.m0 == 0.0864 and n.m1 == 0.65 and n.a == 0.2
        assert n.d == 0.235 and n.j == 0.2 and n.beta == 0.085 and n.eps == 0.02
        assert n.x_threshold == 0.235

    def test_default_regime_is_valid(self):
        """Defaults satisfy the published parameter region (eq. 6): Jmin<d<Jmax, 0<J<d, m0<1."""
        n = CourageNekorkinMapNeuron()
        jmin, jmax = _breakpoints()
        assert jmin < n.d < jmax
        assert 0.0 < n.j < n.d
        assert n.m0 < 1.0

    def test_step_returns_binary(self):
        assert CourageNekorkinMapNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self):
        n = CourageNekorkinMapNeuron()
        for _ in range(50_000):
            n.step(0.0)
        assert np.isfinite(n.x) and np.isfinite(n.y)

    def test_reset_restores_state(self):
        n = CourageNekorkinMapNeuron()
        for _ in range(1000):
            n.step(0.0)
        n.reset()
        assert n.x == 0.0 and n.y == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = CourageNekorkinMapNeuron()
            trace = [(n.step(0.0), n.x, n.y) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — breakpoints, piecewise F, Heaviside, update formulas
# ---------------------------------------------------------------------------
class TestCNAnalytical:
    def test_breakpoints_formula(self):
        n = CourageNekorkinMapNeuron()
        jmin, jmax = n._breakpoints()
        exp_min, exp_max = _breakpoints()
        assert abs(jmin - exp_min) < 1e-15
        assert abs(jmax - exp_max) < 1e-15

    def test_f_lower_branch(self):
        """x <= Jmin: F(x) = -m0*x."""
        n = CourageNekorkinMapNeuron()
        jmin, _ = n._breakpoints()
        x = jmin - 0.05
        assert abs(n._f(x) - (-n.m0 * x)) < 1e-15

    def test_f_middle_branch(self):
        """Jmin < x < Jmax: F(x) = m1*(x - a)."""
        n = CourageNekorkinMapNeuron()
        jmin, jmax = n._breakpoints()
        x = 0.5 * (jmin + jmax)
        assert abs(n._f(x) - (n.m1 * (x - n.a))) < 1e-15

    def test_f_upper_branch(self):
        """x >= Jmax: F(x) = -m0*(x - 1)."""
        n = CourageNekorkinMapNeuron()
        _, jmax = n._breakpoints()
        x = jmax + 0.05
        assert abs(n._f(x) - (-n.m0 * (x - 1.0))) < 1e-15

    def test_f_continuous_at_breakpoints(self):
        """F is continuous at Jmin and Jmax by construction."""
        n = CourageNekorkinMapNeuron()
        jmin, jmax = n._breakpoints()
        d = 1e-9
        assert abs(n._f(jmin - d) - n._f(jmin + d)) < 1e-7
        assert abs(n._f(jmax - d) - n._f(jmax + d)) < 1e-7

    def test_x_update_formula_subthreshold(self):
        """Below d, H=0: x_new = x + F(x) - y + I."""
        n = CourageNekorkinMapNeuron()
        x0, y0 = n.x, n.y  # x0 = 0 < d -> H = 0
        cur = 0.05
        expected = x0 + n._f(x0) - y0 - n.beta * 0.0 + cur
        n.step(cur)
        assert abs(n.x - expected) < 1e-15

    def test_heaviside_active_above_d(self):
        """At x >= d, H=1 subtracts beta from the x update."""
        below = CourageNekorkinMapNeuron(x=0.30)  # >= d
        above = CourageNekorkinMapNeuron(x=0.30)
        x0 = 0.30
        below.beta = 0.0
        above.beta = 0.085
        below.step(0.0)
        above.step(0.0)
        # Difference between beta=0 and beta=0.085 updates is exactly -beta (H=1).
        assert abs((above.x - below.x) - (-0.085)) < 1e-15
        assert x0 >= above.d  # confirm H was active

    def test_y_update_formula(self):
        """y_new = y + eps*(x - J)."""
        n = CourageNekorkinMapNeuron()
        x0, y0 = n.x, n.y
        expected_dy = n.eps * (x0 - n.j)
        n.step(0.0)
        assert abs((n.y - y0) - expected_dy) < 1e-16

    def test_no_clip(self):
        """The canonical map has no clip — large states evolve by the raw map."""
        n = CourageNekorkinMapNeuron(x=5.0)
        x0 = 5.0
        expected = x0 + n._f(x0) - n.y - n.beta * 1.0
        n.step(0.0)
        assert abs(n.x - expected) < 1e-12


# ---------------------------------------------------------------------------
# 3. DYNAMICS — the published chaotic spiking-bursting regime
# ---------------------------------------------------------------------------
class TestCNDynamics:
    def test_sustained_bounded_spiking(self):
        """Default regime fires repeatedly and stays bounded (no clip-pegging)."""
        n = CourageNekorkinMapNeuron()
        trace, spikes = n.simulate(20_000, backend="python")
        assert spikes > 1000
        assert np.all(np.isfinite(trace))
        assert trace.max() - trace.min() < 5.0

    def test_burst_structure(self):
        """ISI distribution shows both in-burst (short) and inter-burst (long) gaps."""
        n = CourageNekorkinMapNeuron()
        prev = n.x
        times = []
        for t in range(20_000):
            if n.step(0.0) == 1:
                times.append(t)
            prev = n.x
        _ = prev
        intervals = np.diff(times)
        assert intervals.min() <= 3  # in-burst spikes
        assert intervals.max() >= 8  # inter-burst quiescence

    def test_chaos_sensitivity(self):
        """Tiny initial offset amplifies — sensitive dependence on initial conditions."""
        a = CourageNekorkinMapNeuron(x=0.0)
        b = CourageNekorkinMapNeuron(x=1e-9)
        tr_a, _ = a.simulate(2000, backend="python")
        tr_b, _ = b.simulate(2000, backend="python")
        assert abs(tr_a[-1] - tr_b[-1]) > 1e-3

    def test_quiescent_below_threshold_regime(self):
        """J < Jmin gives the excitable (non-spiking-bursting) regime — far fewer spikes."""
        jmin, _ = _breakpoints()
        excitable = CourageNekorkinMapNeuron(j=jmin - 0.05)
        _, spikes = excitable.simulate(20_000, backend="python")
        _, spikes_default = CourageNekorkinMapNeuron().simulate(20_000, backend="python")
        assert spikes < spikes_default

    @pytest.mark.parametrize("current", [-0.02, 0.0, 0.05, 0.1])
    def test_fi_sweep_finite(self, current: float):
        n = CourageNekorkinMapNeuron()
        trace, _ = n.simulate(5000, current, backend="python")
        assert np.all(np.isfinite(trace))

    def test_upward_crossing_only(self):
        n = CourageNekorkinMapNeuron()
        prev_x = n.x
        for _ in range(5000):
            spike = n.step(0.0)
            if spike == 1:
                assert prev_x < n.x_threshold
            prev_x = n.x


# ---------------------------------------------------------------------------
# 4. PARAMETERS
# ---------------------------------------------------------------------------
class TestCNParameters:
    @pytest.mark.parametrize("m1", [0.5, 0.65, 0.8])
    def test_m1_sweep(self, m1: float):
        n = CourageNekorkinMapNeuron(m1=m1)
        trace, _ = n.simulate(5000, backend="python")
        assert np.all(np.isfinite(trace))

    @pytest.mark.parametrize("beta", [0.08, 0.085, 0.09])
    def test_beta_sweep(self, beta: float):
        n = CourageNekorkinMapNeuron(beta=beta)
        trace, _ = n.simulate(5000, backend="python")
        assert np.all(np.isfinite(trace))

    @pytest.mark.parametrize("eps", [0.01, 0.02, 0.04])
    def test_eps_sweep(self, eps: float):
        n = CourageNekorkinMapNeuron(eps=eps)
        trace, _ = n.simulate(5000, backend="python")
        assert np.all(np.isfinite(trace))


# ---------------------------------------------------------------------------
# 5. PERFORMANCE
# ---------------------------------------------------------------------------
class TestCNPerformance:
    def test_isolation_throughput(self):
        n = CourageNekorkinMapNeuron()
        N = 200_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(0.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        assert rate > 100_000, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(CourageNekorkinMapNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
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
class TestCNPipeline:
    def test_population(self):
        assert Population(CourageNekorkinMapNeuron, n=10, label="cn").n == 10

    def test_projection_wiring(self):
        src = Population(CourageNekorkinMapNeuron, n=5, label="src")
        tgt = Population(CourageNekorkinMapNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=0.3, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert isinstance(mon_src.count, int)

    def test_network_spikes(self):
        pop = Population(CourageNekorkinMapNeuron, n=10, label="cn")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)

    def test_analysis_spike_count(self):
        n = CourageNekorkinMapNeuron()
        train = np.array([float(n.step(0.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 0

    def test_analysis_isi(self):
        n = CourageNekorkinMapNeuron()
        train = np.array([float(n.step(0.0)) for _ in range(5000)])
        intervals = isi(train, dt=0.001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))

    def test_analysis_firing_rate(self):
        n = CourageNekorkinMapNeuron()
        train = np.array([float(n.step(0.0)) for _ in range(5000)])
        rate = firing_rate(train, dt=0.001)
        assert rate >= 0
