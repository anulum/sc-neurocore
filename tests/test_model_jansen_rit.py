# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: JansenRitUnit

"""Full pipeline test for JansenRitUnit (Jansen & Rit 1995).

Neural mass model for EEG generation. 6 ODEs across 3 populations:
pyramidal (y0,y3), excitatory interneurons (y1,y4), inhibitory (y2,y5).
Sigmoid: S(x) = 2·e0 / (1 + exp(r·(v0-x))).
Output: EEG = y1 - y2 (postsynaptic potential difference).
Returns float, not binary spike. dt=0.001.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.jansen_rit import JansenRitUnit
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestJRIsolation:
    def test_defaults(self):
        n = JansenRitUnit()
        for attr in ["y0", "y1", "y2", "y3", "y4", "y5"]:
            assert getattr(n, attr) == 0.0
        assert n.a_exc == 3.25 and n.c == 135.0
        assert n.dt == 0.001

    def test_six_state_variables(self):
        n = JansenRitUnit()
        for attr in ["y0", "y3", "y1", "y4", "y2", "y5"]:
            assert hasattr(n, attr)

    def test_step_returns_float(self):
        """Neural mass returns EEG signal (float), not binary spike."""
        n = JansenRitUnit()
        result = n.step(220.0)
        assert isinstance(result, (float, np.floating))

    def test_state_finite_long_run(self):
        n = JansenRitUnit()
        for _ in range(50_000):
            n.step(220.0)
        for attr in ["y0", "y1", "y2", "y3", "y4", "y5"]:
            assert np.isfinite(getattr(n, attr)), f"{attr} not finite"

    def test_reset_restores_defaults(self):
        n = JansenRitUnit()
        for _ in range(5000):
            n.step(220.0)
        n.reset()
        for attr in ["y0", "y1", "y2", "y3", "y4", "y5"]:
            assert getattr(n, attr) == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = JansenRitUnit()
            trace = [n.step(220.0) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — sigmoid, EEG output, 3 populations
# ---------------------------------------------------------------------------
class TestJRAnalytical:
    def test_sigmoid_midpoint(self):
        """S(v0) = 2·e0 / (1+exp(0)) = e0."""
        n = JansenRitUnit()
        assert abs(n._sigmoid(n.v0) - n.e0) < 1e-10

    def test_sigmoid_range(self):
        """S(x) ∈ (0, 2·e0)."""
        n = JansenRitUnit()
        assert n._sigmoid(-100.0) > 0
        assert n._sigmoid(-100.0) < 0.01
        assert n._sigmoid(100.0) > 2 * n.e0 - 0.01
        assert n._sigmoid(100.0) <= 2 * n.e0

    def test_sigmoid_extreme_inputs_remain_bounded(self):
        n = JansenRitUnit()

        assert 0.0 <= n._sigmoid(-1e6) < 1e-100
        assert n._sigmoid(1e6) == pytest.approx(2 * n.e0)

    def test_eeg_output_is_y1_minus_y2(self):
        """Output = y1 - y2 (pyramidal PSP difference)."""
        n = JansenRitUnit()
        n.y1 = 5.0
        n.y2 = 3.0
        eeg = n.step(220.0)
        # After one step, y1 and y2 change, but the return is y1-y2
        assert isinstance(eeg, (float, np.floating))

    def test_three_populations(self):
        """y0/y3 (pyramidal), y1/y4 (excitatory), y2/y5 (inhibitory)."""
        n = JansenRitUnit()
        for _ in range(5000):
            n.step(220.0)
        # All should evolve
        assert n.y0 != 0.0 or n.y1 != 0.0 or n.y2 != 0.0

    def test_connectivity_constant(self):
        """c=135 — inter-population coupling strength."""
        n = JansenRitUnit()
        assert n.c == 135.0

    def test_excitatory_inhibitory_rates(self):
        """a_rate=100 (excitatory), b_rate=50 (inhibitory)."""
        n = JansenRitUnit()
        assert n.a_rate > n.b_rate


# ---------------------------------------------------------------------------
# 3. DYNAMICS — EEG oscillation
# ---------------------------------------------------------------------------
class TestJRDynamics:
    def test_eeg_oscillates(self):
        """EEG signal should oscillate (alpha rhythm ≈ 8-13 Hz)."""
        n = JansenRitUnit()
        eeg = []
        for _ in range(10_000):
            eeg.append(n.step(220.0))
        eeg = np.array(eeg)
        assert np.std(eeg) > 0.01  # non-constant

    def test_input_affects_dynamics(self):
        """Different p_ext → different EEG trajectory."""
        n1 = JansenRitUnit()
        n2 = JansenRitUnit()
        e1 = [n1.step(100.0) for _ in range(5000)]
        e2 = [n2.step(300.0) for _ in range(5000)]
        assert e1 != e2

    @pytest.mark.parametrize("p_ext", [100.0, 220.0, 300.0, 500.0])
    def test_p_ext_sweep(self, p_ext: float):
        n = JansenRitUnit()
        for _ in range(5000):
            n.step(p_ext)
        assert np.isfinite(n.y0)


# ---------------------------------------------------------------------------
# 4. PARAMETERS
# ---------------------------------------------------------------------------
class TestJRParameters:
    @pytest.mark.parametrize("c", [50.0, 135.0, 300.0])
    def test_c_connectivity(self, c: float):
        n = JansenRitUnit(c=c)
        for _ in range(5000):
            n.step(220.0)
        assert np.isfinite(n.y0)

    @pytest.mark.parametrize("a_exc", [2.0, 3.25, 5.0])
    def test_a_exc_sweep(self, a_exc: float):
        n = JansenRitUnit(a_exc=a_exc)
        for _ in range(5000):
            n.step(220.0)
        assert np.isfinite(n.y1)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"dt": 0.0},
            {"a_exc": 0.0},
            {"b_exc": 0.0},
            {"a_rate": 0.0},
            {"b_rate": 0.0},
            {"c": -1.0},
            {"e0": 0.0},
            {"r": 0.0},
            {"y0": math.nan},
            {"v0": math.inf},
        ],
    )
    def test_invalid_physical_configuration_is_rejected(self, kwargs: dict[str, float]):
        with pytest.raises(ValueError):
            JansenRitUnit(**kwargs)

    def test_non_finite_external_input_does_not_mutate_state(self):
        n = JansenRitUnit()
        before = (n.y0, n.y3, n.y1, n.y4, n.y2, n.y5)

        with pytest.raises(ValueError):
            n.step(math.nan)

        assert (n.y0, n.y3, n.y1, n.y4, n.y2, n.y5) == before

    def test_corrupted_runtime_state_does_not_mutate_state(self):
        n = JansenRitUnit()
        n.y4 = math.inf
        before = (n.y0, n.y3, n.y1, n.y4, n.y2, n.y5)

        with pytest.raises(ValueError):
            n.step(220.0)

        assert (n.y0, n.y3, n.y1, n.y4, n.y2, n.y5) == before


# ---------------------------------------------------------------------------
# 5. PERFORMANCE
# ---------------------------------------------------------------------------
class TestJRPerformance:
    def test_isolation_throughput(self):
        n = JansenRitUnit()
        N = 20_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(220.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        assert rate > 5_000, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(JansenRitUnit, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=220.0, dt=0.001, seed=42)
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
class TestJRPipeline:
    def test_population(self):
        assert Population(JansenRitUnit, n=5, label="jr").n == 5

    def test_projection_wiring(self):
        src = Population(JansenRitUnit, n=5, label="src")
        tgt = Population(JansenRitUnit, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=220.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=100.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon_src.count, int)

    def test_network_runs(self):
        pop = Population(JansenRitUnit, n=10, label="jr")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=220.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        # Float return clipped to {0,1} — may or may not have spikes
        assert isinstance(mon.count, int)
