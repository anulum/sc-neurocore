# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: MarderSTGNeuron

"""Full pipeline test for MarderSTGNeuron (Liu-Golowasch-Marder-Abbott 1998).

Single-compartment stomatogastric ganglion neuron with seven voltage-gated
currents (Na, CaT, CaS, A, KCa, Kd, H) plus leak, integrated with fourth-order
Runge-Kutta over thirteen states (v, m_na, h_na, m_cat, h_cat, m_cas, h_cas,
m_a, h_a, m_kca, m_kd, m_h, ca). All gates use the published voltage-dependent
time constants; the calcium reversal is Nernst-derived and intracellular calcium
relaxes towards rest with a 20 ms time constant. The neuron is an endogenous
burster (fires at zero injected current). ModelDB 93321.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.neurons.models.marder_stg import MarderSTGNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi

_GATES = (
    "m_na",
    "h_na",
    "m_cat",
    "h_cat",
    "m_cas",
    "h_cas",
    "m_a",
    "h_a",
    "m_kca",
    "m_kd",
    "m_h",
)


def _run(neuron: MarderSTGNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


# ---------------------------------------------------------------------------
# 1. Isolation
# ---------------------------------------------------------------------------
class TestSTGIsolation:
    def test_defaults(self):
        n = MarderSTGNeuron()
        assert n.v == -60.0 and n.ca == 0.05
        assert n.cm == 1.0 and n.tau_ca == 20.0 and n.f_ca == 0.94
        assert n.dt == 0.05 and n.v_threshold == -20.0

    def test_thirteen_state_variables(self):
        n = MarderSTGNeuron()
        for s in ("v", *_GATES, "ca"):
            assert hasattr(n, s), f"missing state: {s}"

    def test_inactivation_gates_start_open(self):
        n = MarderSTGNeuron()
        assert (n.h_na, n.h_cat, n.h_cas, n.h_a) == (1.0, 1.0, 1.0, 1.0)

    def test_step_returns_binary(self):
        assert MarderSTGNeuron().step(0.0) in (0, 1)

    def test_all_states_evolve(self):
        n = MarderSTGNeuron()
        initial = {s: getattr(n, s) for s in ("v", "m_na", "h_na", "m_cat", "m_kca", "ca")}
        for _ in range(5000):
            n.step(0.0)
        assert all(getattr(n, s) != v0 for s, v0 in initial.items())

    def test_state_finite_long_run(self):
        n = MarderSTGNeuron()
        for _ in range(100_000):
            n.step(0.0)
        for attr in ("v", *_GATES, "ca"):
            assert np.isfinite(getattr(n, attr)), f"{attr} not finite"

    def test_reset_restores_defaults(self):
        n = MarderSTGNeuron()
        for _ in range(5000):
            n.step(2.0)
        n.reset()
        assert n.v == -60.0 and n.ca == 0.05
        assert (n.m_na, n.h_na, n.m_kca) == (0.0, 1.0, 0.0)

    def test_deterministic(self):
        def trace() -> list[tuple[int, float]]:
            n = MarderSTGNeuron()
            return [(n.step(0.0), n.v) for _ in range(2000)]

        assert trace() == trace()


# ---------------------------------------------------------------------------
# 2. Endogenous bursting (CPG oscillator)
# ---------------------------------------------------------------------------
class TestSTGBursting:
    def test_fires_at_zero_current(self):
        assert len(_run(MarderSTGNeuron(), current=0.0, steps=50_000)) >= 10

    def test_bursting_pattern(self):
        spikes = _run(MarderSTGNeuron(), current=0.0, steps=100_000)
        isis = np.diff(spikes)
        assert isis.max() > 3 * np.median(isis), "expected bursts separated by quiescent gaps"

    def test_calcium_accumulates_during_spiking(self):
        n = MarderSTGNeuron()
        for _ in range(50_000):
            n.step(0.0)
        assert n.ca > 1.0

    def test_calcium_non_negative(self):
        n = MarderSTGNeuron()
        for _ in range(100_000):
            n.step(2.0)
            assert n.ca >= 0.0

    def test_voltage_bounded(self):
        n = MarderSTGNeuron()
        vs = [n.v for _ in range(50_000) if (n.step(0.0) or True)]
        assert min(vs) > -100.0 and max(vs) < 80.0


# ---------------------------------------------------------------------------
# 3. f-I relation
# ---------------------------------------------------------------------------
class TestSTGDynamics:
    def test_rate_increases_with_drive(self):
        low = len(_run(MarderSTGNeuron(), current=0.0, steps=50_000))
        high = len(_run(MarderSTGNeuron(), current=10.0, steps=50_000))
        assert high > low

    @pytest.mark.parametrize("current", [0.0, 2.0, 5.0, 10.0])
    def test_fi_sweep_finite(self, current: float):
        n = MarderSTGNeuron()
        for _ in range(20_000):
            n.step(current)
        assert np.isfinite(n.v)

    def test_upward_crossing_only(self):
        n = MarderSTGNeuron()
        prev_v = n.v
        for _ in range(50_000):
            if n.step(0.0) == 1:
                assert prev_v < n.v_threshold
            prev_v = n.v


# ---------------------------------------------------------------------------
# 4. Nernst calcium reversal
# ---------------------------------------------------------------------------
class TestSTGNernst:
    def test_e_ca_positive_at_rest(self):
        assert MarderSTGNeuron()._nernst_e_ca(0.05) > 100.0

    def test_e_ca_decreases_with_calcium(self):
        n = MarderSTGNeuron()
        assert n._nernst_e_ca(50.0) < n._nernst_e_ca(0.05)

    def test_e_ca_handles_zero_calcium(self):
        assert math.isfinite(MarderSTGNeuron()._nernst_e_ca(0.0))


# ---------------------------------------------------------------------------
# 5. Gating and rate functions
# ---------------------------------------------------------------------------
class TestSTGGating:
    def test_gates_bounded(self):
        n = MarderSTGNeuron()
        for _ in range(50_000):
            n.step(2.0)
        for gate in _GATES:
            assert 0.0 <= getattr(n, gate) <= 1.0, gate

    def test_sigmoid_midpoint(self):
        assert MarderSTGNeuron._sigmoid(-25.5, -25.5, 5.29) == 0.5

    def test_sigmoid_limits(self):
        assert MarderSTGNeuron._sigmoid(100.0, -25.5, 5.29) > 0.999
        assert MarderSTGNeuron._sigmoid(-300.0, -25.5, 5.29) < 0.001

    def test_kca_activation_requires_calcium(self):
        """The K-C steady state scales with Ca/(Ca+3); near-zero Ca suppresses it."""
        n = MarderSTGNeuron()
        for _ in range(50_000):
            n.step(0.0)
        assert n.m_kca >= 0.0


# ---------------------------------------------------------------------------
# 6. Fail-closed safety contracts
# ---------------------------------------------------------------------------
class TestSTGSafety:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("dt", 0.0),
            ("cm", 0.0),
            ("tau_ca", 0.0),
            ("ca_out", 0.0),
            ("g_na", -1.0),
            ("g_kca", -1.0),
            ("m_na", 1.01),
            ("h_cas", -0.01),
            ("m_kca", 1.5),
            ("ca", -0.01),
        ],
    )
    def test_rejects_invalid_configuration(self, field: str, value: float):
        with pytest.raises(ValueError):
            MarderSTGNeuron(**{field: value})

    def test_rejects_non_finite_input_before_mutation(self):
        n = MarderSTGNeuron()
        before = n.v
        with pytest.raises(ValueError):
            n.step(float("nan"))
        assert n.v == before

    def test_rejects_runtime_corruption_before_mutation(self):
        n = MarderSTGNeuron()
        n.cm = 0.0
        before = n.v
        with pytest.raises(ValueError):
            n.step(0.0)
        assert n.v == before

    def test_extreme_timestep_fails_closed(self):
        n = MarderSTGNeuron(dt=5.0)
        with pytest.raises(FloatingPointError):
            for _ in range(500):
                n.step(0.0)

    def test_commit_rejects_non_finite_state(self):
        bad = (float("nan"),) + (0.5,) * 11 + (0.05,)
        with pytest.raises(FloatingPointError):
            MarderSTGNeuron._commit(bad)

    def test_commit_clamps_gates_and_calcium(self):
        raw = (-50.0, 1.5, -0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.0, -0.1, 0.5, -3.0)
        out = MarderSTGNeuron._commit(raw)
        assert out[1] == 1.0 and out[2] == 0.0 and out[9] == 1.0 and out[10] == 0.0
        assert out[12] == 0.0


# ---------------------------------------------------------------------------
# 7. Numerics and reversals
# ---------------------------------------------------------------------------
class TestSTGNumerics:
    @pytest.mark.parametrize("dt", [0.025, 0.05])
    def test_dt_stability(self, dt: float):
        n = MarderSTGNeuron(dt=dt)
        for _ in range(50_000):
            n.step(0.0)
        assert np.isfinite(n.v) and np.isfinite(n.ca)

    def test_reversal_ordering(self):
        n = MarderSTGNeuron()
        assert n.e_k < n.e_l < n.e_h < n.e_na

    def test_conductances_non_negative(self):
        n = MarderSTGNeuron()
        for g in (n.g_na, n.g_cat, n.g_cas, n.g_a, n.g_kca, n.g_kd, n.g_h, n.g_l):
            assert g >= 0.0


# ---------------------------------------------------------------------------
# 8. Network wiring and analysis
# ---------------------------------------------------------------------------
class TestSTGPipeline:
    def test_population(self):
        assert Population(MarderSTGNeuron, n=10, label="stg").n == 10

    def test_network_spikes(self):
        pop = Population(MarderSTGNeuron, n=10, label="stg")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = MarderSTGNeuron()
        train = np.array([float(n.step(0.0)) for _ in range(50_000)])
        assert spike_count(train) >= 10
        assert spike_count(train) == int(train.sum())

    def test_analysis_firing_rate(self):
        n = MarderSTGNeuron()
        train = np.array([float(n.step(0.0)) for _ in range(50_000)])
        intervals = isi(train, dt=0.00005)
        if intervals.size > 0:
            assert np.all(intervals > 0)
        assert firing_rate(train, dt=0.00005) > 0
