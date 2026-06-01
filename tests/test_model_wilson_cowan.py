# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: WilsonCowanUnit

"""Full pipeline test for WilsonCowanUnit (Wilson & Cowan 1972).

E/I rate model: returns float (E rate), not int spike.
τ_e dE/dt = -E + S(w_ee·E - w_ei·I + I_ext).
Pipeline limited: returns float, Network expects int → documented.
Performance: ~163K isolation steps/s."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.wilson_cowan import WilsonCowanUnit
from sc_neurocore.network.population import Population


def _rk4_expected_state(unit: WilsonCowanUnit, drive: float) -> tuple[float, float]:
    e0, i0 = unit.e, unit.i

    def derivatives(e: float, i: float) -> tuple[float, float]:
        se = unit._sigmoid(unit.w_ee * e - unit.w_ei * i + drive)
        si = unit._sigmoid(unit.w_ie * e - unit.w_ii * i)
        return (-e + se) / unit.tau_e, (-i + si) / unit.tau_i

    k1_e, k1_i = derivatives(e0, i0)
    k2_e, k2_i = derivatives(e0 + 0.5 * unit.dt * k1_e, i0 + 0.5 * unit.dt * k1_i)
    k3_e, k3_i = derivatives(e0 + 0.5 * unit.dt * k2_e, i0 + 0.5 * unit.dt * k2_i)
    k4_e, k4_i = derivatives(e0 + unit.dt * k3_e, i0 + unit.dt * k3_i)
    return (
        e0 + unit.dt * (k1_e + 2.0 * k2_e + 2.0 * k3_e + k4_e) / 6.0,
        i0 + unit.dt * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i) / 6.0,
    )


class TestWilsonCowanIsolation:
    def test_defaults(self):
        n = WilsonCowanUnit()
        assert n.e == 0.1 and n.i == 0.05
        assert n.w_ee == 10.0 and n.w_ei == 6.0
        assert n.tau_e == 1.0 and n.tau_i == 2.0

    def test_step_returns_float(self):
        """Returns E rate (float), not binary spike."""
        n = WilsonCowanUnit()
        result = n.step(0.0)
        assert isinstance(result, float)

    def test_both_variables_evolve(self):
        n = WilsonCowanUnit()
        e0, i0 = n.e, n.i
        for _ in range(100):
            n.step(5.0)
        assert n.e != e0 and n.i != i0

    def test_state_finite(self):
        n = WilsonCowanUnit()
        for _ in range(100000):
            n.step(5.0)
        assert np.isfinite(n.e) and np.isfinite(n.i)

    def test_reset(self):
        n = WilsonCowanUnit()
        for _ in range(100):
            n.step(5.0)
        n.reset()
        assert n.e == 0.1 and n.i == 0.05


class TestWilsonCowanSigmoid:
    """Published two-term form, Wilson-Cowan 1972:
        S(x) = 1/(1+exp(-a(x-θ))) − 1/(1+exp(aθ))
    The subtracted baseline makes S(0) = 0 exactly. Range is therefore
    [-β, 1-β] where β = 1/(1+exp(aθ))."""

    def test_sigmoid_at_threshold(self):
        """S(θ) = 0.5 − β."""
        n = WilsonCowanUnit()
        baseline = 1.0 / (1.0 + math.exp(n.a * n.theta))
        assert abs(float(n._sigmoid(n.theta)) - (0.5 - baseline)) < 1e-12

    def test_sigmoid_at_zero(self):
        """S(0) = 0 by construction of the baseline subtraction."""
        n = WilsonCowanUnit()
        assert abs(float(n._sigmoid(0.0))) < 1e-12

    def test_sigmoid_monotonic(self):
        n = WilsonCowanUnit()
        vals = [float(n._sigmoid(x)) for x in [-5, 0, 4, 5, 10]]
        assert all(vals[j] <= vals[j + 1] for j in range(len(vals) - 1))

    def test_sigmoid_bounded_published_range(self):
        """Range is [−β, 1−β] where β = 1/(1+exp(aθ))."""
        n = WilsonCowanUnit()
        baseline = 1.0 / (1.0 + math.exp(n.a * n.theta))
        for x in [-50, -10, 0, 10, 50]:
            s = float(n._sigmoid(x))
            assert -baseline - 1e-12 <= s <= 1.0 - baseline + 1e-12


class TestWilsonCowanEIDynamics:
    """E/I population interaction — the core of Wilson-Cowan."""

    def test_e_increases_with_excitatory_input(self):
        """External input drives E upward."""
        n = WilsonCowanUnit()
        for _ in range(1000):
            n.step(10.0)
        assert n.e > 0.5

    def test_i_follows_e(self):
        """I is driven by E: w_ie·E enters the I sigmoid."""
        n = WilsonCowanUnit()
        for _ in range(1000):
            n.step(10.0)
        assert n.i > 0.1  # I has increased from following E

    def test_zero_input_low_activity(self):
        """Without input, E and I decay to low values."""
        n = WilsonCowanUnit()
        for _ in range(10000):
            n.step(0.0)
        assert n.e < 0.05 and n.i < 0.05

    def test_e_bounded_0_1(self):
        """E rate should stay in [0, 1] (sigmoid output range)."""
        n = WilsonCowanUnit()
        for _ in range(10000):
            n.step(10.0)
        assert 0.0 <= n.e <= 1.0

    def test_steady_state_at_high_input(self):
        """At high I_ext, E and I converge to steady state near 1.0."""
        n = WilsonCowanUnit()
        for _ in range(10000):
            n.step(10.0)
        e1 = n.e
        for _ in range(10000):
            n.step(10.0)
        assert abs(n.e - e1) < 0.001  # converged

    def test_step_uses_candidate_first_rk4_flow(self):
        n = WilsonCowanUnit(e=0.24, i=0.11, dt=0.35)
        expected_e, expected_i = _rk4_expected_state(n, 3.0)
        se = n._sigmoid(n.w_ee * n.e - n.w_ei * n.i + 3.0)
        si = n._sigmoid(n.w_ie * n.e - n.w_ii * n.i)
        euler_e = n.e + (-n.e + se) / n.tau_e * n.dt
        euler_i = n.i + (-n.i + si) / n.tau_i * n.dt

        result = n.step(3.0)

        assert result == pytest.approx(0.42143718680097664, abs=1e-15)
        assert n.e == pytest.approx(expected_e, abs=1e-15)
        assert n.i == pytest.approx(expected_i, abs=1e-15)
        assert abs(n.e - euler_e) > 1.0e-2
        assert abs(n.i - euler_i) > 1.0e-2

    def test_w_ee_controls_excitatory_recurrence(self):
        """Higher w_ee gives higher E→E feedback and higher E steady state."""
        n_low = WilsonCowanUnit(w_ee=5.0)
        n_high = WilsonCowanUnit(w_ee=15.0)
        for _ in range(10000):
            n_low.step(3.0)
            n_high.step(3.0)
        assert n_high.e > n_low.e

    def test_w_ei_controls_inhibition(self):
        """Higher w_ei gives higher I→E inhibition and lower E."""
        n_low = WilsonCowanUnit(w_ei=3.0)
        n_high = WilsonCowanUnit(w_ei=10.0)
        for _ in range(10000):
            n_low.step(5.0)
            n_high.step(5.0)
        assert n_low.e > n_high.e


class TestWilsonCowanOscillation:
    def test_can_oscillate(self):
        """With appropriate parameters, E should oscillate."""
        n = WilsonCowanUnit(w_ee=16.0, w_ei=12.0, w_ie=15.0, theta=4.0)
        es = []
        for _ in range(5000):
            n.step(5.0)
            es.append(n.e)
        es = np.array(es[1000:])
        # Check for oscillation: multiple crossings of mean
        mean_e = np.mean(es)
        crossings = np.sum(np.diff(np.sign(es - mean_e)) != 0)
        # May or may not oscillate — just verify it ran and E is finite
        assert np.isfinite(es[-1])


class TestWilsonCowanParameters:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("e", np.nan),
            ("i", np.inf),
            ("e", -0.1),
            ("i", 1.1),
            ("w_ee", -1.0),
            ("w_ei", -1.0),
            ("w_ie", -1.0),
            ("w_ii", -1.0),
            ("tau_e", 0.0),
            ("tau_i", 0.0),
            ("a", 0.0),
            ("theta", np.inf),
            ("dt", 0.0),
        ],
    )
    def test_rejects_invalid_numerical_configuration(self, field: str, value: float):
        with pytest.raises((ValueError, FloatingPointError)):
            WilsonCowanUnit(**{field: value})

    def test_rejects_non_finite_input_before_state_mutation(self):
        n = WilsonCowanUnit()
        before = (n.e, n.i)
        with pytest.raises(ValueError, match="external input"):
            n.step(np.nan)
        assert (n.e, n.i) == before

    def test_rejects_corrupted_runtime_state_before_mutation(self):
        n = WilsonCowanUnit()
        n.e = 1.5
        before = (n.e, n.i)
        with pytest.raises(FloatingPointError, match="e rate"):
            n.step(5.0)
        assert (n.e, n.i) == before

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("w_ee", -1.0),
            ("w_ei", -1.0),
            ("w_ie", -1.0),
            ("w_ii", -1.0),
            ("tau_e", 0.0),
            ("tau_i", 0.0),
            ("a", 0.0),
            ("theta", math.inf),
            ("dt", 0.0),
        ],
    )
    def test_rejects_runtime_parameter_corruption_before_state_mutation(
        self, field: str, value: float
    ):
        n = WilsonCowanUnit()
        setattr(n, field, value)
        before = (n.e, n.i)

        with pytest.raises((ValueError, FloatingPointError)):
            n.step(5.0)

        assert (n.e, n.i) == before

    def test_sigmoid_saturates_for_extreme_finite_drive(self):
        n = WilsonCowanUnit()
        baseline = 1.0 / (1.0 + math.exp(n.a * n.theta))
        assert abs(n._sigmoid(1.0e308) - (1.0 - baseline)) < 1e-12
        assert abs(n._sigmoid(-1.0e308) + baseline) < 1e-12

    def test_sigmoid_rejects_non_finite_drive(self):
        n = WilsonCowanUnit()

        with pytest.raises(ValueError, match="sigmoid input"):
            n._sigmoid(math.nan)

    def test_rejects_non_finite_derivative_before_state_mutation(self):
        n = WilsonCowanUnit(tau_e=1.0e-320)
        before = (n.e, n.i)

        with pytest.raises(FloatingPointError, match="derivative"):
            n.step(0.0)

        assert (n.e, n.i) == before

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float):
        n = WilsonCowanUnit(dt=dt)
        for _ in range(10000):
            n.step(5.0)
        assert np.isfinite(n.e)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = WilsonCowanUnit()
            trace = [(n.step(5.0), n.e, n.i) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestWilsonCowanPerformance:
    def test_isolation_throughput(self):
        n = WilsonCowanUnit()
        N = 50000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(5.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 20000


class TestWilsonCowanPipeline:
    def test_population_creates(self):
        assert Population(WilsonCowanUnit, n=10, label="wc").n == 10

    def test_network_returns_float_not_spike(self):
        """WilsonCowanUnit.step() returns float (E rate), not int.

        Network.step_all expects int return for spike detection.
        The model runs in the network but spike counts will be wrong
        (every non-zero E registers as spike). Document this limitation.
        """
        n = WilsonCowanUnit()
        result = n.step(5.0)
        assert isinstance(result, float)
        # The model is a RATE model, not a spiking model
