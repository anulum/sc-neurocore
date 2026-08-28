# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wilson-HR parameter and failure-boundary contracts

"""Verify strict Wilson-HR parameter validation and failure atomicity."""

from __future__ import annotations

from typing import cast

from tests.model_wilson_hr_support import *


def _construct_with_field(field: str, value: object) -> WilsonHRNeuron:
    """Construct through a named field while retaining deliberate invalid values."""
    numeric = cast(float, value)
    if field == "v":
        return WilsonHRNeuron(v=numeric)
    if field == "r":
        return WilsonHRNeuron(r=numeric)
    if field == "v_peak":
        return WilsonHRNeuron(v_peak=numeric)
    if field == "capacitance":
        return WilsonHRNeuron(capacitance=numeric)
    if field == "tau_r":
        return WilsonHRNeuron(tau_r=numeric)
    if field == "dt":
        return WilsonHRNeuron(dt=numeric)
    raise AssertionError(f"unknown Wilson-HR field {field!r}")


class TestWilsonHRParameters:
    @pytest.mark.parametrize("field", ["v", "r", "v_peak"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_state_and_threshold(self, field: str, value: float) -> None:
        with pytest.raises(ValueError, match=field):
            _construct_with_field(field, value)

    @pytest.mark.parametrize("field", ["v", "r", "v_peak"])
    @pytest.mark.parametrize("value", [object(), "0.1", True])
    def test_rejects_non_numeric_state_and_threshold(self, field: str, value: object) -> None:
        with pytest.raises(TypeError, match=field):
            _construct_with_field(field, value)

    @pytest.mark.parametrize("field", ["capacitance", "tau_r", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf])
    def test_rejects_non_positive_or_non_finite_scales(self, field: str, value: float) -> None:
        with pytest.raises(ValueError, match=field):
            _construct_with_field(field, value)

    @pytest.mark.parametrize("field", ["capacitance", "tau_r", "dt"])
    @pytest.mark.parametrize("value", [object(), "0.1", True])
    def test_rejects_non_numeric_scales(self, field: str, value: object) -> None:
        with pytest.raises(TypeError, match=field):
            _construct_with_field(field, value)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float) -> None:
        n = WilsonHRNeuron()
        before = (n.v, n.r)
        with pytest.raises(FloatingPointError, match="current"):
            n.step(current)
        assert (n.v, n.r) == before

    @pytest.mark.parametrize("current", [object(), "0.3", True])
    def test_rejects_non_numeric_current_before_state_mutation(self, current: object) -> None:
        n = WilsonHRNeuron()
        before = (n.v, n.r)
        with pytest.raises(TypeError, match="current"):
            n.step(cast(float, current))
        assert (n.v, n.r) == before

    def test_rejects_corrupted_runtime_state_before_mutation(self) -> None:
        n = WilsonHRNeuron()
        n.r = np.inf
        before = (n.v, n.r)
        with pytest.raises(FloatingPointError, match="r must be finite"):
            n.step(0.3)
        assert (n.v, n.r) == before

    def test_rejects_corrupted_runtime_scale_before_mutation(self) -> None:
        n = WilsonHRNeuron()
        n.tau_r = 0.0
        before = (n.v, n.r)
        with pytest.raises(ValueError, match="tau_r"):
            n.step(0.3)
        assert (n.v, n.r) == before

    def test_rejects_polynomial_overflow_before_state_mutation(self) -> None:
        n = WilsonHRNeuron(v=1.0e308)
        before = (n.v, n.r)
        with pytest.raises(FloatingPointError, match="polynomial|candidate|derivative"):
            n.step(0.3)
        assert (n.v, n.r) == before

    def test_direct_derivative_rejects_non_finite_state(self) -> None:
        n = WilsonHRNeuron()
        with pytest.raises(FloatingPointError, match="state and current"):
            n._derivatives(np.nan, n.r, 0.3)

    def test_direct_derivative_rejects_non_finite_output(self) -> None:
        n = WilsonHRNeuron()
        with pytest.raises(FloatingPointError, match="derivative"):
            n._derivatives(0.0, 1.0e308, 0.3)

    def test_direct_candidate_validation_rejects_non_finite_candidate(self) -> None:
        with pytest.raises(FloatingPointError, match="candidate"):
            WilsonHRNeuron._validate_candidate(np.nan, 0.0)

    def test_tau_r_affects_recovery(self) -> None:
        n_fast = WilsonHRNeuron(tau_r=1.0)
        n_slow = WilsonHRNeuron(tau_r=5.0)
        s_fast = len(_run(n_fast, current=0.3, steps=50_000))
        s_slow = len(_run(n_slow, current=0.3, steps=50_000))
        assert s_fast != s_slow

    def test_capacitance_controls_membrane_rate(self) -> None:
        fast = WilsonHRNeuron(capacitance=0.4)
        slow = WilsonHRNeuron(capacitance=1.6)
        fast.step(0.1)
        slow.step(0.1)
        assert abs(fast.v + 0.7) > abs(slow.v + 0.7)

    def test_v_peak_controls_threshold(self) -> None:
        n_low = WilsonHRNeuron(v_peak=-0.1)
        n_high = WilsonHRNeuron(v_peak=0.2)
        s_low = len(_run(n_low, current=0.1, steps=50_000))
        s_high = len(_run(n_high, current=0.1, steps=50_000))
        assert s_low >= s_high

    @pytest.mark.parametrize("dt", [0.02, 0.05, 0.1])
    def test_dt_stability(self, dt: float) -> None:
        n = WilsonHRNeuron(dt=dt)
        for _ in range(50_000):
            n.step(0.3)
        assert np.isfinite(n.v)

    def test_deterministic(self) -> None:
        traces = []
        for _ in range(2):
            n = WilsonHRNeuron()
            trace = [(n.step(0.3), n.v, n.r) for _ in range(300)]
            traces.append(trace)
        assert traces[0] == traces[1]
