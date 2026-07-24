# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWilsonHRParameters from former test_model_wilson_hr.py

"""Focused suite: TestWilsonHRParameters from former test_model_wilson_hr.py."""

from __future__ import annotations

from tests.model_wilson_hr_support import *  # noqa: F403


class TestWilsonHRParameters:
    @pytest.mark.parametrize("field", ["v", "r", "v_peak"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_state_and_threshold(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            WilsonHRNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["v", "r", "v_peak"])
    @pytest.mark.parametrize("value", [object(), "0.1", True])
    def test_rejects_non_numeric_state_and_threshold(self, field: str, value: object):
        with pytest.raises(TypeError, match=field):
            WilsonHRNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["tau_r", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf])
    def test_rejects_non_positive_or_non_finite_scales(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            WilsonHRNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["tau_r", "dt"])
    @pytest.mark.parametrize("value", [object(), "0.1", True])
    def test_rejects_non_numeric_scales(self, field: str, value: object):
        with pytest.raises(TypeError, match=field):
            WilsonHRNeuron(**{field: value})

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = WilsonHRNeuron()
        before = (n.v, n.r)
        with pytest.raises(FloatingPointError, match="current"):
            n.step(current)
        assert (n.v, n.r) == before

    @pytest.mark.parametrize("current", [object(), "0.3", True])
    def test_rejects_non_numeric_current_before_state_mutation(self, current: object):
        n = WilsonHRNeuron()
        before = (n.v, n.r)
        with pytest.raises(TypeError, match="current"):
            n.step(current)
        assert (n.v, n.r) == before

    def test_rejects_corrupted_runtime_state_before_mutation(self):
        n = WilsonHRNeuron()
        n.r = np.inf
        before = (n.v, n.r)
        with pytest.raises(FloatingPointError, match="r must be finite"):
            n.step(0.3)
        assert (n.v, n.r) == before

    def test_rejects_corrupted_runtime_scale_before_mutation(self):
        n = WilsonHRNeuron()
        n.tau_r = 0.0
        before = (n.v, n.r)
        with pytest.raises(ValueError, match="tau_r"):
            n.step(0.3)
        assert (n.v, n.r) == before

    def test_rejects_polynomial_overflow_before_state_mutation(self):
        n = WilsonHRNeuron(v=1.0e308)
        before = (n.v, n.r)
        with pytest.raises(FloatingPointError, match="polynomial|candidate|derivative"):
            n.step(0.3)
        assert (n.v, n.r) == before

    def test_direct_derivative_rejects_non_finite_state(self):
        n = WilsonHRNeuron()
        with pytest.raises(FloatingPointError, match="state and current"):
            n._derivatives(np.nan, n.r, 0.3)

    def test_direct_derivative_rejects_non_finite_output(self):
        n = WilsonHRNeuron()
        with pytest.raises(FloatingPointError, match="derivative"):
            n._derivatives(0.0, 1.0e308, 0.3)

    def test_direct_candidate_validation_rejects_non_finite_candidate(self):
        with pytest.raises(FloatingPointError, match="candidate"):
            WilsonHRNeuron._validate_candidate(np.nan, 0.0)

    def test_tau_r_affects_recovery(self):
        n_fast = WilsonHRNeuron(tau_r=1.0)
        n_slow = WilsonHRNeuron(tau_r=5.0)
        s_fast = len(_run(n_fast, current=0.3, steps=50_000))
        s_slow = len(_run(n_slow, current=0.3, steps=50_000))
        assert s_fast != s_slow

    def test_v_peak_controls_threshold(self):
        n_low = WilsonHRNeuron(v_peak=0.2)
        n_high = WilsonHRNeuron(v_peak=0.6)
        s_low = len(_run(n_low, current=0.3, steps=50_000))
        s_high = len(_run(n_high, current=0.3, steps=50_000))
        assert s_low >= s_high

    @pytest.mark.parametrize("dt", [0.02, 0.05, 0.1])
    def test_dt_stability(self, dt: float):
        n = WilsonHRNeuron(dt=dt)
        for _ in range(50_000):
            n.step(0.3)
        assert np.isfinite(n.v)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = WilsonHRNeuron()
            trace = [(n.step(0.3), n.v, n.r) for _ in range(300)]
            traces.append(trace)
        assert traces[0] == traces[1]
