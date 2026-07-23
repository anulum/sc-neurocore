# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMcKeanValidation from former test_model_mckean.py

"""Focused suite: TestMcKeanValidation from former test_model_mckean.py."""

from __future__ import annotations

from tests.model_mckean_support import *  # noqa: F403

class TestMcKeanValidation:
    @pytest.mark.parametrize("field", ["v", "w", "v_peak"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_state_and_threshold(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            McKeanNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["v", "w", "v_peak"])
    @pytest.mark.parametrize("value", [object(), "0.1", True])
    def test_rejects_non_numeric_state_and_threshold(self, field: str, value: object):
        with pytest.raises(TypeError, match=field):
            McKeanNeuron(**{field: value})

    @pytest.mark.parametrize("a", [0.0, -0.1, 1.0, 1.1, np.nan, np.inf, -np.inf])
    def test_rejects_invalid_piecewise_breakpoint_parameter(self, a: float):
        with pytest.raises(ValueError, match="a"):
            McKeanNeuron(a=a)

    @pytest.mark.parametrize("value", [object(), "0.25", True])
    def test_rejects_non_numeric_piecewise_breakpoint_parameter(self, value: object):
        with pytest.raises(TypeError, match="a"):
            McKeanNeuron(a=value)

    @pytest.mark.parametrize("field", ["epsilon", "gamma", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf])
    def test_rejects_non_positive_or_non_finite_scales(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            McKeanNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["epsilon", "gamma", "dt"])
    @pytest.mark.parametrize("value", [object(), "0.1", True])
    def test_rejects_non_numeric_scales(self, field: str, value: object):
        with pytest.raises(TypeError, match=field):
            McKeanNeuron(**{field: value})

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = McKeanNeuron(v=0.25, w=-0.1)
        before = (n.v, n.w)
        with pytest.raises(FloatingPointError, match="current"):
            n.step(current)
        assert (n.v, n.w) == before

    @pytest.mark.parametrize("current", [object(), "0.5", True])
    def test_rejects_non_numeric_current_before_state_mutation(self, current: object):
        n = McKeanNeuron(v=0.25, w=-0.1)
        before = (n.v, n.w)
        with pytest.raises(TypeError, match="current"):
            n.step(current)
        assert (n.v, n.w) == before

    def test_rejects_corrupted_runtime_scale_before_state_mutation(self):
        n = McKeanNeuron(v=0.25, w=-0.1)
        n.dt = 0.0
        before = (n.v, n.w)
        with pytest.raises(ValueError, match="dt"):
            n.step(0.5)
        assert (n.v, n.w) == before

    def test_rejects_corrupted_runtime_breakpoint_before_state_mutation(self):
        n = McKeanNeuron(v=0.25, w=-0.1)
        n.a = 0.0
        before = (n.v, n.w)
        with pytest.raises(ValueError, match="a"):
            n.step(0.5)
        assert (n.v, n.w) == before

    def test_direct_derivative_rejects_non_finite_state(self):
        n = McKeanNeuron()
        with pytest.raises(FloatingPointError, match="state and current"):
            n._derivatives(np.nan, n.w, 0.5)

    def test_direct_derivative_rejects_non_finite_output(self):
        n = McKeanNeuron()
        n.epsilon = np.inf
        with pytest.raises(FloatingPointError, match="derivative"):
            n._derivatives(0.2, -0.1, 0.5)

    def test_direct_candidate_validation_rejects_non_finite_candidate(self):
        with pytest.raises(FloatingPointError, match="candidate"):
            McKeanNeuron._validate_candidate(np.nan, 0.0)
