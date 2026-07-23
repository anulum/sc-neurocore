# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPernarowskiValidation from former test_model_pernarowski.py

"""Focused suite: TestPernarowskiValidation from former test_model_pernarowski.py."""

from __future__ import annotations

from tests.model_pernarowski_support import *  # noqa: F403

class TestPernarowskiValidation:
    @pytest.mark.parametrize("field", ["v", "w", "z", "alpha", "beta", "v_threshold"])
    def test_rejects_non_numeric_state_offsets_and_threshold(self, field: str):
        with pytest.raises(TypeError, match=field):
            PernarowskiNeuron(**{field: object()})

    @pytest.mark.parametrize("field", ["v", "w", "z", "alpha", "beta", "v_threshold"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_state_offsets_and_threshold(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            PernarowskiNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["eps1", "eps2", "gamma", "dt"])
    def test_rejects_non_numeric_scales(self, field: str):
        with pytest.raises(TypeError, match=field):
            PernarowskiNeuron(**{field: object()})

    @pytest.mark.parametrize("field", ["eps1", "eps2", "gamma", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf])
    def test_rejects_non_positive_or_non_finite_scales(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            PernarowskiNeuron(**{field: value})

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = PernarowskiNeuron(v=-0.5, w=0.1, z=-0.2)
        before = (n.v, n.w, n.z)
        with pytest.raises(FloatingPointError, match="current"):
            n.step(current)
        assert (n.v, n.w, n.z) == before

    def test_rejects_non_numeric_runtime_current_before_mutation(self):
        n = PernarowskiNeuron(v=-0.5, w=0.1, z=-0.2)
        before = (n.v, n.w, n.z)
        with pytest.raises(TypeError, match="current"):
            n.step(object())
        assert (n.v, n.w, n.z) == before

    def test_rejects_corrupted_positive_runtime_scale_before_mutation(self):
        n = PernarowskiNeuron(v=-0.5, w=0.1, z=-0.2)
        n.eps1 = 0.0
        before = (n.v, n.w, n.z)
        with pytest.raises(ValueError, match="eps1"):
            n.step(0.5)
        assert (n.v, n.w, n.z) == before

    def test_rejects_corrupted_runtime_state_before_mutation(self):
        n = PernarowskiNeuron(v=-0.5, w=0.1, z=-0.2)
        n.w = math.nan
        before = (n.v, n.w, n.z)
        with pytest.raises(FloatingPointError, match="w"):
            n.step(0.5)
        assert n.v == before[0]
        assert math.isnan(n.w)
        assert n.z == before[2]

    def test_rejects_nonfinite_derivative_without_mutation(self):
        n = PernarowskiNeuron(v=1.0e160, w=0.1, z=-0.2)
        before = (n.v, n.w, n.z)
        with pytest.raises(FloatingPointError, match="derivative"):
            n.step(0.5)
        assert (n.v, n.w, n.z) == before

    def test_derivative_rejects_nonfinite_runtime_inputs(self):
        n = PernarowskiNeuron()
        with pytest.raises(FloatingPointError, match="state and current must be finite"):
            n._derivatives(math.nan, n.w, n.z, 0.5)

    def test_derivative_rejects_nonfinite_output(self):
        n = PernarowskiNeuron()
        n.eps1 = math.inf
        with pytest.raises(FloatingPointError, match="derivative"):
            n._derivatives(n.v, n.w, n.z, 0.5)

    def test_rejects_nonfinite_candidate_directly(self):
        with pytest.raises(FloatingPointError, match="candidate"):
            PernarowskiNeuron._validate_candidate(math.nan, 0.0, 0.0)
