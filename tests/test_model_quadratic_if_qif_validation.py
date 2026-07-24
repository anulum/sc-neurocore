# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQIFValidation from former test_model_quadratic_if.py

"""Focused suite: TestQIFValidation from former test_model_quadratic_if.py."""

from __future__ import annotations

from tests.model_quadratic_if_support import *  # noqa: F403


class TestQIFValidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("v", np.nan),
            ("v_reset", np.inf),
            ("v_peak", -np.inf),
        ],
    )
    def test_rejects_non_finite_voltage_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            QuadraticIFNeuron(**{field: value})

    @pytest.mark.parametrize("dt", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_dt(self, dt: float):
        with pytest.raises(ValueError, match="dt"):
            QuadraticIFNeuron(dt=dt)

    @pytest.mark.parametrize(
        ("v_reset", "v_peak"),
        [
            (1.0, 1.0),
            (2.0, 1.0),
        ],
    )
    def test_rejects_reset_not_below_peak(self, v_reset: float, v_peak: float):
        with pytest.raises(ValueError, match="v_peak"):
            QuadraticIFNeuron(v_reset=v_reset, v_peak=v_peak)

    def test_rejects_initial_voltage_at_or_above_peak(self):
        with pytest.raises(ValueError, match="v must be below v_peak"):
            QuadraticIFNeuron(v=1.0)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = QuadraticIFNeuron(v=-0.25)
        before = n.v
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert n.v == before

    def test_rejects_non_finite_exact_flow_before_state_mutation(self):
        n = QuadraticIFNeuron(v=-0.25)
        before = n.v
        with pytest.raises(ValueError, match="exact-flow"):
            n.step(-1.0e308)
        assert n.v == before

    def test_negative_current_fixed_point_is_preserved(self):
        n = QuadraticIFNeuron(v=-1.0)
        spike = n.step(-1.0)
        assert spike == 0
        assert n.v == -1.0
