# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLapicqueValidation from former test_model_lapicque.py

"""Focused suite: TestLapicqueValidation from former test_model_lapicque.py."""

from __future__ import annotations

from tests.model_lapicque_support import *  # noqa: F403


class TestLapicqueValidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("v", np.nan),
            ("v_rest", np.inf),
            ("v_reset", -np.inf),
            ("v_threshold", np.nan),
        ],
    )
    def test_rejects_non_finite_voltage_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            LapicqueNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["tau", "resistance", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_rc_scale_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            LapicqueNeuron(**{field: value})

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"v_threshold": 0.0, "v_rest": 0.0},
            {"v_threshold": -1.0, "v_rest": 0.0},
            {"v_threshold": 0.0, "v_reset": 0.0},
            {"v_threshold": -1.0, "v_reset": 0.0},
        ],
    )
    def test_rejects_invalid_threshold_geometry(self, kwargs):
        with pytest.raises(ValueError, match="v_threshold"):
            LapicqueNeuron(**kwargs)

    def test_rejects_initial_voltage_at_or_above_threshold(self):
        with pytest.raises(ValueError, match="v must be below v_threshold"):
            LapicqueNeuron(v=1.0)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("tau", 0.0, "tau"),
            ("resistance", -1.0, "resistance"),
            ("dt", np.nan, "dt"),
            ("v_threshold", 0.0, "v_threshold"),
            ("v", 1.0, "v must be below v_threshold"),
        ],
    )
    def test_rejects_corrupted_runtime_state_before_integration(
        self, field: str, value: float, message: str
    ):
        n = LapicqueNeuron(v=0.25)
        setattr(n, field, value)
        before = n.v
        with pytest.raises(ValueError, match=message):
            n.step(1.0)
        assert n.v == before

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = LapicqueNeuron(v=0.25)
        before = n.v
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert n.v == before

    def test_rejects_non_finite_voltage_candidate_before_state_mutation(self):
        n = LapicqueNeuron(v=0.25, v_threshold=1.0e308, resistance=1.0e308)
        before = n.v
        with pytest.raises(ValueError, match="voltage candidate"):
            n.step(1.0e308)
        assert n.v == before
