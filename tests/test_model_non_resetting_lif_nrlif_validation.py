# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNRLIFValidation from former test_model_non_resetting_lif.py

"""Focused suite: TestNRLIFValidation from former test_model_non_resetting_lif.py."""

from __future__ import annotations

from tests.model_non_resetting_lif_support import *  # noqa: F403


class TestNRLIFValidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("v", np.nan),
            ("theta", np.inf),
            ("v_rest", -np.inf),
            ("theta_rest", np.nan),
        ],
    )
    def test_rejects_non_finite_voltage_or_threshold_state(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            NonResettingLIFNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["delta_theta", "r_m"])
    @pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
    def test_rejects_negative_or_non_finite_non_negative_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            NonResettingLIFNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["tau_m", "tau_theta", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_scale_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            NonResettingLIFNeuron(**{field: value})

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = NonResettingLIFNeuron(v=-60.0, theta=-45.0)
        before = (n.v, n.theta)
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert (n.v, n.theta) == before

    @pytest.mark.parametrize(
        "field",
        [
            "v",
            "theta",
            "v_rest",
            "theta_rest",
            "delta_theta",
            "tau_m",
            "tau_theta",
            "r_m",
            "dt",
        ],
    )
    def test_rejects_corrupted_runtime_state_before_mutation(self, field: str):
        n = NonResettingLIFNeuron(v=-60.0, theta=-45.0)
        before = (n.v, n.theta)
        setattr(n, field, np.nan)
        with pytest.raises(ValueError, match="runtime"):
            n.step(20.0)
        if field not in {"v", "theta"}:
            assert (n.v, n.theta) == before

    @pytest.mark.parametrize("field", ["tau_m", "tau_theta", "dt"])
    def test_rejects_non_positive_runtime_time_constants_before_mutation(self, field: str):
        n = NonResettingLIFNeuron(v=-60.0, theta=-45.0)
        before = (n.v, n.theta)
        setattr(n, field, 0.0)
        with pytest.raises(ValueError, match="runtime"):
            n.step(20.0)
        assert (n.v, n.theta) == before

    def test_rejects_non_finite_membrane_candidate_before_mutation(self):
        n = NonResettingLIFNeuron(v=-60.0, theta=-45.0, r_m=10.0)
        before = (n.v, n.theta)
        with pytest.raises(ValueError, match="exact relaxation"):
            n.step(1.0e308)
        assert (n.v, n.theta) == before

    def test_rejects_non_finite_threshold_candidate_before_mutation(self):
        n = NonResettingLIFNeuron(v=1.0e308, theta=9.0e307, theta_rest=9.0e307, delta_theta=9.0e307)
        before = (n.v, n.theta)
        with pytest.raises(ValueError, match="exact relaxation"):
            n.step(0.0)
        assert (n.v, n.theta) == before
