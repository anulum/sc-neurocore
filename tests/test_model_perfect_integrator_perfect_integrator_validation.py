# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPerfectIntegratorValidation from former test_model_perfect_integrator.py

"""Focused suite: TestPerfectIntegratorValidation from former test_model_perfect_integrator.py."""

from __future__ import annotations

from tests.model_perfect_integrator_support import *  # noqa: F403


class TestPerfectIntegratorValidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("v", np.nan),
            ("v_threshold", np.inf),
            ("v_reset", -np.inf),
        ],
    )
    def test_rejects_non_finite_voltage_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            PerfectIntegratorNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["c_m", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_scale_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            PerfectIntegratorNeuron(**{field: value})

    @pytest.mark.parametrize(
        ("v_threshold", "v_reset"),
        [
            (0.0, 0.0),
            (-1.0, 0.0),
        ],
    )
    def test_rejects_non_positive_threshold_excursion(self, v_threshold: float, v_reset: float):
        with pytest.raises(ValueError, match="v_threshold"):
            PerfectIntegratorNeuron(v_threshold=v_threshold, v_reset=v_reset)

    def test_rejects_initial_voltage_at_or_above_threshold(self):
        with pytest.raises(ValueError, match="v must be below v_threshold"):
            PerfectIntegratorNeuron(v=1.0)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = PerfectIntegratorNeuron(v=0.25)
        before = n.v
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert n.v == before

    def test_rejects_non_finite_voltage_increment_before_state_mutation(self):
        n = PerfectIntegratorNeuron(v=0.25, v_threshold=1.0e308, c_m=1.0e-308)
        before = n.v
        with pytest.raises(ValueError, match="voltage increment"):
            n.step(1.0e308)
        assert n.v == before

    @pytest.mark.parametrize("field", ["v", "c_m", "dt", "v_threshold", "v_reset"])
    def test_rejects_corrupted_runtime_state_before_voltage_mutation(self, field: str):
        n = PerfectIntegratorNeuron(v=0.25)
        before = n.v
        setattr(n, field, np.nan)
        with pytest.raises(ValueError, match="runtime"):
            n.step(1.0)
        if field != "v":
            assert n.v == before

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("c_m", 0.0),
            ("dt", 0.0),
            ("v_threshold", 0.0),
        ],
    )
    def test_rejects_invalid_runtime_geometry_before_voltage_mutation(
        self, field: str, value: float
    ):
        n = PerfectIntegratorNeuron(v=0.25)
        before = n.v
        setattr(n, field, value)
        with pytest.raises(ValueError, match="runtime"):
            n.step(1.0)
        assert n.v == before
