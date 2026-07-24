# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConstructionAndValidation from former test_model_adaptive_threshold_if.py

"""Focused suite: TestConstructionAndValidation from former test_model_adaptive_threshold_if.py."""

from __future__ import annotations

from tests.model_adaptive_threshold_if_support import *  # noqa: F403


class TestConstructionAndValidation:
    """Construction normalises fields and rejects invalid configurations."""

    def test_catalogue_defaults(self) -> None:
        n = AdaptiveThresholdIFNeuron()
        assert (n.v, n.theta) == (-65.0, -50.0)
        assert (n.v_rest, n.v_reset, n.theta_rest) == (-65.0, -65.0, -50.0)
        assert (n.delta_theta, n.tau_m, n.tau_theta, n.dt) == (5.0, 10.0, 50.0, 0.1)

    def test_scalar_fields_are_normalised_to_float(self) -> None:
        n = AdaptiveThresholdIFNeuron(v=-60, tau_m=8)  # type: ignore[arg-type]
        assert isinstance(n.v, float) and isinstance(n.tau_m, float)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"v": float("nan")},
            {"theta": float("inf")},
            {"v_rest": float("nan")},
            {"v_reset": float("inf")},
            {"theta_rest": float("nan")},
            {"theta_rest": -70.0},
            {"v_reset": -45.0},
            {"delta_theta": -0.1},
            {"delta_theta": float("inf")},
            {"tau_m": 0.0},
            {"tau_m": float("nan")},
            {"tau_theta": 0.0},
            {"tau_theta": float("inf")},
            {"dt": 0.0},
            {"dt": float("nan")},
        ],
    )
    def test_rejects_non_physical_configuration(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            AdaptiveThresholdIFNeuron(**kwargs)

    @pytest.mark.parametrize("field", ["v", "theta", "tau_m"])
    def test_rejects_non_numeric_fields(self, field: str) -> None:
        with pytest.raises(ValueError, match="must be numeric"):
            AdaptiveThresholdIFNeuron(**{field: "fast"})  # type: ignore[arg-type]
