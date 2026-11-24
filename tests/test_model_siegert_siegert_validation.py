# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSiegertValidation from former test_model_siegert.py

"""Focused suite: TestSiegertValidation from former test_model_siegert.py."""

from __future__ import annotations

from typing import Any
import numpy as np
import pytest
from sc_neurocore.neurons.models.siegert import (
    SiegertTransferFunction,
)


class TestSiegertValidation:
    """Input and state validation of the SiegertTransferFunction."""

    @pytest.mark.parametrize("field", ["v_threshold", "v_reset", "v_rest"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_voltage_parameters(self, field: str, value: float) -> None:
        """The model rejects non-finite voltage parameters."""
        with pytest.raises(ValueError, match=field):
            kwargs: dict[str, Any] = {field: value}
            SiegertTransferFunction(**kwargs)

    @pytest.mark.parametrize("field", ["tau_m", "tau_rp"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf])
    def test_rejects_non_positive_or_non_finite_time_constants(
        self, field: str, value: float
    ) -> None:
        """The model rejects non-positive or non-finite time constants."""
        with pytest.raises(ValueError, match=field):
            kwargs: dict[str, Any] = {field: value}
            SiegertTransferFunction(**kwargs)

    def test_rejects_reset_not_below_threshold(self) -> None:
        """The model rejects reset not below threshold."""
        with pytest.raises(ValueError, match="v_threshold"):
            SiegertTransferFunction(v_reset=-50.0, v_threshold=-50.0)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current(self, current: float) -> None:
        """The model rejects non-finite current."""
        n = SiegertTransferFunction()
        with pytest.raises(ValueError, match="current"):
            n.step(current)

    @pytest.mark.parametrize("field", ["tau_m", "tau_rp", "v_threshold", "v_reset", "v_rest"])
    def test_rejects_corrupted_runtime_parameters(self, field: str) -> None:
        """The model rejects corrupted runtime parameters."""
        n = SiegertTransferFunction()
        setattr(n, field, np.nan)
        with pytest.raises(ValueError, match="runtime"):
            n.step(20.0)

    def test_rejects_corrupted_runtime_boundary_ordering(self) -> None:
        """The model rejects corrupted runtime boundary ordering."""
        n = SiegertTransferFunction()
        n.v_reset = n.v_threshold
        with pytest.raises(ValueError, match="runtime"):
            n.step(20.0)

    def test_rejects_non_finite_diffusion_scale_before_rate_floor(self) -> None:
        """The model rejects non-finite diffusion scale before rate floor."""
        n = SiegertTransferFunction()
        n.v_rest = -np.inf
        with pytest.raises(ValueError, match="runtime"):
            n.step(20.0)

    def test_rate_is_finite_non_negative_and_refractory_bounded(self) -> None:
        """Rate is finite non-negative and refractory bounded."""
        n = SiegertTransferFunction(tau_rp=2.0)
        rates = [n.step(current) for current in [-20.0, 0.0, 20.0, 50.0, 1.0e6]]
        assert all(np.isfinite(rate) for rate in rates)
        assert all(0.0 <= rate <= 500.0 for rate in rates)
