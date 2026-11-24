# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestILIFValidation from former test_model_ilif.py

"""Focused suite: TestILIFValidation from former test_model_ilif.py."""

from __future__ import annotations

from typing import Any
import numpy as np
import pytest
from sc_neurocore.neurons.models.ilif import (
    InhibitoryLIFNeuron,
)


class TestILIFValidation:
    """Input and state validation of the InhibitoryLIFNeuron."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("v", np.nan),
            ("v_reset", np.inf),
        ],
    )
    def test_rejects_non_finite_voltage_state_or_reset(self, field: str, value: float) -> None:
        """The model rejects non-finite voltage state or reset."""
        with pytest.raises(ValueError, match=field):
            kwargs: dict[str, Any] = {field: value}
            InhibitoryLIFNeuron(**kwargs)

    @pytest.mark.parametrize("field", ["inh_trace", "inh_strength"])
    @pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
    def test_rejects_negative_or_non_finite_inhibitory_terms(
        self, field: str, value: float
    ) -> None:
        """The model rejects negative or non-finite inhibitory terms."""
        with pytest.raises(ValueError, match=field):
            kwargs: dict[str, Any] = {field: value}
            InhibitoryLIFNeuron(**kwargs)

    @pytest.mark.parametrize("field", ["tau_m", "tau_inh", "v_threshold", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_scale_parameters(
        self, field: str, value: float
    ) -> None:
        """The model rejects non-positive or non-finite scale parameters."""
        with pytest.raises(ValueError, match=field):
            kwargs: dict[str, Any] = {field: value}
            InhibitoryLIFNeuron(**kwargs)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float) -> None:
        """A non-finite current is rejected before any state change."""
        n = InhibitoryLIFNeuron(v=0.25, inh_trace=0.5)
        before = (n.v, n.inh_trace)
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert (n.v, n.inh_trace) == before

    def test_rejects_corrupted_negative_trace_before_state_mutation(self) -> None:
        """The model rejects corrupted negative trace before state mutation."""
        n = InhibitoryLIFNeuron(v=0.25, inh_trace=0.5)
        n.inh_trace = -1.0
        before = (n.v, n.inh_trace)
        with pytest.raises(ValueError, match="inh_trace"):
            n.step(1.0)
        assert (n.v, n.inh_trace) == before
