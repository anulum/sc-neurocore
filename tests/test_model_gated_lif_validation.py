# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestValidation from former test_model_gated_lif.py

"""Focused suite: TestValidation from former test_model_gated_lif.py."""

from __future__ import annotations

from typing import Any
import numpy as np
import pytest
from sc_neurocore.neurons.models.gated_lif import (
    GatedLIFNeuron,
)


class TestValidation:
    """Input and state validation of the GatedLIFNeuron."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("v", np.nan),
            ("gate_i", np.inf),
        ],
    )
    def test_rejects_non_finite_state_or_input_gate(self, field: str, value: float) -> None:
        """The model rejects non-finite state or input gate."""
        with pytest.raises(ValueError, match=field):
            kwargs: dict[str, Any] = {field: value}
            GatedLIFNeuron(**kwargs)

    @pytest.mark.parametrize("gate_v", [-0.1, 1.1, np.nan, np.inf])
    def test_rejects_leak_gate_outside_closed_unit_interval(self, gate_v: float) -> None:
        """The model rejects leak gate outside closed unit interval."""
        with pytest.raises(ValueError, match="gate_v"):
            GatedLIFNeuron(gate_v=gate_v)

    @pytest.mark.parametrize("field", ["v_threshold", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_scale_parameters(
        self, field: str, value: float
    ) -> None:
        """The model rejects non-positive or non-finite scale parameters."""
        with pytest.raises(ValueError, match=field):
            kwargs: dict[str, Any] = {field: value}
            GatedLIFNeuron(**kwargs)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float) -> None:
        """A non-finite current is rejected before any state change."""
        n = GatedLIFNeuron(v=0.25)
        before = n.v
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert n.v == before
