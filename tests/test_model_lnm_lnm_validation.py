# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLNMValidation from former test_model_lnm.py

"""Focused suite: TestLNMValidation from former test_model_lnm.py."""

from __future__ import annotations

from typing import Any
import numpy as np
import pytest
from sc_neurocore.neurons.models.lnm import (
    LearnableNeuronModel,
)


class TestLNMValidation:
    """Input and state validation of the LearnableNeuronModel."""

    @pytest.mark.parametrize(
        "field",
        ["v", "alpha", "beta", "gamma", "v_reset", "f_shift"],
    )
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_state_and_trainable_parameters(
        self, field: str, value: float
    ) -> None:
        """The model rejects non-finite state and trainable parameters."""
        with pytest.raises(ValueError, match=field):
            kwargs: dict[str, Any] = {field: value}
            LearnableNeuronModel(**kwargs)

    @pytest.mark.parametrize("field", ["v_threshold", "f_slope"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf])
    def test_rejects_non_positive_or_non_finite_physical_scales(
        self, field: str, value: float
    ) -> None:
        """The model rejects non-positive or non-finite physical scales."""
        with pytest.raises(ValueError, match=field):
            kwargs: dict[str, Any] = {field: value}
            LearnableNeuronModel(**kwargs)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float) -> None:
        """A non-finite current is rejected before any state change."""
        n = LearnableNeuronModel(v=0.25)
        before = n.v
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert n.v == before
