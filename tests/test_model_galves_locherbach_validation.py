# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestValidation from former test_model_galves_locherbach.py

"""Focused suite: TestValidation from former test_model_galves_locherbach.py."""

from __future__ import annotations

from typing import Any
import numpy as np
import pytest
from sc_neurocore.neurons.models.galves_locherbach import (
    GalvesLocherbachNeuron,
)


class TestValidation:
    """Input and state validation of the GalvesLocherbachNeuron."""

    @pytest.mark.parametrize("field", ["v", "v_rest", "threshold_rate"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_voltage_parameters(self, field: str, value: float) -> None:
        """The model rejects non-finite voltage parameters."""
        with pytest.raises(ValueError, match=field):
            kwargs: dict[str, Any] = {field: value}
            GalvesLocherbachNeuron(**kwargs)

    @pytest.mark.parametrize("decay", [-0.1, 1.1, np.nan, np.inf, -np.inf])
    def test_rejects_decay_outside_unit_interval(self, decay: float) -> None:
        """The model rejects decay outside unit interval."""
        with pytest.raises(ValueError, match="decay"):
            GalvesLocherbachNeuron(decay=decay)

    @pytest.mark.parametrize("steepness", [0.0, -1.0, np.nan, np.inf, -np.inf])
    def test_rejects_non_positive_or_non_finite_steepness(self, steepness: float) -> None:
        """The model rejects non-positive or non-finite steepness."""
        with pytest.raises(ValueError, match="steepness"):
            GalvesLocherbachNeuron(steepness=steepness)

    @pytest.mark.parametrize("dt", [0.0, -1.0, 1.1, np.nan, np.inf, -np.inf])
    def test_rejects_invalid_probability_timestep(self, dt: float) -> None:
        """The model rejects invalid probability timestep."""
        with pytest.raises(ValueError, match="dt"):
            GalvesLocherbachNeuron(dt=dt)

    @pytest.mark.parametrize("weighted_input", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_weighted_input_before_state_mutation(
        self, weighted_input: float
    ) -> None:
        """The model rejects non-finite weighted input before state mutation."""
        n = GalvesLocherbachNeuron(v=0.25)
        before = n.v
        with pytest.raises(ValueError, match="weighted_input"):
            n.step(weighted_input)
        assert n.v == before
