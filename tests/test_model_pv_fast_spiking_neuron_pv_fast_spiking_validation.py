# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPVFastSpikingValidation from former test_model_pv_fast_spiking_neuron.py

"""Focused suite: TestPVFastSpikingValidation from former test_model_pv_fast_spiking_neuron.py."""

from __future__ import annotations

from typing import Any
import pytest
from sc_neurocore.neurons.models.pv_fast_spiking_neuron import (
    PVFastSpikingNeuron,
)


class TestPVFastSpikingValidation:
    """Input and state validation of the PVFastSpikingNeuron."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"g_na": -1.0},
            {"g_k": 0.0},
            {"g_l": -0.1},
            {"c_m": 0.0},
            {"phi": 0.0},
            {"dt": 0.0},
            {"g_kv3": -1.0},
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs: dict[str, Any]) -> None:
        """The model rejects invalid parameters."""
        with pytest.raises(ValueError):
            PVFastSpikingNeuron(**kwargs)

    def test_accepts_zero_kv3_conductance(self) -> None:
        """The model accepts zero kv3 conductance."""
        assert PVFastSpikingNeuron(g_kv3=0.0).g_kv3 == 0.0

    @pytest.mark.parametrize("field", ["v", "e_na", "e_k", "e_l"])
    def test_rejects_non_finite_field(self, field: str) -> None:
        """The model rejects non-finite field."""
        with pytest.raises(ValueError, match="must be finite"):
            kwargs: dict[str, Any] = {field: float("nan")}
            PVFastSpikingNeuron(**kwargs)

    def test_rejects_boolean_field(self) -> None:
        """The model rejects boolean field."""
        with pytest.raises(ValueError, match="must be finite"):
            PVFastSpikingNeuron(v=True)

    def test_rejects_non_finite_current(self) -> None:
        """The model rejects non-finite current."""
        with pytest.raises(ValueError, match="must be finite"):
            PVFastSpikingNeuron().step(float("inf"))

    def test_runtime_validation_catches_corrupted_state(self) -> None:
        """Runtime validation catches corrupted state."""
        n = PVFastSpikingNeuron()
        n.dt = -1.0
        with pytest.raises(ValueError, match="dt must be positive"):
            n.step(0.0)

    def test_non_finite_candidate_fails_closed(self) -> None:
        """Non-finite candidate fails closed."""
        n = PVFastSpikingNeuron()
        with pytest.raises((FloatingPointError, OverflowError)):
            for _ in range(60):
                n.step(1e308)
