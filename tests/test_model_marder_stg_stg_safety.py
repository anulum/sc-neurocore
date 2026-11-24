# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSTGSafety from former test_model_marder_stg.py

"""Focused suite: TestSTGSafety from former test_model_marder_stg.py."""

from __future__ import annotations

from typing import Any
import pytest
from sc_neurocore.neurons.models.marder_stg import (
    MarderSTGNeuron,
)


class TestSTGSafety:
    """Input, configuration and state validation of the MarderSTGNeuron."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("dt", 0.0),
            ("cm", 0.0),
            ("tau_ca", 0.0),
            ("ca_out", 0.0),
            ("g_na", -1.0),
            ("g_kca", -1.0),
            ("m_na", 1.01),
            ("h_cas", -0.01),
            ("m_kca", 1.5),
            ("ca", -0.01),
        ],
    )
    def test_rejects_invalid_configuration(self, field: str, value: float) -> None:
        """The model rejects invalid configuration."""
        with pytest.raises(ValueError):
            kwargs: dict[str, Any] = {field: value}
            MarderSTGNeuron(**kwargs)

    def test_rejects_non_finite_input_before_mutation(self) -> None:
        """The model rejects non-finite input before mutation."""
        n = MarderSTGNeuron()
        before = n.v
        with pytest.raises(ValueError):
            n.step(float("nan"))
        assert n.v == before

    def test_rejects_runtime_corruption_before_mutation(self) -> None:
        """The model rejects runtime corruption before mutation."""
        n = MarderSTGNeuron()
        n.cm = 0.0
        before = n.v
        with pytest.raises(ValueError):
            n.step(0.0)
        assert n.v == before

    def test_extreme_timestep_fails_closed(self) -> None:
        """Extreme timestep fails closed."""
        n = MarderSTGNeuron(dt=5.0)
        with pytest.raises(FloatingPointError):
            for _ in range(500):
                n.step(0.0)

    def test_commit_rejects_non_finite_state(self) -> None:
        """Commit rejects non-finite state."""
        bad = (float("nan"),) + (0.5,) * 11 + (0.05,)
        with pytest.raises(FloatingPointError):
            MarderSTGNeuron._commit(bad)

    def test_commit_clamps_gates_and_calcium(self) -> None:
        """Commit clamps gates and calcium."""
        raw = (-50.0, 1.5, -0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.0, -0.1, 0.5, -3.0)
        out = MarderSTGNeuron._commit(raw)
        assert out[1] == 1.0 and out[2] == 0.0 and out[9] == 1.0 and out[10] == 0.0
        assert out[12] == 0.0
