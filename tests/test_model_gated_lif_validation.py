# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestValidation from former test_model_gated_lif.py

"""Focused suite: TestValidation from former test_model_gated_lif.py."""

from __future__ import annotations

from tests.model_gated_lif_support import *  # noqa: F403


class TestValidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("v", np.nan),
            ("gate_i", np.inf),
        ],
    )
    def test_rejects_non_finite_state_or_input_gate(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            GatedLIFNeuron(**{field: value})

    @pytest.mark.parametrize("gate_v", [-0.1, 1.1, np.nan, np.inf])
    def test_rejects_leak_gate_outside_closed_unit_interval(self, gate_v: float):
        with pytest.raises(ValueError, match="gate_v"):
            GatedLIFNeuron(gate_v=gate_v)

    @pytest.mark.parametrize("field", ["v_threshold", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_scale_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            GatedLIFNeuron(**{field: value})

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = GatedLIFNeuron(v=0.25)
        before = n.v
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert n.v == before
