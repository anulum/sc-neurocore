# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHPAuthoritativeOutput from former test_adaptive_runtime_precision.py

"""Focused suite: TestHPAuthoritativeOutput from former test_adaptive_runtime_precision.py."""

from __future__ import annotations

from tests.adaptive_runtime_precision_support import *  # noqa: F403


class TestHPAuthoritativeOutput:
    """Verify outputs are taken from HP, never LP-converted state."""

    def test_output_register_uses_hp_spike(self, lif_neuron):
        """Spike output must be driven by HP."""
        v = compile_adaptive_precision(lif_neuron, lp_width=16, lp_frac=8, hp_width=32, hp_frac=16)
        assert "spike_out <= hp_spike;" in v

    def test_output_register_uses_hp_state(self, lif_neuron):
        """State output must be driven by HP."""
        v = compile_adaptive_precision(lif_neuron, lp_width=16, lp_frac=8, hp_width=32, hp_frac=16)
        assert "v_out <= hp_v_out;" in v
        assert "v_out <= lp_v_out" not in v
