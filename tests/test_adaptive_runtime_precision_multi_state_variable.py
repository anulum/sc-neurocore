# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMultiStateVariable from former test_adaptive_runtime_precision.py

"""Focused suite: TestMultiStateVariable from former test_adaptive_runtime_precision.py."""

from __future__ import annotations

from tests.adaptive_runtime_precision_support import *  # noqa: F403

class TestMultiStateVariable:
    """Verify adaptive precision with multi-state-variable neurons."""

    def test_izhikevich_both_vars_present(self, izhikevich_neuron):
        """Both v and u must appear in all three modules."""
        v = compile_adaptive_precision(izhikevich_neuron, module_name="sc_izh_adapt")
        assert "v_reg" in v
        assert "u_reg" in v
        assert "lp_v_out" in v
        assert "lp_u_out" in v
        assert "hp_v_out" in v
        assert "hp_u_out" in v

    def test_izhikevich_hp_authoritative_both_vars(self, izhikevich_neuron):
        """HP output assignment must be applied to both state variables."""
        v = compile_adaptive_precision(
            izhikevich_neuron,
            module_name="sc_izh_adapt",
            lp_width=16,
            lp_frac=8,
            hp_width=32,
            hp_frac=16,
        )
        assert "v_out <= hp_v_out;" in v
        assert "u_out <= hp_u_out;" in v
