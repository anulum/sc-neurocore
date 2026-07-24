# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDualDatapath from former test_adaptive_runtime_precision.py

"""Focused suite: TestDualDatapath from former test_adaptive_runtime_precision.py."""

from __future__ import annotations

from tests.adaptive_runtime_precision_support import *  # noqa: F403


class TestDualDatapath:
    """Verify that both LP and HP datapaths are generated."""

    def test_contains_lp_module(self, lif_neuron):
        """LP sub-module must be present."""
        v = compile_adaptive_precision(lif_neuron, module_name="sc_lif_adapt")
        assert "module sc_lif_adapt_lp" in v

    def test_contains_hp_module(self, lif_neuron):
        """HP sub-module must be present."""
        v = compile_adaptive_precision(lif_neuron, module_name="sc_lif_adapt")
        assert "module sc_lif_adapt_hp" in v

    def test_contains_wrapper_module(self, lif_neuron):
        """Top-level wrapper module must be present."""
        v = compile_adaptive_precision(lif_neuron, module_name="sc_lif_adapt")
        assert "module sc_lif_adapt " in v or "module sc_lif_adapt\n" in v

    def test_lp_instantiation(self, lif_neuron):
        """LP datapath must be instantiated."""
        v = compile_adaptive_precision(lif_neuron, module_name="sc_lif_adapt")
        assert "lp_inst" in v

    def test_hp_instantiation(self, lif_neuron):
        """HP datapath must be instantiated."""
        v = compile_adaptive_precision(lif_neuron, module_name="sc_lif_adapt")
        assert "hp_inst" in v

    def test_three_endmodule(self, lif_neuron):
        """Must have 3 endmodule statements (LP, HP, wrapper)."""
        v = compile_adaptive_precision(lif_neuron, module_name="sc_lif_adapt")
        assert v.count("endmodule") == 3
