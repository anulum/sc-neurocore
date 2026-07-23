# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHPAuthoritativeClocking from former test_adaptive_runtime_precision.py

"""Focused suite: TestHPAuthoritativeClocking from former test_adaptive_runtime_precision.py."""

from __future__ import annotations

from tests.adaptive_runtime_precision_support import *  # noqa: F403

class TestHPAuthoritativeClocking:
    """Verify the HP datapath remains clocked and authoritative."""

    def test_no_hp_clock_gate(self, lif_neuron):
        """Generated RTL must not gate clk with use_hp in fabric."""
        v = compile_adaptive_precision(lif_neuron)
        assert "hp_clk" not in v

    def test_no_clk_and_use_hp(self, lif_neuron):
        """Generated RTL must not create clk & use_hp."""
        v = compile_adaptive_precision(lif_neuron)
        assert "clk & use_hp" not in v

    def test_hp_inst_uses_clk(self, lif_neuron):
        """HP instance must use the primary clock."""
        v = compile_adaptive_precision(lif_neuron)
        hp_inst = v.split("hp_inst", 1)[1]
        assert ".clk(clk)" in hp_inst

    def test_use_hp_port(self, lif_neuron):
        """use_hp output port must be present as telemetry."""
        v = compile_adaptive_precision(lif_neuron)
        assert "output wire use_hp" in v
