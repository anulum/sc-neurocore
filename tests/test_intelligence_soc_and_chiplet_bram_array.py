# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBRAMArray from former test_intelligence_soc_and_chiplet.py

"""Focused suite: TestBRAMArray from former test_intelligence_soc_and_chiplet.py."""

from __future__ import annotations

from tests.intelligence_soc_and_chiplet_support import *  # noqa: F403


class TestBRAMArray:
    """Tests for BRAM-backed neuron array generation."""

    def test_basic_array(self):
        """Default array generates valid Verilog."""
        from sc_neurocore.compiler.intelligence import generate_bram_array

        v = generate_bram_array()
        assert "module sc_neuron_array" in v
        assert "state_bram" in v
        assert "ram_style" in v
        assert "endmodule" in v

    def test_custom_count(self):
        """Custom neuron count."""
        from sc_neurocore.compiler.intelligence import generate_bram_array

        v = generate_bram_array(neuron_count=256)
        assert "[0:255]" in v

    def test_custom_module_name(self):
        """Custom module name."""
        from sc_neurocore.compiler.intelligence import generate_bram_array

        v = generate_bram_array(module_name="my_array")
        assert "module my_array" in v

    def test_spike_output(self):
        """Array has spike output ports."""
        from sc_neurocore.compiler.intelligence import generate_bram_array

        v = generate_bram_array()
        assert "spike_out" in v
        assert "spike_neuron_id" in v
        assert "tick_done" in v

    def test_current_based_lif_datapath_is_explicit(self):
        """Array documents and emits the concrete LIF update it implements."""
        from sc_neurocore.compiler.intelligence import generate_bram_array

        v = generate_bram_array()
        assert "current-based LIF datapath" in v
        assert "assign v_next = v_curr + (I_global >>> 4) - (v_curr >>> 3);" in v
        assert "TODO" not in v
