# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVerilogGenerator from former test_model_zoo.py

"""Focused suite: TestVerilogGenerator from former test_model_zoo.py."""

from __future__ import annotations

from model_zoo_support import *  # noqa: F403

class TestVerilogGenerator:
    def test_generates_valid_module(self):
        gen = VerilogGenerator()
        sv = gen.generate(LIFPlugin())
        assert "module sc_neuron_lif" in sv
        assert "endmodule" in sv

    def test_contains_spdx_header(self):
        gen = VerilogGenerator()
        sv = gen.generate(LIFPlugin())
        assert "SPDX-License-Identifier" in sv
        assert "AGPL-3.0-or-later" in sv

    def test_ports_include_spike(self):
        gen = VerilogGenerator()
        sv = gen.generate(LIFPlugin())
        assert "o_spike" in sv

    def test_ports_include_state(self):
        gen = VerilogGenerator()
        sv = gen.generate(IzhikevichPlugin())
        assert "o_V" in sv
        assert "o_u" in sv

    def test_parameters_present(self):
        gen = VerilogGenerator()
        sv = gen.generate(LIFPlugin())
        assert "TAU_M" in sv or "V_REST" in sv

    def test_hh_generates_four_outputs(self):
        gen = VerilogGenerator()
        sv = gen.generate(HodgkinHuxleyPlugin())
        for var in ("o_V", "o_m", "o_h", "o_n"):
            assert var in sv

    def test_bit_width_configurable(self):
        gen = VerilogGenerator(bit_width=32, frac_bits=16)
        sv = gen.generate(LIFPlugin())
        assert "[31:0]" in sv

    def test_all_builtins_generate(self):
        gen = VerilogGenerator()
        for cls in (LIFPlugin, IzhikevichPlugin, AdExPlugin, HodgkinHuxleyPlugin):
            sv = gen.generate(cls())
            assert "module" in sv
            assert "endmodule" in sv
