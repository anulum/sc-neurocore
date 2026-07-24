# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRTLModuleParsing from former test_uvm_gen.py

"""Focused suite: TestRTLModuleParsing from former test_uvm_gen.py."""

from __future__ import annotations

from uvm_gen_support import *  # noqa: F403


class TestRTLModuleParsing:
    def test_parse_module_name(self):
        rtl = lif_module()
        assert rtl.name == "sc_lif_neuron"

    def test_parse_ports(self):
        rtl = lif_module()
        assert len(rtl.ports) >= 6

    def test_parse_params(self):
        rtl = lif_module()
        names = [p.name for p in rtl.params]
        assert "DATA_WIDTH" in names
        assert "V_THRESHOLD" in names

    def test_clock_detection(self):
        rtl = lif_module()
        assert rtl.clock_port is not None
        assert rtl.clock_port.name == "clk"

    def test_reset_detection(self):
        rtl = lif_module()
        assert rtl.reset_port is not None
        assert rtl.reset_port.name == "rst_n"

    def test_input_ports_exclude_clock_reset(self):
        rtl = lif_module()
        names = [p.name for p in rtl.input_ports]
        assert "clk" not in names
        assert "rst_n" not in names
        assert "I_t" in names

    def test_output_ports(self):
        rtl = lif_module()
        names = [p.name for p in rtl.output_ports]
        assert "spike_out" in names
        assert "v_out" in names

    def test_signed_ports(self):
        rtl = lif_module()
        it = next(p for p in rtl.ports if p.name == "I_t")
        assert it.is_signed is True

    def test_port_width(self):
        rtl = lif_module()
        it = next(p for p in rtl.ports if p.name == "I_t")
        assert it.width == 16

    def test_total_input_bits(self):
        rtl = lif_module()
        assert rtl.total_input_bits > 0

    def test_total_output_bits(self):
        rtl = lif_module()
        assert rtl.total_output_bits > 0

    def test_dense_module(self):
        rtl = dense_module()
        assert rtl.name == "sc_dense_layer_core"
        assert len(rtl.input_ports) == 1
        assert rtl.input_ports[0].name == "input_bus"
        assert rtl.input_ports[0].width == 8

    def test_no_module_raises(self):
        with pytest.raises(ValueError, match="No module"):
            RTLModule.from_verilog_source("// empty file")

    def test_sv_decl(self):
        p = ModulePort("foo", PortDirection.INPUT, PortType.LOGIC, 8, False)
        assert "input" in p.sv_decl
        assert "[7:0]" in p.sv_decl
