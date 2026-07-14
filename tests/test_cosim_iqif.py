# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exact IQIF Python-to-Verilog co-simulation

"""Bit-true source, schema, registered RTL, and folded PE trace parity."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

from sc_neurocore.compiler.verilog_compiler import compile_to_datapath
from sc_neurocore.neurons.models.iqif import IntegerQIFNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

_N_STEPS = 400
_CURRENT = 10
_DATA_WIDTH = 32
_FRACTION = 0


def _compile_run(tmp_path: Path, stem: str, rtl: str, testbench: str) -> list[tuple[int, int]]:
    rtl_path = tmp_path / f"{stem}.v"
    testbench_path = tmp_path / f"tb_{stem}.v"
    executable = tmp_path / stem
    rtl_path.write_text(rtl, encoding="utf-8")
    testbench_path.write_text(testbench, encoding="utf-8")
    subprocess.run(
        ["iverilog", "-g2012", "-o", str(executable), str(rtl_path), str(testbench_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    completed = subprocess.run(
        ["vvp", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows = re.findall(r"^IQIF_TRACE ([01]) (-?\d+)$", completed.stdout, re.MULTILINE)
    assert len(rows) == _N_STEPS, completed.stdout
    return [(int(event), int(v)) for event, v in rows]


def _registered_trace(tmp_path: Path) -> list[tuple[int, int]]:
    module_name = "sc_iqif_q320_registered"
    rtl = UniversalNeuron.from_schema("iqif").to_verilog(
        module_name=module_name,
        data_width=_DATA_WIDTH,
        fraction=_FRACTION,
    )
    testbench = f"""`timescale 1ns/1ps
module tb_iqif_registered;
reg clk = 1'b0;
reg rst_n = 1'b0;
wire spike_out;
wire signed [31:0] v_out;
integer index;
always #5 clk = ~clk;
{module_name} dut (
    .clk(clk), .rst_n(rst_n), .I_t(32'sd{_CURRENT}),
    .spike_out(spike_out), .v_out(v_out)
);
initial begin
    #23; rst_n = 1'b1;
    for (index = 0; index < {_N_STEPS}; index = index + 1) begin
        @(posedge clk); #1;
        $display("IQIF_TRACE %0d %0d", spike_out, v_out);
    end
    $finish;
end
endmodule
"""
    return _compile_run(tmp_path, "registered", rtl, testbench)


def _folded_trace(tmp_path: Path) -> list[tuple[int, int]]:
    module_name = "sc_iqif_q320_folded"
    model = UniversalNeuron.from_schema("iqif").to_equation_neuron()
    rtl = compile_to_datapath(
        model,
        module_name=module_name,
        data_width=_DATA_WIDTH,
        fraction=_FRACTION,
    )
    testbench = f"""`timescale 1ns/1ps
module tb_iqif_folded;
reg signed [31:0] v_reg = 32'sd128;
wire signed [31:0] v_next_out;
wire spike_out;
integer index;
{module_name} dut (
    .I_t(32'sd{_CURRENT}), .v_reg(v_reg),
    .v_next_out(v_next_out), .spike_out(spike_out)
);
initial begin
    for (index = 0; index < {_N_STEPS}; index = index + 1) begin
        #1;
        $display("IQIF_TRACE %0d %0d", spike_out, v_next_out);
        v_reg = v_next_out;
        #1;
    end
    $finish;
end
endmodule
"""
    return _compile_run(tmp_path, "folded", rtl, testbench)


def test_iqif_q320_registered_and_folded_match_source_trace(tmp_path: Path) -> None:
    """Both production RTL forms reproduce all 400 source states and events."""
    assert HAS_IVERILOG, "Icarus Verilog is required for IQIF fidelity closure"
    hand = IntegerQIFNeuron()
    expected: list[tuple[int, int]] = []
    for _ in range(_N_STEPS):
        event = hand.step(_CURRENT)
        expected.append((event, hand.v))

    registered = _registered_trace(tmp_path)
    folded = _folded_trace(tmp_path)
    assert registered == expected
    assert folded == expected
    assert sum(event for event, _v in registered) == 26
    assert registered[-1] == (0, 198)


def test_iqif_q320_generated_rtl_contains_exact_floor_shift() -> None:
    """The source Q0.3 divider is a signed three-bit shift in integer RTL."""
    rtl = UniversalNeuron.from_schema("iqif").to_verilog(data_width=32, fraction=0)
    shifts = re.findall(r"\$signed\(_floordiv\d+_dividend\) >>> 3", rtl)
    assert len(shifts) == 2
