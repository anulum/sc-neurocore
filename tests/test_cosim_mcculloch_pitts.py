# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exact McCulloch-Pitts Python-to-Verilog co-simulation

"""Source rule, signed encoding, registered RTL and folded PE parity."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

from sc_neurocore.compiler.verilog_compiler import compile_to_datapath
from sc_neurocore.neurons.models.mcculloch_pitts import (
    McCullochPittsNeuron,
    encode_hardware_input,
)
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

_DATA_WIDTH = 32
_FRACTION = 0
_THETA = 2
_SOURCE_ROWS = (
    (0, True),
    (0, False),
    (1, False),
    (2, False),
    (3, True),
    ((1 << 31) - 1, False),
    (3, False),
)
_ENCODED_INPUTS = tuple(encode_hardware_input(*row) for row in _SOURCE_ROWS)


def _schema() -> UniversalNeuron:
    """Return the source threshold enrolled for exact RTL evaluation."""
    return UniversalNeuron.from_schema(
        "mcculloch_pitts",
        parameter_overrides={"theta": _THETA},
    )


def _compile_run(tmp_path: Path, stem: str, rtl: str, testbench: str) -> list[int]:
    """Compile one production RTL form and parse its complete binary trace."""
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
    events = re.findall(r"^MCP_TRACE ([01])$", completed.stdout, re.MULTILINE)
    assert len(events) == len(_SOURCE_ROWS), completed.stdout
    return [int(event) for event in events]


def _input_initialisers() -> str:
    """Return deterministic signed-Q32.0 testbench assignments."""

    def literal(value: int) -> str:
        return f"-32'sd{-value}" if value < 0 else f"32'sd{value}"

    return "\n".join(
        f"    inputs[{index}] = {literal(value)};" for index, value in enumerate(_ENCODED_INPUTS)
    )


def _registered_trace(tmp_path: Path) -> list[int]:
    """Run the production state-owning wrapper across every source row."""
    module_name = "sc_mcculloch_pitts_q320_registered"
    rtl = _schema().to_verilog(
        module_name=module_name,
        data_width=_DATA_WIDTH,
        fraction=_FRACTION,
    )
    testbench = f"""`timescale 1ns/1ps
module tb_mcculloch_pitts_registered;
reg clk = 1'b0;
reg rst_n = 1'b0;
reg signed [31:0] I_t = 32'sd0;
reg signed [31:0] inputs [0:{len(_SOURCE_ROWS) - 1}];
wire spike_out;
integer index;
always #5 clk = ~clk;
{module_name} dut (
    .clk(clk), .rst_n(rst_n), .I_t(I_t), .spike_out(spike_out)
);
initial begin
{_input_initialisers()}
    repeat (2) @(posedge clk);
    @(negedge clk); rst_n = 1'b1;
    for (index = 0; index < {len(_SOURCE_ROWS)}; index = index + 1) begin
        @(negedge clk); I_t = inputs[index];
        @(posedge clk); #1;
        $display("MCP_TRACE %0d", spike_out);
    end
    $finish;
end
endmodule
"""
    return _compile_run(tmp_path, "registered", rtl, testbench)


def _folded_trace(tmp_path: Path) -> list[int]:
    """Run the production folded combinational PE across every source row."""
    module_name = "sc_mcculloch_pitts_q320_folded"
    rtl = compile_to_datapath(
        _schema().to_equation_neuron(),
        module_name=module_name,
        data_width=_DATA_WIDTH,
        fraction=_FRACTION,
    )
    testbench = f"""`timescale 1ns/1ps
module tb_mcculloch_pitts_folded;
reg signed [31:0] I_t = 32'sd0;
reg signed [31:0] inputs [0:{len(_SOURCE_ROWS) - 1}];
wire spike_out;
integer index;
{module_name} dut (.I_t(I_t), .spike_out(spike_out));
initial begin
{_input_initialisers()}
    for (index = 0; index < {len(_SOURCE_ROWS)}; index = index + 1) begin
        I_t = inputs[index]; #1;
        $display("MCP_TRACE %0d", spike_out);
    end
    $finish;
end
endmodule
"""
    return _compile_run(tmp_path, "folded", rtl, testbench)


def test_q320_registered_and_folded_match_source_truth_rows(tmp_path: Path) -> None:
    """Both production RTL forms preserve threshold equality and veto rows."""
    assert HAS_IVERILOG, "Icarus Verilog is required for McCulloch-Pitts closure"
    source = McCullochPittsNeuron(theta=_THETA)
    expected = [source.step(*row) for row in _SOURCE_ROWS]
    assert expected == [0, 0, 0, 1, 0, 1, 1]
    assert _registered_trace(tmp_path) == expected
    assert _folded_trace(tmp_path) == expected


def test_q320_rtl_uses_only_signed_count_threshold_and_binary_output() -> None:
    """The hardware contract has no fabricated cell state or weighted sum."""
    registered = _schema().to_verilog(data_width=_DATA_WIDTH, fraction=_FRACTION)
    folded = compile_to_datapath(
        _schema().to_equation_neuron(),
        data_width=_DATA_WIDTH,
        fraction=_FRACTION,
    )
    for rtl in (registered, folded):
        assert "input wire signed [31:0] I_t" in rtl
        assert "parameter signed [31:0] P_THETA = 32'sd2" in rtl
        assert "I_t >= P_THETA" in rtl
        assert "output" in rtl and "spike_out" in rtl
        assert "v_out" not in rtl
