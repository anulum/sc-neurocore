# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_stochastic_source_emitters.py

from __future__ import annotations

"""Support extracted from test_stochastic_source_emitters.py."""

import shutil


import subprocess


from types import SimpleNamespace


import pytest


from sc_neurocore.edge.lfsr import Lfsr16


from sc_neurocore.edge.sobol import SobolGenerator


from sc_neurocore.hdl_gen import (
    Lfsr16Emitter,
    Sobol16Emitter,
    VerilogGenerator,
    emit_sources_from_ir,
)


_RTL_SAMPLE_COUNT = 24


def _lfsr16_step(state: int) -> int:
    feedback = ((state >> 0) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1
    return ((state >> 1) | (feedback << 15)) & 0xFFFF


def _sobol16_step(value: int, index: int) -> tuple[int, int]:
    directions = tuple(int(x) for x in SobolGenerator.DIRECTION_NUMBERS)
    if index == 0:
        c = 0
    else:
        c = (index & -index).bit_length() - 1
    return value ^ directions[c], index + 1


def _pack_sample_bits(samples: list[tuple[int, int, int]], word_bits: int) -> list[int]:
    words = [0] * ((len(samples) + word_bits - 1) // word_bits)
    for idx, _, bit in samples:
        if bit:
            words[idx // word_bits] |= 1 << (idx % word_bits)
    return words


def _simulate_source(verilog: str, testbench: str, tmp_path) -> list[tuple[int, int, int]]:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        raise AssertionError("iverilog and vvp must be available for stochastic-source RTL parity")

    rtl_path = tmp_path / "source.v"
    tb_path = tmp_path / "tb.v"
    out_path = tmp_path / "tb.out"
    rtl_path.write_text(verilog)
    tb_path.write_text(testbench)

    compile_result = subprocess.run(
        [iverilog, "-g2012", "-o", str(out_path), str(rtl_path), str(tb_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run(
        [vvp, str(out_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_result.returncode == 0, run_result.stderr

    samples = []
    for line in run_result.stdout.splitlines():
        if not line.startswith("sample "):
            continue
        _, idx, value, bit = line.split()
        samples.append((int(idx), int(value, 16), int(bit)))
    assert len(samples) == _RTL_SAMPLE_COUNT
    return samples


def _lfsr_testbench(module_name: str, threshold: int) -> str:
    return f"""
module tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg [15:0] threshold = 16'h{threshold:04X};
    wire bit_out;
    wire [15:0] state;
    integer i;

    {module_name} uut (
        .clk(clk),
        .rst_n(rst_n),
        .threshold(threshold),
        .bit_out(bit_out),
        .state(state)
    );

    initial begin
        #1 clk = 1'b1;
        #1 clk = 1'b0;
        rst_n = 1'b1;
        for (i = 0; i < {_RTL_SAMPLE_COUNT}; i = i + 1) begin
            #1 $display("sample %0d %04h %0d", i, state, bit_out);
            #1 clk = 1'b1;
            #1 clk = 1'b0;
        end
        $finish;
    end
endmodule
"""


def _sobol_testbench(module_name: str, threshold: int) -> str:
    return f"""
module tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg [15:0] threshold = 16'h{threshold:04X};
    wire bit_out;
    wire [15:0] value;
    wire [15:0] index;
    integer i;

    {module_name} uut (
        .clk(clk),
        .rst_n(rst_n),
        .threshold(threshold),
        .bit_out(bit_out),
        .value(value),
        .index(index)
    );

    initial begin
        #1 clk = 1'b1;
        #1 clk = 1'b0;
        rst_n = 1'b1;
        for (i = 0; i < {_RTL_SAMPLE_COUNT}; i = i + 1) begin
            #1 $display("sample %0d %04h %0d", i, value, bit_out);
            #1 clk = 1'b1;
            #1 clk = 1'b0;
        end
        $finish;
    end
endmodule
"""


__all__ = [
    "shutil",
    "subprocess",
    "SimpleNamespace",
    "pytest",
    "Lfsr16",
    "SobolGenerator",
    "Lfsr16Emitter",
    "Sobol16Emitter",
    "VerilogGenerator",
    "emit_sources_from_ir",
    "_RTL_SAMPLE_COUNT",
    "_lfsr16_step",
    "_sobol16_step",
    "_pack_sample_bits",
    "_simulate_source",
    "_lfsr_testbench",
    "_sobol_testbench",
]
