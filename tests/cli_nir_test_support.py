# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR CLI test fixtures and simulation references

"""Build NIR fixtures and simulate emitted CLI network artefacts."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _small_lif_nir_graph() -> Any:
    nir = pytest.importorskip("nir")
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.array([[0.25, -0.5], [0.75, 0.125]], dtype=np.float32),
                bias=np.zeros(2, dtype=np.float32),
            ),
            "lif": nir.LIF(
                tau=np.full(2, 20.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[("input", "aff"), ("aff", "lif"), ("lif", "output")],
    )


def _nested_single_port_lif_nir_graph() -> Any:
    nir = pytest.importorskip("nir")
    inner = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.array([[0.25, -0.5], [0.75, 0.125]], dtype=np.float32),
                bias=np.array([0.125, -0.25], dtype=np.float32),
            ),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[("input", "aff"), ("aff", "output")],
    )
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "subgraph": inner,
            "lif": nir.LIF(
                tau=np.full(2, 20.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[("input", "subgraph"), ("subgraph", "lif"), ("lif", "output")],
    )


def _nested_multiport_multioutput_lif_nir_graph() -> Any:
    nir = pytest.importorskip("nir")
    inner = nir.NIRGraph(
        nodes={
            "a": nir.Input(input_type={"input": np.array([1])}),
            "b": nir.Input(input_type={"input": np.array([1])}),
            "aff_a": nir.Affine(
                weight=np.array([[0.5]], dtype=np.float32),
                bias=np.zeros(1, dtype=np.float32),
            ),
            "aff_b": nir.Affine(
                weight=np.array([[-0.25]], dtype=np.float32),
                bias=np.zeros(1, dtype=np.float32),
            ),
            "out_a": nir.Output(output_type={"output": np.array([1])}),
            "out_b": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[
            ("a", "aff_a"),
            ("aff_a", "out_a"),
            ("b", "aff_b"),
            ("aff_b", "out_b"),
        ],
        type_check=False,
    )
    return nir.NIRGraph(
        nodes={
            "left": nir.Input(input_type={"input": np.array([1])}),
            "right": nir.Input(input_type={"input": np.array([1])}),
            "subgraph": inner,
            "lif_a": nir.LIF(
                tau=np.full(1, 20.0),
                r=np.ones(1),
                v_leak=np.zeros(1),
                v_threshold=np.ones(1),
            ),
            "lif_b": nir.LIF(
                tau=np.full(1, 20.0),
                r=np.ones(1),
                v_leak=np.zeros(1),
                v_threshold=np.ones(1),
            ),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[
            ("left", "subgraph"),
            ("right", "subgraph"),
            ("subgraph", "lif_a"),
            ("subgraph", "lif_b"),
            ("lif_a", "output"),
            ("lif_b", "output"),
        ],
        type_check=False,
    )


def _dense_lif_nir_graph() -> Any:
    nir = pytest.importorskip("nir")
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([3])}),
            "aff1": nir.Affine(
                weight=np.array(
                    [
                        [0.25, -0.5, 0.125],
                        [0.75, 0.125, -0.375],
                        [-0.25, 0.5, 0.625],
                        [0.0, -0.125, 0.25],
                    ],
                    dtype=np.float32,
                ),
                bias=np.zeros(4, dtype=np.float32),
            ),
            "lif1": nir.LIF(
                tau=np.full(4, 20.0),
                r=np.ones(4),
                v_leak=np.zeros(4),
                v_threshold=np.ones(4),
            ),
            "aff2": nir.Affine(
                weight=np.array(
                    [
                        [0.5, -0.25, 0.125, 0.0],
                        [-0.125, 0.375, -0.5, 0.25],
                    ],
                    dtype=np.float32,
                ),
                bias=np.zeros(2, dtype=np.float32),
            ),
            "lif2": nir.LIF(
                tau=np.full(2, 20.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[
            ("input", "aff1"),
            ("aff1", "lif1"),
            ("lif1", "aff2"),
            ("aff2", "lif2"),
            ("lif2", "output"),
        ],
    )


def _aer_lif_nir_graph() -> Any:
    nir = pytest.importorskip("nir")
    n_in = 4
    n_hidden = 65
    n_out = 2
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([n_in])}),
            "aff1": nir.Affine(
                weight=np.full((n_hidden, n_in), 0.125, dtype=np.float32),
                bias=np.zeros(n_hidden, dtype=np.float32),
            ),
            "lif1": nir.LIF(
                tau=np.full(n_hidden, 20.0),
                r=np.ones(n_hidden),
                v_leak=np.zeros(n_hidden),
                v_threshold=np.ones(n_hidden),
            ),
            "aff2": nir.Affine(
                weight=np.full((n_out, n_hidden), -0.0625, dtype=np.float32),
                bias=np.zeros(n_out, dtype=np.float32),
            ),
            "lif2": nir.LIF(
                tau=np.full(n_out, 20.0),
                r=np.ones(n_out),
                v_leak=np.zeros(n_out),
                v_threshold=np.ones(n_out),
            ),
            "output": nir.Output(output_type={"output": np.array([n_out])}),
        },
        edges=[
            ("input", "aff1"),
            ("aff1", "lif1"),
            ("lif1", "aff2"),
            ("aff2", "lif2"),
            ("lif2", "output"),
        ],
    )


def _recurrent_lif_nir_graph() -> Any:
    nir = pytest.importorskip("nir")
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.eye(2, dtype=np.float32),
                bias=np.zeros(2, dtype=np.float32),
            ),
            "lif": nir.LIF(
                tau=np.full(2, 20.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "rec": nir.Linear(weight=np.array([[0.125, 0.0], [0.0, -0.25]], dtype=np.float32)),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[
            ("input", "aff"),
            ("aff", "lif"),
            ("lif", "rec"),
            ("rec", "lif"),
            ("lif", "output"),
        ],
    )


def _mixed_li_lif_nir_graph() -> Any:
    nir = pytest.importorskip("nir")
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.eye(2, dtype=np.float32),
                bias=np.zeros(2, dtype=np.float32),
            ),
            "li": nir.LI(
                tau=np.full(2, 15.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
            ),
            "readout": nir.Linear(weight=np.array([[0.5, -0.25]], dtype=np.float32)),
            "lif": nir.LIF(
                tau=np.full(1, 20.0),
                r=np.ones(1),
                v_leak=np.zeros(1),
                v_threshold=np.ones(1),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[
            ("input", "aff"),
            ("aff", "li"),
            ("li", "readout"),
            ("readout", "lif"),
            ("lif", "output"),
        ],
    )


def _mixed_aer_li_lif_nir_graph() -> Any:
    nir = pytest.importorskip("nir")
    n_hidden = 65
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([3])}),
            "aff": nir.Affine(
                weight=np.array(
                    [[0.25, -0.125, 0.5], [-0.5, 0.375, 0.125]],
                    dtype=np.float32,
                ),
                bias=np.zeros(2, dtype=np.float32),
            ),
            "li": nir.LI(
                tau=np.full(2, 15.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
            ),
            "expand": nir.Linear(
                weight=np.tile(np.array([[0.5, -0.25]], dtype=np.float32), (n_hidden, 1))
            ),
            "lif1": nir.LIF(
                tau=np.full(n_hidden, 20.0),
                r=np.ones(n_hidden),
                v_leak=np.zeros(n_hidden),
                v_threshold=np.ones(n_hidden),
            ),
            "readout": nir.Linear(weight=np.full((1, n_hidden), 0.0625, dtype=np.float32)),
            "lif2": nir.LIF(
                tau=np.full(1, 20.0),
                r=np.ones(1),
                v_leak=np.zeros(1),
                v_threshold=np.ones(1),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[
            ("input", "aff"),
            ("aff", "li"),
            ("li", "expand"),
            ("expand", "lif1"),
            ("lif1", "readout"),
            ("readout", "lif2"),
            ("lif2", "output"),
        ],
    )


def _sobol_source_smoke_testbench(module_name: str) -> str:
    return f"""
module tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg [15:0] threshold = 16'h9000;
    wire bit_out;
    wire [15:0] value;
    wire [15:0] index;

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
        #1 $display("sample0 %04h %0d %0d", value, bit_out, index);
        $finish;
    end
endmodule
"""


def _lfsr_source_smoke_testbench(module_name: str) -> str:
    return f"""
module tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg [15:0] threshold = 16'h9000;
    wire bit_out;
    wire [15:0] state;

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
        #1 $display("sample0 %04h %0d", state, bit_out);
        $finish;
    end
endmodule
"""


def _lfsr16_step(state: int) -> int:
    feedback = ((state >> 0) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1
    return ((state >> 1) | (feedback << 15)) & 0xFFFF


def _simulate_single_source_module(module_name: str, verilog: str, tmp_path: Path) -> str:
    return _simulate_source_module(
        module_name,
        verilog,
        testbench=_sobol_source_smoke_testbench(module_name),
        tmp_path=tmp_path,
    )


def _simulate_source_module(
    module_name: str,
    verilog: str,
    *,
    testbench: str,
    tmp_path: Path,
) -> str:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    assert iverilog is not None and vvp is not None, "Icarus Verilog must be installed"

    rtl_path = tmp_path / f"{module_name}.v"
    tb_path = tmp_path / "tb_source.v"
    out_path = tmp_path / "tb_source.out"
    rtl_path.write_text(verilog, encoding="utf-8")
    tb_path.write_text(testbench, encoding="utf-8")

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
    return run_result.stdout


def _simulate_manifest_source(
    out_dir: Path,
    row: Mapping[str, Any],
    tmp_path: Path,
) -> str:
    module_name = str(row["module_name"])
    source_verilog = (out_dir / f"{module_name}.v").read_text(encoding="utf-8")

    if row["source_kind"] == "sobol16":
        stdout = _simulate_source_module(
            module_name,
            source_verilog,
            testbench=_sobol_source_smoke_testbench(module_name),
            tmp_path=tmp_path,
        )
        first_sample = (int(row["seed"]) ^ 0x8000) & 0xFFFF
        assert f"sample0 {first_sample:04x} {int(first_sample < 0x9000)} 1" in stdout
        return stdout

    if row["source_kind"] == "lfsr16":
        stdout = _simulate_source_module(
            module_name,
            source_verilog,
            testbench=_lfsr_source_smoke_testbench(module_name),
            tmp_path=tmp_path,
        )
        first_sample = _lfsr16_step(int(row["seed"]))
        assert f"sample0 {first_sample:04x} {int(first_sample < 0x9000)}" in stdout
        return stdout

    raise AssertionError(f"unsupported source_kind {row['source_kind']!r}")


def _network_smoke_testbench(module_name: str, input_words: int, total_neurons: int) -> str:
    input_width = max(1, input_words * 16)
    spike_width = max(1, total_neurons)
    input_hex = "".join("0200" for _ in range(input_words))
    if not input_hex:
        input_hex = "0"
    return f"""
module tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg en = 1'b0;
    reg signed [{input_width - 1}:0] I_ext_flat = {input_width}'h{input_hex};
    wire [{spike_width - 1}:0] spike_bus;
    integer cycle;

    {module_name} uut (
        .clk(clk),
        .rst_n(rst_n),
        .en(en),
        .I_ext_flat(I_ext_flat),
        .spike_bus(spike_bus)
    );

    initial begin
        $display("network_start {total_neurons}");
        #1 clk = 1'b1;
        #1 clk = 1'b0;
        rst_n = 1'b1;
        en = 1'b1;
        #1;
        for (cycle = 0; cycle < 8; cycle = cycle + 1) begin
            #1 clk = 1'b1;
            #1 $display("cycle %0d spikes %0h", cycle, spike_bus);
            #1 clk = 1'b0;
        end
        $display("network_done {module_name}");
        $finish;
    end
endmodule
"""


def _simulate_network_bundle(
    out_dir: Path,
    *,
    module_name: str,
    input_words: int,
    total_neurons: int,
    tmp_path: Path,
) -> str:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    assert iverilog is not None and vvp is not None, "Icarus Verilog must be installed"

    rtl_paths = sorted(out_dir.glob("*.v"))
    assert rtl_paths, f"no RTL files emitted in {out_dir}"
    tb_path = tmp_path / f"{module_name}_network_tb.v"
    out_path = tmp_path / f"{module_name}_network_tb.out"
    tb_path.write_text(
        _network_smoke_testbench(module_name, input_words, total_neurons),
        encoding="utf-8",
    )

    compile_result = subprocess.run(
        [iverilog, "-g2012", "-o", str(out_path), *map(str, rtl_paths), str(tb_path)],
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
    return run_result.stdout


def _direct_equivalence_testbench(module_name: str) -> str:
    return f"""
module tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg en = 1'b0;
    reg signed [31:0] I_ext_flat = 32'h02000200;
    wire [1:0] spike_bus;
    integer cycle;

    {module_name} uut (
        .clk(clk),
        .rst_n(rst_n),
        .en(en),
        .I_ext_flat(I_ext_flat),
        .spike_bus(spike_bus)
    );

    initial begin
        #1 clk = 1'b1;
        #1 clk = 1'b0;
        rst_n = 1'b1;
        en = 1'b1;
        for (cycle = 0; cycle < 8; cycle = cycle + 1) begin
            #1 clk = 1'b1;
            #1 $display(
                "cycle %0d i0 %0d v0 %0d s0 %0d i1 %0d v1 %0d s1 %0d",
                cycle,
                uut.p0_n0_I,
                uut.p0_n0_v,
                spike_bus[0],
                uut.p0_n1_I,
                uut.p0_n1_v,
                spike_bus[1]
            );
            #1 clk = 1'b0;
        end
        $finish;
    end
endmodule
"""


def _recurrent_equivalence_testbench(module_name: str) -> str:
    return f"""
module tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg en = 1'b0;
    reg signed [31:0] I_ext_flat = 32'h20002000;
    wire [1:0] spike_bus;
    reg signed [15:0] i0_pre;
    reg signed [15:0] i1_pre;
    reg d0_pre;
    reg d1_pre;
    integer cycle;

    {module_name} uut (
        .clk(clk),
        .rst_n(rst_n),
        .en(en),
        .I_ext_flat(I_ext_flat),
        .spike_bus(spike_bus)
    );

    initial begin
        #1 clk = 1'b1;
        #1 clk = 1'b0;
        rst_n = 1'b1;
        en = 1'b1;
        #1;
        for (cycle = 0; cycle < 8; cycle = cycle + 1) begin
            i0_pre = uut.p0_n0_I;
            i1_pre = uut.p0_n1_I;
            d0_pre = uut.p0_n0_spike_d1;
            d1_pre = uut.p0_n1_spike_d1;
            #1 clk = 1'b1;
            #1 $display(
                "cycle %0d i0_pre %0d d0_pre %0d v0 %0d s0 %0d d0_post %0d i0_post %0d i1_pre %0d d1_pre %0d v1 %0d s1 %0d d1_post %0d i1_post %0d",
                cycle,
                i0_pre,
                d0_pre,
                uut.p0_n0_v,
                spike_bus[0],
                uut.p0_n0_spike_d1,
                uut.p0_n0_I,
                i1_pre,
                d1_pre,
                uut.p0_n1_v,
                spike_bus[1],
                uut.p0_n1_spike_d1,
                uut.p0_n1_I
            );
            #1 clk = 1'b0;
        end
        $finish;
    end
endmodule
"""


def _aer_equivalence_testbench(module_name: str) -> str:
    return f"""
module tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg en = 1'b0;
    reg signed [63:0] I_ext_flat = 64'h2000200020002000;
    wire [66:0] spike_bus;
    reg signed [15:0] h0_i_pre;
    reg signed [15:0] out0_i_pre;
    integer hidden_pre;
    integer hidden_post;
    integer cycle;

    {module_name} uut (
        .clk(clk),
        .rst_n(rst_n),
        .en(en),
        .I_ext_flat(I_ext_flat),
        .spike_bus(spike_bus)
    );

    function integer count_hidden_spikes;
        integer j;
        begin
            count_hidden_spikes = 0;
            for (j = 0; j < 65; j = j + 1) begin
                if (spike_bus[j]) begin
                    count_hidden_spikes = count_hidden_spikes + 1;
                end
            end
        end
    endfunction

    initial begin
        #1 clk = 1'b1;
        #1 clk = 1'b0;
        rst_n = 1'b1;
        en = 1'b1;
        #1;
        for (cycle = 0; cycle < 8; cycle = cycle + 1) begin
            h0_i_pre = uut.p0_n0_I;
            out0_i_pre = uut.p1_n0_I;
            hidden_pre = count_hidden_spikes();
            #1 clk = 1'b1;
            #1 hidden_post = count_hidden_spikes();
            #1 $display(
                "cycle %0d hpre %0d hpost %0d h0_i %0d h0_v %0d h0_s %0d out_i_pre %0d out_i_post %0d out_v %0d out_s %0d",
                cycle,
                hidden_pre,
                hidden_post,
                h0_i_pre,
                uut.p0_n0_v,
                spike_bus[0],
                out0_i_pre,
                uut.p1_n0_I,
                uut.p1_n0_v,
                spike_bus[65]
            );
            #1 clk = 1'b0;
        end
        $finish;
    end
endmodule
"""


def _mixed_equivalence_testbench(module_name: str) -> str:
    return f"""
module tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg en = 1'b0;
    reg signed [31:0] I_ext_flat = 32'h08002000;
    wire [2:0] spike_bus;
    reg signed [15:0] li0_i_pre;
    reg signed [15:0] li1_i_pre;
    reg signed [15:0] lif_i_pre;
    integer cycle;

    {module_name} uut (
        .clk(clk),
        .rst_n(rst_n),
        .en(en),
        .I_ext_flat(I_ext_flat),
        .spike_bus(spike_bus)
    );

    initial begin
        #1 clk = 1'b1;
        #1 clk = 1'b0;
        rst_n = 1'b1;
        en = 1'b1;
        #1;
        for (cycle = 0; cycle < 8; cycle = cycle + 1) begin
            li0_i_pre = uut.p0_n0_I;
            li1_i_pre = uut.p0_n1_I;
            lif_i_pre = uut.p1_n0_I;
            #1 clk = 1'b1;
            #1 $display(
                "cycle %0d li0_i_pre %0d li1_i_pre %0d lif_i_pre %0d li0_v %0d li1_v %0d lif_i_post %0d lif_v %0d lif_s %0d",
                cycle,
                li0_i_pre,
                li1_i_pre,
                lif_i_pre,
                uut.p0_n0_v,
                uut.p0_n1_v,
                uut.p1_n0_I,
                uut.p1_n0_v,
                spike_bus[2]
            );
            #1 clk = 1'b0;
        end
        $finish;
    end
endmodule
"""


def _simulate_network_with_testbench(
    out_dir: Path,
    *,
    module_name: str,
    testbench: str,
    tmp_path: Path,
) -> str:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    assert iverilog is not None and vvp is not None, "Icarus Verilog must be installed"

    rtl_paths = sorted(out_dir.glob("*.v"))
    assert rtl_paths, f"no RTL files emitted in {out_dir}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    tb_path = tmp_path / f"{module_name}_equivalence_tb.v"
    out_path = tmp_path / f"{module_name}_equivalence_tb.out"
    tb_path.write_text(testbench, encoding="utf-8")

    compile_result = subprocess.run(
        [iverilog, "-g2012", "-o", str(out_path), *map(str, rtl_paths), str(tb_path)],
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
    return run_result.stdout


def _trunc_div(numerator: int, denominator: int) -> int:
    """Divide with truncation toward zero, matching signed Verilog division."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    sign = -1 if numerator < 0 else 1
    return sign * (abs(numerator) // denominator)


def _small_direct_fixed_point_reference(
    cycles: int,
) -> list[tuple[int, ...]]:
    ext0 = 0x0200
    ext1 = 0x0200
    current0 = ((ext0 * 0x0040) >> 8) + _trunc_div(ext1 * -0x0080, 1 << 8)
    current1 = ((ext0 * 0x00C0) >> 8) + ((ext1 * 0x0020) >> 8)
    voltages = [0, 0]
    currents = [current0, current1]
    rows: list[tuple[int, ...]] = []

    for cycle in range(cycles):
        row: list[int] = [cycle]
        for idx, current in enumerate(currents):
            leak = _trunc_div((-voltages[idx]) << 8, 5120)
            drive = _trunc_div(current << 8, 5120)
            next_voltage = max(-32768, min(32767, voltages[idx] + leak + drive))
            spike = int(next_voltage > 256)
            if spike:
                observed_voltage = 0
                voltages[idx] = 0
            else:
                observed_voltage = next_voltage
                voltages[idx] = next_voltage
            row.extend([current, observed_voltage, spike])
        rows.append(tuple(row))

    return rows


def _lif_q88_step(voltage: int, current: int) -> tuple[int, int, int]:
    """Advance the NIR LIF recurrence and expose its post-reset output state."""
    leak = _trunc_div((-voltage) << 8, 5120)
    drive = _trunc_div(current << 8, 5120)
    next_voltage = max(-32768, min(32767, voltage + leak + drive))
    spike = int(next_voltage > 256)
    if spike:
        return 0, spike, 0
    return next_voltage, spike, next_voltage


def _li_q88_step(voltage: int, current: int) -> int:
    leak = _trunc_div((-voltage) << 8, 3840)
    drive = _trunc_div(current << 8, 3840)
    return max(-32768, min(32767, voltage + leak + drive))


def _recurrent_fixed_point_reference(
    cycles: int,
) -> list[tuple[int, int, int, int, int, int, int, int, int, int, int, int, int]]:
    ext_current = 0x2000
    voltages = [0, 0]
    spike_outputs = [0, 0]
    delayed_spikes = [0, 0]
    rows: list[tuple[int, int, int, int, int, int, int, int, int, int, int, int, int]] = []

    for cycle in range(cycles):
        current_pre = [
            ext_current + (0x0020 if delayed_spikes[0] else 0),
            ext_current + (-0x0040 if delayed_spikes[1] else 0),
        ]
        delayed_pre = delayed_spikes.copy()
        observed: list[tuple[int, int, int]] = []
        next_voltages: list[int] = []
        for voltage, current in zip(voltages, current_pre, strict=True):
            observed_voltage, spike, next_voltage = _lif_q88_step(voltage, current)
            observed.append((observed_voltage, spike, next_voltage))
            next_voltages.append(next_voltage)

        delayed_post = spike_outputs.copy()
        current_post = [
            ext_current + (0x0020 if delayed_post[0] else 0),
            ext_current + (-0x0040 if delayed_post[1] else 0),
        ]
        rows.append(
            (
                cycle,
                current_pre[0],
                delayed_pre[0],
                observed[0][0],
                observed[0][1],
                delayed_post[0],
                current_post[0],
                current_pre[1],
                delayed_pre[1],
                observed[1][0],
                observed[1][1],
                delayed_post[1],
                current_post[1],
            )
        )
        voltages = next_voltages
        spike_outputs = [observed[0][1], observed[1][1]]
        delayed_spikes = delayed_post

    return rows


def _mixed_readout_current(li0_voltage: int, li1_voltage: int) -> int:
    return ((li0_voltage * 0x0080) >> 8) + ((li1_voltage * -0x0040) >> 8)


def _mixed_fixed_point_reference(
    cycles: int,
) -> list[tuple[int, int, int, int, int, int, int, int, int]]:
    li0_current = 0x2000
    li1_current = 0x0800
    li0_voltage = 0
    li1_voltage = 0
    lif_voltage = 0
    rows: list[tuple[int, int, int, int, int, int, int, int, int]] = []

    for cycle in range(cycles):
        lif_current_pre = _mixed_readout_current(li0_voltage, li1_voltage)
        next_li0_voltage = _li_q88_step(li0_voltage, li0_current)
        next_li1_voltage = _li_q88_step(li1_voltage, li1_current)
        observed_lif_voltage, lif_spike, next_lif_voltage = _lif_q88_step(
            lif_voltage, lif_current_pre
        )
        lif_current_post = _mixed_readout_current(next_li0_voltage, next_li1_voltage)
        rows.append(
            (
                cycle,
                li0_current,
                li1_current,
                lif_current_pre,
                next_li0_voltage,
                next_li1_voltage,
                lif_current_post,
                observed_lif_voltage,
                lif_spike,
            )
        )
        li0_voltage = next_li0_voltage
        li1_voltage = next_li1_voltage
        lif_voltage = next_lif_voltage

    return rows


def _aer_fixed_point_reference(
    cycles: int,
) -> list[tuple[int, int, int, int, int, int, int, int, int, int]]:
    hidden_voltage = 0
    output_voltage = 0
    hidden_spike = 0
    hidden_current = 4 * ((0x2000 * 0x0020) >> 8)
    rows: list[tuple[int, int, int, int, int, int, int, int, int, int]] = []

    for cycle in range(cycles):
        hidden_pre = 65 if hidden_spike else 0
        output_current_pre = -0x0010 * hidden_pre
        observed_hidden, next_hidden_spike, next_hidden_voltage = _lif_q88_step(
            hidden_voltage, hidden_current
        )
        observed_output, next_output_spike, next_output_voltage = _lif_q88_step(
            output_voltage, output_current_pre
        )
        hidden_post = 65 if next_hidden_spike else 0
        output_current_post = -0x0010 * hidden_post
        rows.append(
            (
                cycle,
                hidden_pre,
                hidden_post,
                hidden_current,
                observed_hidden,
                next_hidden_spike,
                output_current_pre,
                output_current_post,
                observed_output,
                next_output_spike,
            )
        )
        hidden_voltage = next_hidden_voltage
        hidden_spike = next_hidden_spike
        output_voltage = next_output_voltage

    return rows


def _parse_mixed_equivalence_stdout(
    stdout: str,
) -> list[tuple[int, int, int, int, int, int, int, int, int]]:
    rows: list[tuple[int, int, int, int, int, int, int, int, int]] = []
    for line in stdout.splitlines():
        parts = line.split()
        if not parts or parts[0] != "cycle":
            continue
        rows.append(
            (
                int(parts[1]),
                int(parts[3]),
                int(parts[5]),
                int(parts[7]),
                int(parts[9]),
                int(parts[11]),
                int(parts[13]),
                int(parts[15]),
                int(parts[17]),
            )
        )
    return rows


def _parse_direct_equivalence_stdout(stdout: str) -> list[tuple[int, int, int, int, int, int, int]]:
    rows: list[tuple[int, int, int, int, int, int, int]] = []
    for line in stdout.splitlines():
        parts = line.split()
        if not parts or parts[0] != "cycle":
            continue
        rows.append(
            (
                int(parts[1]),
                int(parts[3]),
                int(parts[5]),
                int(parts[7]),
                int(parts[9]),
                int(parts[11]),
                int(parts[13]),
            )
        )
    return rows


def _parse_recurrent_equivalence_stdout(
    stdout: str,
) -> list[tuple[int, int, int, int, int, int, int, int, int, int, int, int, int]]:
    rows: list[tuple[int, int, int, int, int, int, int, int, int, int, int, int, int]] = []
    for line in stdout.splitlines():
        parts = line.split()
        if not parts or parts[0] != "cycle":
            continue
        rows.append(
            (
                int(parts[1]),
                int(parts[3]),
                int(parts[5]),
                int(parts[7]),
                int(parts[9]),
                int(parts[11]),
                int(parts[13]),
                int(parts[15]),
                int(parts[17]),
                int(parts[19]),
                int(parts[21]),
                int(parts[23]),
                int(parts[25]),
            )
        )
    return rows


def _parse_aer_equivalence_stdout(
    stdout: str,
) -> list[tuple[int, int, int, int, int, int, int, int, int, int]]:
    rows: list[tuple[int, int, int, int, int, int, int, int, int, int]] = []
    for line in stdout.splitlines():
        parts = line.split()
        if not parts or parts[0] != "cycle":
            continue
        rows.append(
            (
                int(parts[1]),
                int(parts[3]),
                int(parts[5]),
                int(parts[7]),
                int(parts[9]),
                int(parts[11]),
                int(parts[13]),
                int(parts[15]),
                int(parts[17]),
                int(parts[19]),
            )
        )
    return rows
