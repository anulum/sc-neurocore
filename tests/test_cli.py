# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for sc_neurocore.cli

"""Tests for sc_neurocore.cli."""

import builtins
import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import types
from unittest import mock

import numpy as np
import pytest

from sc_neurocore.cli import _cmd_info, _cmd_studio, _format_engine_status, main
from sc_neurocore.formal import validate_formal_network_report
from sc_neurocore.ir import SCNIR_SCHEMA_VERSION, validate_scnir_dict


def _run_main(*argv: str) -> int:
    with mock.patch("sys.argv", ["sc-neurocore", *argv]):
        return main()


def _fake_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _small_lif_nir_graph():
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


def _nested_single_port_lif_nir_graph():
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


def _nested_multiport_multioutput_lif_nir_graph():
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


def _dense_lif_nir_graph():
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


def _aer_lif_nir_graph():
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


def _recurrent_lif_nir_graph():
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


def _mixed_li_lif_nir_graph():
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


def _mixed_aer_li_lif_nir_graph():
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


def _simulate_single_source_module(module_name: str, verilog: str, tmp_path) -> str:
    return _simulate_source_module(
        module_name,
        verilog,
        testbench=_sobol_source_smoke_testbench(module_name),
        tmp_path=tmp_path,
    )


def _simulate_source_module(module_name: str, verilog: str, *, testbench: str, tmp_path) -> str:
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


def _simulate_manifest_source(out_dir, row: dict[str, object], tmp_path) -> str:
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
    out_dir,
    *,
    module_name: str,
    input_words: int,
    total_neurons: int,
    tmp_path,
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


def _simulate_network_with_testbench(out_dir, *, module_name: str, testbench: str, tmp_path) -> str:
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
    assert denominator > 0
    sign = -1 if numerator < 0 else 1
    return sign * (abs(numerator) // denominator)


def _small_direct_fixed_point_reference(
    cycles: int,
) -> list[tuple[int, int, int, int, int, int, int]]:
    ext0 = 0x0200
    ext1 = 0x0200
    current0 = ((ext0 * 0x0040) >> 8) + _trunc_div(ext1 * -0x0080, 1 << 8)
    current1 = ((ext0 * 0x00C0) >> 8) + ((ext1 * 0x0020) >> 8)
    voltages = [0, 0]
    currents = [current0, current1]
    rows: list[tuple[int, int, int, int, int, int, int]] = []

    for cycle in range(cycles):
        row: list[int] = [cycle]
        for idx, current in enumerate(currents):
            leak = _trunc_div((-voltages[idx]) << 8, 5120)
            drive = _trunc_div(current << 8, 5120)
            next_voltage = max(-32768, min(32767, voltages[idx] + leak + drive))
            spike = int(next_voltage > 256)
            if spike:
                observed_voltage = voltages[idx]
                voltages[idx] = 0
            else:
                observed_voltage = next_voltage
                voltages[idx] = next_voltage
            row.extend([current, observed_voltage, spike])
        rows.append(tuple(row))

    return rows


def _lif_q88_step(voltage: int, current: int) -> tuple[int, int, int]:
    leak = _trunc_div((-voltage) << 8, 5120)
    drive = _trunc_div(current << 8, 5120)
    next_voltage = max(-32768, min(32767, voltage + leak + drive))
    spike = int(next_voltage > 256)
    if spike:
        return voltage, spike, 0
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
        assert next_output_spike in (0, 1)

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


def test_version_flag(capsys):
    rc = _run_main("--version")
    assert rc == 0
    from sc_neurocore import __version__

    assert __version__ in capsys.readouterr().out


def test_info_command(capsys):
    fake_jax = _fake_module("jax", __version__="0.0-test")
    with mock.patch.dict("sys.modules", {"jax": fake_jax}):
        rc = _run_main("info")
    assert rc == 0
    out = capsys.readouterr().out
    assert "sc-neurocore" in out
    assert "Python" in out
    assert "NumPy" in out
    assert "JAX: 0.0-test" in out


def test_no_command_prints_help(capsys):
    rc = _run_main()
    assert rc == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_info_without_rust_engine(capsys):
    with mock.patch.dict("sys.modules", {"sc_neurocore_engine": None}):
        rc = _cmd_info()
    assert rc == 0
    assert "not available" in capsys.readouterr().out


def test_info_reports_engine_version_mismatch(capsys):
    fake = _fake_module(
        "sc_neurocore_engine",
        __version__="0.0.0",
        simd_tier=lambda: "mock-tier",
    )
    with mock.patch.dict("sys.modules", {"sc_neurocore_engine": fake}):
        rc = _cmd_info()
    assert rc == 0
    out = capsys.readouterr().out
    assert "version mismatch" in out
    assert "expected" in out


def test_info_uses_metadata_without_importing_optional_jax(capsys):
    def fake_version(name: str) -> str:
        if name == "jax":
            return "0.0-meta"
        if name == "numpy":
            return "0.0-numpy"
        raise importlib.metadata.PackageNotFoundError(name)

    with (
        mock.patch.dict("sys.modules", {"jax": None}),
        mock.patch("sc_neurocore.cli.importlib.metadata.version", side_effect=fake_version),
    ):
        rc = _cmd_info()
    assert rc == 0
    out = capsys.readouterr().out
    assert "JAX: 0.0-meta" in out


def test_info_ignores_missing_optional_metadata(capsys):
    with (
        mock.patch.dict("sys.modules", {"numpy": None, "jax": None}),
        mock.patch(
            "sc_neurocore.cli.importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError("missing"),
        ),
    ):
        rc = _cmd_info()
    assert rc == 0
    assert "NumPy:" not in capsys.readouterr().out


def test_format_engine_status_without_simd_tier():
    fake = _fake_module("sc_neurocore_engine", __version__="3.13.0")
    with mock.patch.dict("sys.modules", {"sc_neurocore_engine": fake}):
        status = _format_engine_status("3.13.0")
    assert status == "Rust engine: 3.13.0 (unknown)"


def test_format_engine_status_with_broken_simd_tier():
    def explode():
        raise RuntimeError("no simd")

    fake = _fake_module(
        "sc_neurocore_engine",
        __version__="3.13.0",
        simd_tier=explode,
    )
    with mock.patch.dict("sys.modules", {"sc_neurocore_engine": fake}):
        status = _format_engine_status("3.13.0")
    assert status == "Rust engine: 3.13.0 (unknown)"


def test_benchmark_delegates_to_subprocess():
    with mock.patch("subprocess.run") as m:
        m.return_value = mock.Mock(returncode=0)
        rc = _run_main("benchmark")
    assert rc == 0
    m.assert_called_once()


def test_preflight_delegates_to_subprocess():
    with mock.patch("subprocess.run") as m:
        m.return_value = mock.Mock(returncode=0)
        rc = _run_main("preflight")
    assert rc == 0
    m.assert_called_once()


@pytest.mark.skipif(
    not importlib.util.find_spec("uvicorn"),
    reason="uvicorn not installed (studio extra)",
)
def test_studio_launches_uvicorn(capsys):
    with (
        mock.patch("uvicorn.run") as m_uvicorn,
        mock.patch("webbrowser.open") as m_browser,
    ):
        rc = _cmd_studio(port=8001)
    assert rc == 0
    m_uvicorn.assert_called_once()
    m_browser.assert_called_once_with("http://127.0.0.1:8001")


def test_studio_missing_fastapi(capsys):
    real_import = builtins.__import__

    def block_uvicorn(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "uvicorn":
            raise ImportError("No module named 'uvicorn'")
        return real_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=block_uvicorn):
        rc = _cmd_studio(port=8001)
    assert rc == 1
    assert "pip install" in capsys.readouterr().out


def test_studio_command_via_main(capsys):
    with (
        mock.patch("sc_neurocore.cli._cmd_studio", return_value=0) as m_studio,
    ):
        rc = _run_main("studio")
    assert rc == 0
    m_studio.assert_called_once_with(8001)


def test_formal_verify_network_writes_sva_and_report(tmp_path, capsys):
    out_dir = tmp_path / "formal"

    rc = _run_main(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "2",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "16",
        "--max-spikes",
        "3",
        "--output",
        str(out_dir),
    )

    assert rc == 0
    sva_path = out_dir / "dense_lif_frontier_fixture_rate_bound.sv"
    report_path = out_dir / "formal_rate_bound_report.json"
    assert "Formal network verification artifacts written" in capsys.readouterr().out
    assert "a_output0_rate_bound" in sva_path.read_text(encoding="utf-8")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "sc-neurocore.formal-network-rate-bound.v0.1"
    assert report["network"]["name"] == "dense_lif_frontier_fixture"
    assert report["rate_bound"]["window_cycles"] == 16
    assert report["replay"] is None
    assert report["artifacts"]["sva"] == str(sva_path)
    rtl_path = out_dir / "dense_lif_frontier_fixture.v"
    assert report["artifacts"]["rtl"] == str(rtl_path)
    validate_formal_network_report(report, artifact_root=out_dir)
    assert "module dense_lif_frontier_fixture (" in rtl_path.read_text(encoding="utf-8")
    assert "dense_lif_frontier_fixture.v" in (out_dir / "dense_lif_frontier_fixture.sby").read_text(
        encoding="utf-8"
    )


def test_formal_verify_network_replays_safe_trace(tmp_path, capsys):
    trace_path = tmp_path / "safe_trace.json"
    trace_path.write_text("[1, 0, 1, 0, 1, 1]", encoding="utf-8")
    out_dir = tmp_path / "formal"

    rc = _run_main(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "1",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "2",
        "--spike-trace",
        str(trace_path),
        "--output",
        str(out_dir),
    )

    assert rc == 0
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["replay"]["violated"] is False
    assert report["replay"]["cycles_checked"] == 6
    assert "Replay passed" in capsys.readouterr().out


def test_formal_verify_network_replays_unsafe_trace(tmp_path, capsys):
    trace_path = tmp_path / "unsafe_trace.json"
    trace_path.write_text("[1, 0, 1, 1]", encoding="utf-8")
    out_dir = tmp_path / "formal"

    rc = _run_main(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "1",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "2",
        "--spike-trace",
        str(trace_path),
        "--output",
        str(out_dir),
    )

    assert rc == 1
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["replay"]["violated"] is True
    assert report["replay"]["first_violation_cycle"] == 3
    assert report["replay"]["observed_spikes"] == 3
    assert "Replay violation" in capsys.readouterr().out


def test_formal_verify_network_replays_refractory_violation(tmp_path, capsys):
    trace_path = tmp_path / "refractory_unsafe_trace.json"
    trace_path.write_text("[1, 0, 1, 0]", encoding="utf-8")
    out_dir = tmp_path / "formal"

    rc = _run_main(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "1",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "4",
        "--refractory-cycles",
        "3",
        "--spike-trace",
        str(trace_path),
        "--output",
        str(out_dir),
    )

    assert rc == 1
    refractory_path = out_dir / "dense_lif_frontier_fixture_refractory.sv"
    bundle_path = out_dir / "dense_lif_frontier_fixture_formal_bundle.sv"
    assert "a_output0_refractory" in refractory_path.read_text(encoding="utf-8")
    assert "dense_lif_frontier_fixture_refractory_sva" in bundle_path.read_text(encoding="utf-8")
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["refractory"]["refractory_cycles"] == 3
    assert report["refractory_replay"]["violated"] is True
    assert report["refractory_replay"]["first_violation_cycle"] == 2
    assert report["rate_replay"]["violated"] is False
    assert report["artifacts"]["refractory_sva"] == str(refractory_path)
    assert report["artifacts"]["formal_bundle"] == str(bundle_path)
    assert "Refractory violation" in capsys.readouterr().out


def test_formal_verify_network_replays_antagonistic_violation(tmp_path, capsys):
    trace_path = tmp_path / "antagonistic_unsafe_trace.json"
    trace_path.write_text("[[1, 0], [0, 1], [1, 1]]", encoding="utf-8")
    out_dir = tmp_path / "formal"

    rc = _run_main(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "2",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "4",
        "--antagonistic-pair",
        "0,1",
        "--spike-trace",
        str(trace_path),
        "--output",
        str(out_dir),
    )

    assert rc == 1
    antagonistic_path = out_dir / "dense_lif_frontier_fixture_antagonistic.sv"
    bundle_path = out_dir / "dense_lif_frontier_fixture_formal_bundle.sv"
    assert "a_output0_output1_exclusion" in antagonistic_path.read_text(encoding="utf-8")
    assert "dense_lif_frontier_fixture_antagonistic_sva" in bundle_path.read_text(encoding="utf-8")
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["antagonistic_exclusion"]["output_a"] == 0
    assert report["antagonistic_exclusion"]["output_b"] == 1
    assert report["antagonistic_replay"]["violated"] is True
    assert report["antagonistic_replay"]["first_violation_cycle"] == 2
    assert report["rate_replay"]["violated"] is False
    assert report["artifacts"]["antagonistic_sva"] == str(antagonistic_path)
    assert "Antagonistic violation" in capsys.readouterr().out


def test_formal_verify_network_replays_temporal_separation_violation(tmp_path, capsys):
    trace_path = tmp_path / "temporal_unsafe_trace.json"
    trace_path.write_text("[[1, 0], [0, 1], [0, 0]]", encoding="utf-8")
    out_dir = tmp_path / "formal"

    rc = _run_main(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "2",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "4",
        "--temporal-separation",
        "0,1,2",
        "--spike-trace",
        str(trace_path),
        "--output",
        str(out_dir),
    )

    assert rc == 1
    temporal_path = out_dir / "dense_lif_frontier_fixture_temporal_separation.sv"
    bundle_path = out_dir / "dense_lif_frontier_fixture_formal_bundle.sv"
    assert "a_output0_output1_temporal_separation" in temporal_path.read_text(encoding="utf-8")
    assert "dense_lif_frontier_fixture_temporal_separation_sva" in bundle_path.read_text(
        encoding="utf-8"
    )
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["temporal_separation"]["output_a"] == 0
    assert report["temporal_separation"]["output_b"] == 1
    assert report["temporal_separation"]["separation_cycles"] == 2
    assert report["temporal_replay"]["violated"] is True
    assert report["temporal_replay"]["first_violation_cycle"] == 1
    assert report["artifacts"]["temporal_sva"] == str(temporal_path)
    assert "Temporal separation violation" in capsys.readouterr().out


def test_formal_verify_network_replays_population_coactivation_violation(tmp_path, capsys):
    trace_path = tmp_path / "population_unsafe_trace.json"
    trace_path.write_text("[[1, 0, 1], [0, 1, 0]]", encoding="utf-8")
    out_dir = tmp_path / "formal"

    rc = _run_main(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "3",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "4",
        "--coactivation-cap",
        "1",
        "--spike-trace",
        str(trace_path),
        "--output",
        str(out_dir),
    )

    assert rc == 1
    population_path = out_dir / "dense_lif_frontier_fixture_population_coactivation.sv"
    bundle_path = out_dir / "dense_lif_frontier_fixture_formal_bundle.sv"
    assert "a_population_coactivation_cap" in population_path.read_text(encoding="utf-8")
    assert "dense_lif_frontier_fixture_population_coactivation_sva" in bundle_path.read_text(
        encoding="utf-8"
    )
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["population_coactivation"]["max_active_outputs"] == 1
    assert report["population_replay"]["violated"] is True
    assert report["population_replay"]["first_violation_cycle"] == 0
    assert report["population_replay"]["observed_active_outputs"] == 2
    assert report["rate_replay"]["violated"] is False
    assert report["artifacts"]["population_sva"] == str(population_path)
    assert "Population coactivation violation" in capsys.readouterr().out


def test_formal_verify_network_replays_population_silence_violation(tmp_path, capsys):
    trace_path = tmp_path / "population_silence_unsafe_trace.json"
    trace_path.write_text("[[1, 1, 0], [0, 0, 0], [0, 1, 0]]", encoding="utf-8")
    out_dir = tmp_path / "formal"

    rc = _run_main(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "3",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "4",
        "--population-silence",
        "2,2",
        "--spike-trace",
        str(trace_path),
        "--output",
        str(out_dir),
    )

    assert rc == 1
    silence_path = out_dir / "dense_lif_frontier_fixture_population_silence.sv"
    bundle_path = out_dir / "dense_lif_frontier_fixture_formal_bundle.sv"
    assert "a_population_silence_after_coactivation" in silence_path.read_text(encoding="utf-8")
    assert "dense_lif_frontier_fixture_population_silence_sva" in bundle_path.read_text(
        encoding="utf-8"
    )
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["population_silence"]["trigger_active_outputs"] == 2
    assert report["population_silence"]["silence_cycles"] == 2
    assert report["population_silence_replay"]["violated"] is True
    assert report["population_silence_replay"]["first_violation_cycle"] == 2
    assert report["population_silence_replay"]["trigger_cycle"] == 0
    assert report["artifacts"]["population_silence_sva"] == str(silence_path)
    assert "Population silence violation" in capsys.readouterr().out


def test_formal_verify_network_replays_population_inactivity_violation(tmp_path, capsys):
    trace_path = tmp_path / "population_inactivity_unsafe_trace.json"
    trace_path.write_text("[[0, 0], [1, 0], [0, 0], [0, 0], [0, 0]]", encoding="utf-8")
    out_dir = tmp_path / "formal"

    rc = _run_main(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "2",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "8",
        "--max-spikes",
        "8",
        "--population-inactivity",
        "2",
        "--spike-trace",
        str(trace_path),
        "--output",
        str(out_dir),
    )

    assert rc == 1
    inactivity_path = out_dir / "dense_lif_frontier_fixture_population_inactivity.sv"
    bundle_path = out_dir / "dense_lif_frontier_fixture_formal_bundle.sv"
    assert "a_population_inactivity_bound" in inactivity_path.read_text(encoding="utf-8")
    assert "dense_lif_frontier_fixture_population_inactivity_sva" in bundle_path.read_text(
        encoding="utf-8"
    )
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["population_inactivity"]["max_silent_cycles"] == 2
    assert report["population_inactivity_replay"]["violated"] is True
    assert report["population_inactivity_replay"]["first_violation_cycle"] == 4
    assert report["population_inactivity_replay"]["observed_silent_cycles"] == 3
    assert report["artifacts"]["population_inactivity_sva"] == str(inactivity_path)
    assert "Population inactivity violation" in capsys.readouterr().out


def test_formal_verify_network_rejects_missing_action(capsys):
    rc = _run_main("formal")

    assert rc == 1
    assert "formal verify-network" in capsys.readouterr().out


def test_formal_verify_network_records_missing_symbiyosys(tmp_path, capsys):
    out_dir = tmp_path / "formal"

    with mock.patch("sc_neurocore.cli.shutil.which", return_value=None):
        rc = _run_main(
            "formal",
            "verify-network",
            "--module-name",
            "dense_lif_frontier_fixture",
            "--input-width",
            "3",
            "--output-width",
            "1",
            "--state-width",
            "16",
            "--output-index",
            "0",
            "--window-cycles",
            "4",
            "--max-spikes",
            "2",
            "--run-symbiyosys",
            "--output",
            str(out_dir),
        )

    assert rc == 0
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["symbiyosys"]["requested"] is True
    assert report["symbiyosys"]["status"] == "tool_unavailable"
    assert report["symbiyosys"]["returncode"] is None
    assert "SymbiYosys unavailable" in capsys.readouterr().out
    assert (out_dir / "dense_lif_frontier_fixture.sby").exists()


def test_formal_verify_network_runs_symbiyosys_when_available(tmp_path, capsys):
    out_dir = tmp_path / "formal"
    completed = subprocess.CompletedProcess(
        args=["/usr/bin/sby", "-f", str(out_dir / "dense_lif_frontier_fixture.sby")],
        returncode=0,
        stdout="PASS\n",
        stderr="",
    )

    with (
        mock.patch("sc_neurocore.cli.shutil.which", return_value="/usr/bin/sby"),
        mock.patch("sc_neurocore.cli.subprocess.run", return_value=completed) as m_run,
    ):
        rc = _run_main(
            "formal",
            "verify-network",
            "--module-name",
            "dense_lif_frontier_fixture",
            "--input-width",
            "3",
            "--output-width",
            "1",
            "--state-width",
            "16",
            "--output-index",
            "0",
            "--window-cycles",
            "4",
            "--max-spikes",
            "2",
            "--run-symbiyosys",
            "--output",
            str(out_dir),
        )

    assert rc == 0
    m_run.assert_called_once()
    assert m_run.call_args.args[0] == [
        "/usr/bin/sby",
        "-f",
        str(out_dir / "dense_lif_frontier_fixture.sby"),
    ]
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["symbiyosys"]["status"] == "passed"
    assert report["symbiyosys"]["returncode"] == 0
    assert report["symbiyosys"]["stdout"] == "PASS\n"
    assert "SymbiYosys passed" in capsys.readouterr().out


def test_formal_verify_network_returns_nonzero_on_symbiyosys_failure(tmp_path, capsys):
    out_dir = tmp_path / "formal"
    completed = subprocess.CompletedProcess(
        args=["/usr/bin/sby", "-f", str(out_dir / "dense_lif_frontier_fixture.sby")],
        returncode=1,
        stdout="FAIL\n",
        stderr="assert failed\n",
    )

    with (
        mock.patch("sc_neurocore.cli.shutil.which", return_value="/usr/bin/sby"),
        mock.patch("sc_neurocore.cli.subprocess.run", return_value=completed),
    ):
        rc = _run_main(
            "formal",
            "verify-network",
            "--module-name",
            "dense_lif_frontier_fixture",
            "--input-width",
            "3",
            "--output-width",
            "1",
            "--state-width",
            "16",
            "--output-index",
            "0",
            "--window-cycles",
            "4",
            "--max-spikes",
            "2",
            "--run-symbiyosys",
            "--output",
            str(out_dir),
        )

    assert rc == 1
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["symbiyosys"]["status"] == "failed"
    assert report["symbiyosys"]["returncode"] == 1
    assert report["symbiyosys"]["stderr"] == "assert failed\n"
    assert "SymbiYosys failed" in capsys.readouterr().out


def test_formal_verify_network_rejects_invalid_formal_depth(tmp_path, capsys):
    rc = _run_main(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "1",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "2",
        "--formal-depth",
        "0",
        "--output",
        str(tmp_path / "formal"),
    )

    assert rc == 1
    assert "formal-depth" in capsys.readouterr().out


def test_formal_verify_network_rejects_negative_refractory_cycles(tmp_path, capsys):
    rc = _run_main(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "1",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "2",
        "--refractory-cycles",
        "-1",
        "--output",
        str(tmp_path / "formal"),
    )

    assert rc == 1
    assert "refractory-cycles" in capsys.readouterr().out


def test_formal_verify_network_rejects_non_positive_population_inactivity(tmp_path, capsys):
    rc = _run_main(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "1",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "2",
        "--population-inactivity",
        "0",
        "--output",
        str(tmp_path / "formal"),
    )

    assert rc == 1
    assert "population-inactivity" in capsys.readouterr().out
    assert not (tmp_path / "formal").exists()


def test_formal_verify_network_rejects_negative_coactivation_cap(tmp_path, capsys):
    rc = _run_main(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "2",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "2",
        "--coactivation-cap",
        "-1",
        "--output",
        str(tmp_path / "formal"),
    )

    assert rc == 1
    assert "coactivation-cap" in capsys.readouterr().out
    assert not (tmp_path / "formal").exists()


def test_formal_verify_network_rejects_non_positive_temporal_separation(tmp_path, capsys):
    rc = _run_main(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "2",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "2",
        "--temporal-separation",
        "0,1,0",
        "--output",
        str(tmp_path / "formal"),
    )

    assert rc == 1
    assert "temporal-separation" in capsys.readouterr().out
    assert not (tmp_path / "formal").exists()


def test_formal_verify_network_rejects_non_positive_population_silence(tmp_path, capsys):
    rc = _run_main(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "2",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "2",
        "--population-silence",
        "0,2",
        "--output",
        str(tmp_path / "formal"),
    )

    assert rc == 1
    assert "population-silence" in capsys.readouterr().out
    assert not (tmp_path / "formal").exists()


# ---------------------------------------------------------------------------
# Deploy command
# ---------------------------------------------------------------------------


class TestDeployCommand:
    """Tests for `sc-neurocore deploy ...` and the underlying _cmd_deploy."""

    def test_deploy_without_model_arg_returns_1(self, capsys):
        """`sc-neurocore deploy` with no model argument prints usage and exits 1."""
        rc = _run_main("deploy")
        assert rc == 1
        out = capsys.readouterr().out
        assert "deploy requires a model file" in out

    def test_deploy_unsupported_extension_returns_1(self, capsys, tmp_path):
        """A model with an unsupported extension exits 1 with a clear message."""
        from sc_neurocore.cli import _cmd_deploy

        bogus = tmp_path / "model.onnx"
        bogus.write_bytes(b"\x00")
        rc = _cmd_deploy(str(bogus), "ice40", str(tmp_path / "out"), dt=1.0, bitstream_length=256)
        assert rc == 1
        assert "unsupported file format" in capsys.readouterr().out

    def test_deploy_pytorch_writes_verilog_and_hdl_dir(self, tmp_path, capsys):
        """A `.pt` checkpoint with two Linear layers compiles to a Verilog project."""
        torch = pytest.importorskip("torch")

        from sc_neurocore.cli import _cmd_deploy

        # Build a minimal 2-layer Linear stack and save its state_dict
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
        )
        ckpt = tmp_path / "tiny.pt"
        torch.save(model.state_dict(), ckpt)
        checkpoint_sha256 = hashlib.sha256(ckpt.read_bytes()).hexdigest()

        out_dir = tmp_path / "deploy_out"
        rc = _cmd_deploy(
            str(ckpt),
            "ice40",
            str(out_dir),
            dt=1.0,
            bitstream_length=64,
            checkpoint_sha256=checkpoint_sha256,
        )
        assert rc == 0

        # Generated SystemVerilog
        sv = out_dir / "sc_deploy_lif.sv"
        assert sv.exists() and sv.stat().st_size > 0

        # Makefile (Yosys flow → ice40)
        assert (out_dir / "Makefile").exists()

        # README in the deploy dir
        readme = (out_dir / "README.md").read_text()
        assert "ice40" in readme
        power_model = json.loads((out_dir / "power_thermal_model.json").read_text())
        assert power_model["source_mode"] == "pre_silicon_estimate"
        assert power_model["workload"]["layer_sizes"] == [[4, 8], [8, 2]]

    def test_deploy_emits_vivado_tcl_for_artix7(self, tmp_path):
        """artix7 target should emit a project.tcl, not a Makefile."""
        torch = pytest.importorskip("torch")

        from sc_neurocore.cli import _cmd_deploy

        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        ckpt = tmp_path / "tiny.pt"
        torch.save(model.state_dict(), ckpt)
        checkpoint_sha256 = hashlib.sha256(ckpt.read_bytes()).hexdigest()

        out_dir = tmp_path / "vivado_out"
        rc = _cmd_deploy(
            str(ckpt),
            "artix7",
            str(out_dir),
            dt=1.0,
            bitstream_length=64,
            checkpoint_sha256=checkpoint_sha256,
        )
        assert rc == 0
        assert (out_dir / "project.tcl").exists()
        assert not (out_dir / "Makefile").exists()
        # README mentions artix7
        assert "artix7" in (out_dir / "README.md").read_text()

    def test_deploy_via_main_dispatcher(self, tmp_path):
        """`sc-neurocore deploy model.pt --target ice40 -o ...` end-to-end via main()."""
        torch = pytest.importorskip("torch")

        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        ckpt = tmp_path / "m.pt"
        torch.save(model.state_dict(), ckpt)
        checkpoint_sha256 = hashlib.sha256(ckpt.read_bytes()).hexdigest()

        out = tmp_path / "deployed"
        rc = _run_main(
            "deploy",
            str(ckpt),
            "--target",
            "ice40",
            "--checkpoint-sha256",
            checkpoint_sha256,
            "-o",
            str(out),
        )
        assert rc == 0
        assert (out / "sc_deploy_lif.sv").exists()

    def test_deploy_pytorch_without_sha256_fails_closed(self, tmp_path, capsys):
        """Deploy refuses PyTorch checkpoints without an explicit digest."""
        torch = pytest.importorskip("torch")

        from sc_neurocore.cli import _cmd_deploy

        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        ckpt = tmp_path / "nohash.pt"
        torch.save(model.state_dict(), ckpt)

        rc = _cmd_deploy(str(ckpt), "ice40", str(tmp_path / "out"), dt=1.0, bitstream_length=64)
        assert rc == 1
        assert "--checkpoint-sha256" in capsys.readouterr().out

    @pytest.mark.parametrize("bad_digest", ["abc123", "g" * 64])
    def test_deploy_pytorch_invalid_sha256_fails_closed(self, tmp_path, capsys, bad_digest):
        torch = pytest.importorskip("torch")

        from sc_neurocore.cli import _cmd_deploy

        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        ckpt = tmp_path / "badsha.pt"
        torch.save(model.state_dict(), ckpt)

        rc = _cmd_deploy(
            str(ckpt),
            "ice40",
            str(tmp_path / "out"),
            dt=1.0,
            bitstream_length=64,
            checkpoint_sha256=bad_digest,
        )
        assert rc == 1
        assert "64 hexadecimal characters" in capsys.readouterr().out

    def test_deploy_via_main_invalid_sha256_fails_closed(self, tmp_path, capsys):
        torch = pytest.importorskip("torch")

        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        ckpt = tmp_path / "badsha_main.pt"
        torch.save(model.state_dict(), ckpt)

        rc = _run_main(
            "deploy",
            str(ckpt),
            "--target",
            "ice40",
            "--checkpoint-sha256",
            "abc123",
            "-o",
            str(tmp_path / "out"),
        )
        assert rc == 1
        assert "64 hexadecimal characters" in capsys.readouterr().out

    def test_deploy_rejects_non_tensor_state_entries(self, tmp_path, capsys):
        torch = pytest.importorskip("torch")

        from sc_neurocore.cli import _cmd_deploy

        ckpt = tmp_path / "bad_state.pt"
        torch.save({"layer.weight": [1, 2, 3]}, ckpt)
        digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        rc = _cmd_deploy(
            str(ckpt),
            "ice40",
            str(tmp_path / "out"),
            dt=1.0,
            bitstream_length=64,
            checkpoint_sha256=digest,
        )
        assert rc == 1
        assert "entries must be tensors" in capsys.readouterr().out

    def test_deploy_rejects_checkpoint_without_dense_weights(self, tmp_path, capsys):
        torch = pytest.importorskip("torch")

        from sc_neurocore.cli import _cmd_deploy

        ckpt = tmp_path / "conv_only.pt"
        torch.save({"conv.weight": torch.randn(8, 1, 3, 3)}, ckpt)
        digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        rc = _cmd_deploy(
            str(ckpt),
            "ice40",
            str(tmp_path / "out"),
            dt=1.0,
            bitstream_length=64,
            checkpoint_sha256=digest,
        )
        assert rc == 1
        assert "does not contain any 2D dense '.weight' tensors" in capsys.readouterr().out

    def test_deploy_rejects_non_floating_dense_weights(self, tmp_path, capsys):
        torch = pytest.importorskip("torch")

        from sc_neurocore.cli import _cmd_deploy

        ckpt = tmp_path / "int_dense.pt"
        torch.save({"layer.weight": torch.ones(4, 4, dtype=torch.int64)}, ckpt)
        digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        rc = _cmd_deploy(
            str(ckpt),
            "ice40",
            str(tmp_path / "out"),
            dt=1.0,
            bitstream_length=64,
            checkpoint_sha256=digest,
        )
        assert rc == 1
        assert "must use floating-point dtype" in capsys.readouterr().out

    def test_deploy_rejects_non_finite_dense_weights(self, tmp_path, capsys):
        torch = pytest.importorskip("torch")

        from sc_neurocore.cli import _cmd_deploy

        bad_weight = torch.randn(4, 4, dtype=torch.float32)
        bad_weight[0, 0] = torch.nan
        ckpt = tmp_path / "nan_dense.pt"
        torch.save({"layer.weight": bad_weight}, ckpt)
        digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        rc = _cmd_deploy(
            str(ckpt),
            "ice40",
            str(tmp_path / "out"),
            dt=1.0,
            bitstream_length=64,
            checkpoint_sha256=digest,
        )
        assert rc == 1
        assert "contains non-finite values" in capsys.readouterr().out

    def test_deploy_rejects_excessive_dense_parameter_count(self, tmp_path, capsys, monkeypatch):
        torch = pytest.importorskip("torch")

        from sc_neurocore import cli as cli_mod
        from sc_neurocore.cli import _cmd_deploy

        monkeypatch.setattr(cli_mod, "_MAX_DEPLOY_DENSE_PARAMS", 4)
        ckpt = tmp_path / "too_many_dense_params.pt"
        torch.save({"layer.weight": torch.randn(3, 3, dtype=torch.float32)}, ckpt)
        digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        rc = _cmd_deploy(
            str(ckpt),
            "ice40",
            str(tmp_path / "out"),
            dt=1.0,
            bitstream_length=64,
            checkpoint_sha256=digest,
        )
        assert rc == 1
        assert "dense parameter count exceeds safety limit" in capsys.readouterr().out

    def test_deploy_rejects_incompatible_dense_weight_chain(self, tmp_path, capsys):
        torch = pytest.importorskip("torch")

        from sc_neurocore.cli import _cmd_deploy

        ckpt = tmp_path / "bad_chain.pt"
        # layer_b expects 5 inputs, but layer_a outputs 3 -> incompatible chain
        torch.save(
            {
                "layer_a.weight": torch.randn(3, 4, dtype=torch.float32),
                "layer_b.weight": torch.randn(2, 5, dtype=torch.float32),
            },
            ckpt,
        )
        digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        rc = _cmd_deploy(
            str(ckpt),
            "ice40",
            str(tmp_path / "out"),
            dt=1.0,
            bitstream_length=64,
            checkpoint_sha256=digest,
        )
        assert rc == 1
        assert "not composition-compatible" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# NIR silicon mapping command
# ---------------------------------------------------------------------------


class TestMapNirCommand:
    """Tests for `sc-neurocore map-nir ...`."""

    def test_map_nir_rejects_non_nir_extension(self, tmp_path, capsys):
        from sc_neurocore.cli import _cmd_map_nir

        rc = _cmd_map_nir(
            str(tmp_path / "model.pt"),
            str(tmp_path / "out"),
            "loihi2",
            dt=1.0,
            bitstream_length=256,
        )

        assert rc == 1
        assert "supports .nir files only" in capsys.readouterr().out

    def test_map_nir_writes_report_with_mocked_nir_import(self, tmp_path, capsys):
        from sc_neurocore.cli import _cmd_map_nir

        nir_path = tmp_path / "model.nir"
        nir_path.write_bytes(b"")
        fake_graph = mock.MagicMock()
        fake_network = types.SimpleNamespace(
            nodes={
                "input": {"node_type": "Input", "shape": (2,)},
                "dense": {"node_type": "Linear", "weight": (3, 2)},
                "output": {"node_type": "Output", "shape": (3,)},
            },
            topo_order=["input", "dense", "output"],
            edges=[("input", "dense"), ("dense", "output")],
        )
        fake_nir = _fake_module("nir", read=mock.MagicMock(return_value=fake_graph))

        with (
            mock.patch.dict("sys.modules", {"nir": fake_nir}),
            mock.patch("sc_neurocore.nir_bridge.from_nir", return_value=fake_network),
        ):
            rc = _cmd_map_nir(
                str(nir_path),
                str(tmp_path / "mapping"),
                "loihi2,spinnaker2,akida",
                dt=0.5,
                bitstream_length=256,
            )

        report = json.loads(
            (tmp_path / "mapping" / "nir_silicon_mapping_report.json").read_text(encoding="utf-8")
        )
        assert rc == 0
        assert [target["target_id"] for target in report["targets"]] == [
            "loihi2",
            "spinnaker2",
            "akida",
        ]
        assert report["targets"][0]["summary"]["estimated_synapses"] == 6
        assert "NIR silicon mapping report generated" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# NIR FPGA compilation command
# ---------------------------------------------------------------------------


class TestCompileNirCommand:
    """Tests for `sc-neurocore compile-nir ...` exported artefacts."""

    def test_compile_nir_writes_scnir_source_bundle_and_simulates_source(self, tmp_path, capsys):
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "fixture.nir"
        nir.write(str(model_path), _small_lif_nir_graph())

        out_dir = tmp_path / "compiled"
        rc = _run_main(
            "compile-nir",
            str(model_path),
            "--module-name",
            "fixture_net",
            "--T",
            "512",
            "--source-kind",
            "sobol",
            "--base-seed",
            "66",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        assert (out_dir / "fixture_net.v").exists()
        assert (out_dir / "sc_nir_lif.v").exists()
        assert (out_dir / "sc_nir_weight_rom.v").exists()

        manifest_path = out_dir / "scnir_source_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == "sc-neurocore.scnir.hdl-sources.v0.2"
        assert len(manifest["sources"]) == 2

        first = manifest["sources"][0]
        assert first["stream_id"] == "pop.lif.spike"
        assert first["source_kind"] == "sobol16"
        assert first["seed"] == 66
        assert first["bitstream_length"] == 512
        assert first["sobol_dimension"] == 1

        source_path = out_dir / f"{first['module_name']}.v"
        source_verilog = source_path.read_text(encoding="utf-8")
        assert "module " + first["module_name"] in source_verilog
        assert "localparam [15:0] SEED = 16'h0042;" in source_verilog

        stdout = _simulate_single_source_module(first["module_name"], source_verilog, tmp_path)
        assert "sample0 8042 1 1" in stdout
        assert f"{first['module_name']}.v" in capsys.readouterr().out

    def test_compile_nir_folded_interconnect_reports_metrics(self, tmp_path, capsys):
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "folded_fixture.nir"
        nir.write(str(model_path), _small_lif_nir_graph())

        out_dir = tmp_path / "folded"
        rc = _run_main(
            "compile-nir",
            str(model_path),
            "--module-name",
            "folded_net",
            "--interconnect",
            "folded",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        out = capsys.readouterr().out
        assert "Interconnect: folded" in out
        assert "Folded datapath: 1 population(s), 1 PE" in out
        assert "collapses 2 direct neuron instances" in out
        # The folded resource counts are mapped onto a pre-synthesis area estimate.
        assert "Folded area (~est. ice40)" in out
        assert "LUTs" in out and "DSP" in out
        # The shared datapath PE module is emitted alongside the top.
        assert (out_dir / "folded_net.v").exists()
        assert (out_dir / "sc_nir_lif_pe.v").exists()
        # The source-handoff manifest keeps its versioned schema (no metrics pollution).
        manifest = json.loads((out_dir / "scnir_source_manifest.json").read_text(encoding="utf-8"))
        assert manifest["interconnect"] == "folded"
        assert "folded_metrics" not in manifest
        # Folded metrics are persisted to their own machine-readable artefact.
        metrics = json.loads((out_dir / "folded_metrics.json").read_text(encoding="utf-8"))
        assert metrics["pe_instances"] == 1
        assert metrics["neurons"] == 2
        assert metrics["direct_neuron_instances"] == 2
        assert metrics["shared_multipliers"] == 2
        assert metrics["populations"] == 1
        # The pre-synthesis area estimate is persisted alongside the raw counts.
        area = metrics["area_estimate"]
        assert area["target"] == "ice40"
        assert area["latency_cycles"] == metrics["cycles_per_tick"]
        assert area["total_luts"] > 0
        assert "fits_on_target" in area

    def test_compile_nir_writes_valid_dense_scnir_document(self, tmp_path, capsys):
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "dense_fixture.nir"
        nir.write(str(model_path), _dense_lif_nir_graph())

        out_dir = tmp_path / "dense_compiled"
        rc = _run_main(
            "compile-nir",
            str(model_path),
            "--module-name",
            "dense_fixture_net",
            "--T",
            "768",
            "--source-kind",
            "lfsr",
            "--base-seed",
            "101",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        scnir_path = out_dir / "scnir_document.json"
        payload = json.loads(scnir_path.read_text(encoding="utf-8"))
        validate_scnir_dict(payload)
        assert payload["schema_version"] == SCNIR_SCHEMA_VERSION
        assert {stream["bitstream_length"] for stream in payload["streams"]} == {768}
        assert {stream["stream_id"] for stream in payload["streams"]} == {
            "pop.lif1.spike",
            "pop.lif2.spike",
            "conn.input_to_lif1.weight",
            "conn.lif1_to_lif2.weight",
        }
        assert {stream["signal_kind"] for stream in payload["streams"]} == {"spike", "weight"}
        manifest = json.loads((out_dir / "scnir_source_manifest.json").read_text(encoding="utf-8"))
        assert [row["stream_id"] for row in manifest["sources"]] == [
            stream["stream_id"] for stream in payload["streams"]
        ]
        assert "scnir_document.json" in capsys.readouterr().out

    def test_compile_nir_records_aer_interconnect_in_manifest(self, tmp_path, capsys):
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "aer_fixture.nir"
        nir.write(str(model_path), _aer_lif_nir_graph())

        out_dir = tmp_path / "aer_compiled"
        rc = _run_main(
            "compile-nir",
            str(model_path),
            "--module-name",
            "aer_fixture_net",
            "--T",
            "384",
            "--source-kind",
            "lfsr",
            "--base-seed",
            "11",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        top_module = (out_dir / "aer_fixture_net.v").read_text(encoding="utf-8")
        assert "Interconnect: weighted event routing" in top_module
        assert "localparam integer AER_SRC_COUNT = 67;" in top_module
        assert "$signed({{(ACC_WIDTH - DATA_WIDTH){Q_MAX[DATA_WIDTH - 1]}}, Q_MAX})" in top_module
        assert "$signed({{(ACC_WIDTH - DATA_WIDTH){Q_MIN[DATA_WIDTH - 1]}}, Q_MIN})" in top_module

        manifest = json.loads((out_dir / "scnir_source_manifest.json").read_text(encoding="utf-8"))
        assert manifest["interconnect"] == "aer"
        assert manifest["q_format"] == "Q8.8"
        assert manifest["total_neurons"] == 67
        assert manifest["total_synapses"] == 390
        assert manifest["scnir_stream_count"] == 4

        payload = json.loads((out_dir / "scnir_document.json").read_text(encoding="utf-8"))
        validate_scnir_dict(payload)
        assert [row["stream_id"] for row in manifest["sources"]] == [
            stream["stream_id"] for stream in payload["streams"]
        ]
        assert "Interconnect: aer" in capsys.readouterr().out

    def test_compile_nir_records_mixed_signal_summary_in_manifest(self, tmp_path, capsys):
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "mixed_signal_fixture.nir"
        nir.write(str(model_path), _mixed_li_lif_nir_graph())

        out_dir = tmp_path / "mixed_signal_compiled"
        rc = _run_main(
            "compile-nir",
            str(model_path),
            "--module-name",
            "mixed_signal_fixture_net",
            "--T",
            "640",
            "--source-kind",
            "sobol",
            "--base-seed",
            "91",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        payload = json.loads((out_dir / "scnir_document.json").read_text(encoding="utf-8"))
        validate_scnir_dict(payload)
        assert {stream["stream_id"]: stream["signal_kind"] for stream in payload["streams"]} == {
            "pop.li.state": "analogue_state",
            "pop.lif.spike": "spike",
            "conn.input_to_li.weight": "weight",
            "conn.li_to_lif.weight": "weight",
        }

        manifest = json.loads((out_dir / "scnir_source_manifest.json").read_text(encoding="utf-8"))
        assert manifest["scnir_signal_kinds"] == {
            "analogue_state": 1,
            "spike": 1,
            "weight": 2,
        }
        assert [row["signal_kind"] for row in manifest["sources"]] == [
            stream["signal_kind"] for stream in payload["streams"]
        ]
        assert "Interconnect: direct" in capsys.readouterr().out

    def test_compile_nir_records_mixed_aer_routing_summary(self, tmp_path, capsys):
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "mixed_aer_fixture.nir"
        nir.write(str(model_path), _mixed_aer_li_lif_nir_graph())

        out_dir = tmp_path / "mixed_aer_compiled"
        rc = _run_main(
            "compile-nir",
            str(model_path),
            "--module-name",
            "mixed_aer_fixture_net",
            "--T",
            "896",
            "--source-kind",
            "lfsr",
            "--base-seed",
            "123",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        top_module = (out_dir / "mixed_aer_fixture_net.v").read_text(encoding="utf-8")
        assert "Interconnect: weighted event routing" in top_module
        assert "localparam integer AER_SRC_COUNT = 66;" in top_module
        assert "p0_n0_v * 16'sh0080" in top_module
        assert "p0_n1_v * 16'shffc0" in top_module

        payload = json.loads((out_dir / "scnir_document.json").read_text(encoding="utf-8"))
        validate_scnir_dict(payload)
        assert {stream["stream_id"]: stream["signal_kind"] for stream in payload["streams"]} == {
            "pop.li.state": "analogue_state",
            "pop.lif1.spike": "spike",
            "pop.lif2.spike": "spike",
            "conn.input_to_li.weight": "weight",
            "conn.li_to_lif1.weight": "weight",
            "conn.lif1_to_lif2.weight": "weight",
        }

        manifest = json.loads((out_dir / "scnir_source_manifest.json").read_text(encoding="utf-8"))
        assert manifest["interconnect"] == "aer"
        assert manifest["scnir_signal_kinds"] == {
            "analogue_state": 1,
            "spike": 2,
            "weight": 3,
        }
        assert manifest["scnir_signal_routes"] == {
            "analogue_state": "direct_mac",
            "spike": "weighted_event_aer",
            "weight": "stochastic_source_module",
        }
        assert "Interconnect: aer" in capsys.readouterr().out

    def test_compile_nir_cosimulates_sources_across_interconnects(self, tmp_path, capsys):
        nir = pytest.importorskip("nir")
        cases = [
            (
                "direct_sobol",
                _small_lif_nir_graph(),
                "sobol",
                66,
                "direct",
                "pop.lif.spike",
            ),
            (
                "aer_lfsr",
                _aer_lif_nir_graph(),
                "lfsr",
                11,
                "aer",
                "pop.lif1.spike",
            ),
            (
                "recurrent_lfsr",
                _recurrent_lif_nir_graph(),
                "lfsr",
                41,
                "direct",
                "conn.lif_to_lif.weight",
            ),
        ]

        for name, graph, source_kind, base_seed, expected_interconnect, stream_id in cases:
            model_path = tmp_path / f"{name}.nir"
            nir.write(str(model_path), graph)
            out_dir = tmp_path / f"{name}_compiled"

            rc = _run_main(
                "compile-nir",
                str(model_path),
                "--module-name",
                f"{name}_net",
                "--T",
                "512",
                "--source-kind",
                source_kind,
                "--base-seed",
                str(base_seed),
                "-o",
                str(out_dir),
            )

            assert rc == 0
            manifest = json.loads(
                (out_dir / "scnir_source_manifest.json").read_text(encoding="utf-8")
            )
            assert manifest["interconnect"] == expected_interconnect
            row = next(item for item in manifest["sources"] if item["stream_id"] == stream_id)
            assert row["source_kind"] == f"{source_kind}16"
            sim_dir = tmp_path / f"{name}_sim"
            sim_dir.mkdir()
            _simulate_manifest_source(out_dir, row, sim_dir)

        output = capsys.readouterr().out
        assert "Interconnect: direct" in output
        assert "Interconnect: aer" in output

    def test_compile_nir_cosimulates_complete_networks_across_interconnects(self, tmp_path, capsys):
        nir = pytest.importorskip("nir")
        cases = [
            (
                "direct_top_sobol",
                _small_lif_nir_graph(),
                "sobol",
                66,
                2,
                2,
                "direct",
            ),
            (
                "aer_top_lfsr",
                _aer_lif_nir_graph(),
                "lfsr",
                11,
                4,
                67,
                "aer",
            ),
            (
                "recurrent_top_lfsr",
                _recurrent_lif_nir_graph(),
                "lfsr",
                41,
                2,
                2,
                "direct",
            ),
        ]

        for name, graph, source_kind, base_seed, input_words, total_neurons, interconnect in cases:
            model_path = tmp_path / f"{name}.nir"
            nir.write(str(model_path), graph)
            out_dir = tmp_path / f"{name}_compiled"
            module_name = f"{name}_net"

            rc = _run_main(
                "compile-nir",
                str(model_path),
                "--module-name",
                module_name,
                "--T",
                "512",
                "--source-kind",
                source_kind,
                "--base-seed",
                str(base_seed),
                "-o",
                str(out_dir),
            )

            assert rc == 0
            manifest = json.loads(
                (out_dir / "scnir_source_manifest.json").read_text(encoding="utf-8")
            )
            assert manifest["interconnect"] == interconnect
            assert manifest["total_neurons"] == total_neurons

            sim_dir = tmp_path / f"{name}_network_sim"
            sim_dir.mkdir()
            stdout = _simulate_network_bundle(
                out_dir,
                module_name=module_name,
                input_words=input_words,
                total_neurons=total_neurons,
                tmp_path=sim_dir,
            )
            assert f"network_start {total_neurons}" in stdout
            assert f"network_done {module_name}" in stdout

        output = capsys.readouterr().out
        assert "Interconnect: direct" in output
        assert "Interconnect: aer" in output

    def test_compile_nir_direct_network_matches_fixed_point_reference(self, tmp_path):
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "direct_equivalence_fixture.nir"
        nir.write(str(model_path), _small_lif_nir_graph())
        out_dir = tmp_path / "direct_equivalence_compiled"
        module_name = "direct_equivalence_net"

        rc = _run_main(
            "compile-nir",
            str(model_path),
            "--module-name",
            module_name,
            "--T",
            "512",
            "--source-kind",
            "sobol",
            "--base-seed",
            "66",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        stdout = _simulate_network_with_testbench(
            out_dir,
            module_name=module_name,
            testbench=_direct_equivalence_testbench(module_name),
            tmp_path=tmp_path / "direct_equivalence_sim",
        )
        assert _parse_direct_equivalence_stdout(stdout) == _small_direct_fixed_point_reference(8)

    def test_compile_nir_recurrent_network_matches_fixed_point_reference(self, tmp_path):
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "recurrent_equivalence_fixture.nir"
        nir.write(str(model_path), _recurrent_lif_nir_graph())
        out_dir = tmp_path / "recurrent_equivalence_compiled"
        module_name = "recurrent_equivalence_net"

        rc = _run_main(
            "compile-nir",
            str(model_path),
            "--module-name",
            module_name,
            "--T",
            "512",
            "--source-kind",
            "lfsr",
            "--base-seed",
            "41",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        manifest = json.loads((out_dir / "scnir_source_manifest.json").read_text(encoding="utf-8"))
        recurrent_row = next(
            row for row in manifest["sources"] if row["stream_id"] == "conn.lif_to_lif.weight"
        )
        assert recurrent_row["delay_steps"] == 1
        stdout = _simulate_network_with_testbench(
            out_dir,
            module_name=module_name,
            testbench=_recurrent_equivalence_testbench(module_name),
            tmp_path=tmp_path / "recurrent_equivalence_sim",
        )
        assert _parse_recurrent_equivalence_stdout(stdout) == _recurrent_fixed_point_reference(8)

    def test_compile_nir_aer_network_matches_fixed_point_reference(self, tmp_path):
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "aer_equivalence_fixture.nir"
        nir.write(str(model_path), _aer_lif_nir_graph())
        out_dir = tmp_path / "aer_equivalence_compiled"
        module_name = "aer_equivalence_net"

        rc = _run_main(
            "compile-nir",
            str(model_path),
            "--module-name",
            module_name,
            "--T",
            "512",
            "--source-kind",
            "lfsr",
            "--base-seed",
            "11",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        manifest = json.loads((out_dir / "scnir_source_manifest.json").read_text(encoding="utf-8"))
        assert manifest["interconnect"] == "aer"
        assert manifest["total_neurons"] == 67
        stdout = _simulate_network_with_testbench(
            out_dir,
            module_name=module_name,
            testbench=_aer_equivalence_testbench(module_name),
            tmp_path=tmp_path / "aer_equivalence_sim",
        )
        assert _parse_aer_equivalence_stdout(stdout) == _aer_fixed_point_reference(8)

    def test_compile_nir_mixed_network_matches_fixed_point_reference(self, tmp_path):
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "mixed_equivalence_fixture.nir"
        nir.write(str(model_path), _mixed_li_lif_nir_graph())
        out_dir = tmp_path / "mixed_equivalence_compiled"
        module_name = "mixed_equivalence_net"

        rc = _run_main(
            "compile-nir",
            str(model_path),
            "--module-name",
            module_name,
            "--T",
            "512",
            "--source-kind",
            "sobol",
            "--base-seed",
            "91",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        manifest = json.loads((out_dir / "scnir_source_manifest.json").read_text(encoding="utf-8"))
        assert manifest["interconnect"] == "direct"
        assert manifest["scnir_signal_kinds"] == {"analogue_state": 1, "spike": 1, "weight": 2}
        assert manifest["scnir_signal_routes"]["analogue_state"] == "direct_mac"
        stdout = _simulate_network_with_testbench(
            out_dir,
            module_name=module_name,
            testbench=_mixed_equivalence_testbench(module_name),
            tmp_path=tmp_path / "mixed_equivalence_sim",
        )
        assert _parse_mixed_equivalence_stdout(stdout) == _mixed_fixed_point_reference(8)

    def test_compile_nir_can_write_scnir_handoff_audit_report(self, tmp_path):
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "audit_handoff_fixture.nir"
        nir.write(str(model_path), _small_lif_nir_graph())
        out_dir = tmp_path / "audit_handoff_compiled"
        module_name = "audit_handoff_net"

        rc = _run_main(
            "compile-nir",
            str(model_path),
            "--module-name",
            module_name,
            "--T",
            "512",
            "--source-kind",
            "sobol",
            "--base-seed",
            "77",
            "--audit-handoff",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        report = json.loads((out_dir / "scnir_handoff_audit.json").read_text(encoding="utf-8"))
        assert report["status"] == "valid"
        assert report["module_name"] == module_name
        assert report["stream_count"] == 2
        assert report["source_module_count"] == 2

    def test_compile_nir_manifest_and_audit_report_generated_hierarchy(self, tmp_path):
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "nested_hierarchy_fixture.nir"
        nir.write(str(model_path), _nested_single_port_lif_nir_graph())
        out_dir = tmp_path / "nested_hierarchy_compiled"

        rc = _run_main(
            "compile-nir",
            str(model_path),
            "--module-name",
            "nested_hierarchy_net",
            "--T",
            "512",
            "--source-kind",
            "sobol",
            "--base-seed",
            "77",
            "--audit-handoff",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        document = json.loads((out_dir / "scnir_document.json").read_text(encoding="utf-8"))
        validate_scnir_dict(document)
        assert document["hierarchy"] == [
            {
                "instance_id": "subgraph",
                "module_name": "scnir_subgraph",
                "ports": [
                    {
                        "port_name": "weight_0",
                        "direction": "output",
                        "stream_id": "conn.subgraph__input_to_lif.weight",
                        "signal_kind": "weight",
                        "bit_width": 64,
                    }
                ],
            }
        ]

        manifest = json.loads((out_dir / "scnir_source_manifest.json").read_text(encoding="utf-8"))
        assert manifest["scnir_hierarchy_instance_count"] == 1
        assert manifest["scnir_hierarchy_port_count"] == 1
        hierarchy_module = (out_dir / "scnir_subgraph.v").read_text(encoding="utf-8")
        assert "module scnir_subgraph (" in hierarchy_module
        assert "output wire signed [63:0] weight_0" in hierarchy_module
        assert "assign weight_0[0 +: 16] = 16'sh0040;" in hierarchy_module
        assert "assign weight_0[48 +: 16] = 16'sh0020;" in hierarchy_module
        assert "// stream_id: conn.subgraph__input_to_lif.weight" in hierarchy_module

        report = json.loads((out_dir / "scnir_handoff_audit.json").read_text(encoding="utf-8"))
        assert report["status"] == "valid"
        assert report["hierarchy_instance_count"] == 1
        assert report["hierarchy_port_count"] == 1
        assert "scnir_subgraph.v" in report["artefacts"]
        assert report["hierarchy_instances"]["subgraph"]["ports"] == [
            {
                "bit_width": 64,
                "direction": "output",
                "port_name": "weight_0",
                "signal_kind": "weight",
                "stream_id": "conn.subgraph__input_to_lif.weight",
            }
        ]

    def test_compile_nir_audits_exact_multiport_multioutput_hierarchy(
        self,
        tmp_path,
        monkeypatch,
    ):
        nir = pytest.importorskip("nir")
        model_path = tmp_path / "nested_multiport_multioutput_fixture.nir"
        model_path.write_bytes(b"synthetic multi-port fixture")
        monkeypatch.setattr(
            nir,
            "read",
            lambda _path: _nested_multiport_multioutput_lif_nir_graph(),
        )
        out_dir = tmp_path / "nested_multiport_multioutput_compiled"

        rc = _run_main(
            "compile-nir",
            str(model_path),
            "--module-name",
            "nested_multiport_multioutput_net",
            "--T",
            "512",
            "--source-kind",
            "sobol",
            "--base-seed",
            "81",
            "--audit-handoff",
            "-o",
            str(out_dir),
        )

        assert rc == 0
        document = json.loads((out_dir / "scnir_document.json").read_text(encoding="utf-8"))
        validate_scnir_dict(document)
        expected_ports = [
            {
                "port_name": "weight_0",
                "direction": "output",
                "stream_id": "conn.subgraph__a_to_lif_a.weight",
                "signal_kind": "weight",
                "bit_width": 16,
            },
            {
                "port_name": "weight_1",
                "direction": "output",
                "stream_id": "conn.subgraph__b_to_lif_b.weight",
                "signal_kind": "weight",
                "bit_width": 16,
            },
        ]
        assert document["hierarchy"] == [
            {
                "instance_id": "subgraph",
                "module_name": "scnir_subgraph",
                "ports": expected_ports,
            }
        ]

        manifest = json.loads((out_dir / "scnir_source_manifest.json").read_text(encoding="utf-8"))
        assert manifest["scnir_hierarchy_instance_count"] == 1
        assert manifest["scnir_hierarchy_port_count"] == 2
        assert manifest["scnir_stream_count"] == 4
        assert manifest["scnir_external_inputs"] == [
            {"source": "subgraph__a", "offset": 0, "width": 1},
            {"source": "subgraph__b", "offset": 1, "width": 1},
        ]

        top_module = (out_dir / "nested_multiport_multioutput_net.v").read_text(encoding="utf-8")
        assert "ext_input_0 * scnir_subgraph__weight_0" in top_module
        assert "ext_input_1 * scnir_subgraph__weight_1" in top_module
        assert "scnir_subgraph scnir_subgraph_hierarchy_inst (" in top_module
        assert ".weight_0(scnir_subgraph__weight_0)" in top_module
        assert ".weight_1(scnir_subgraph__weight_1)" in top_module
        hierarchy_module = (out_dir / "scnir_subgraph.v").read_text(encoding="utf-8")
        assert "module scnir_subgraph (" in hierarchy_module
        assert "output wire signed [15:0] weight_0" in hierarchy_module
        assert "output wire signed [15:0] weight_1" in hierarchy_module
        assert "assign weight_0 = 16'sh0080;" in hierarchy_module
        assert "assign weight_1 = 16'shffc0;" in hierarchy_module
        assert "// stream_id: conn.subgraph__a_to_lif_a.weight" in hierarchy_module
        assert "// stream_id: conn.subgraph__b_to_lif_b.weight" in hierarchy_module

        report = json.loads((out_dir / "scnir_handoff_audit.json").read_text(encoding="utf-8"))
        assert report["status"] == "valid"
        assert report["hierarchy_instance_count"] == 1
        assert report["hierarchy_port_count"] == 2
        assert report["hierarchy_instances"]["subgraph"]["ports"] == expected_ports
        assert "scnir_subgraph.v" in report["artefacts"]
        assert report["external_input_count"] == 2
        assert report["external_inputs"] == [
            {"source": "subgraph__a", "offset": 0, "width": 1},
            {"source": "subgraph__b", "offset": 1, "width": 1},
        ]


# ---------------------------------------------------------------------------
# Self-hosted hub command
# ---------------------------------------------------------------------------


class TestHubInitCommand:
    """Tests for `sc-neurocore hub-init ...`."""

    def test_hub_init_writes_bundle(self, tmp_path, capsys):
        from sc_neurocore.cli import _cmd_hub_init

        out = tmp_path / "hub"
        rc = _cmd_hub_init(
            str(out),
            port=8111,
            bind_host="10.0.0.5",
            image="sc-neurocore:test",
            offline=False,
        )

        assert rc == 0
        assert (out / "docker-compose.yml").exists()
        manifest = json.loads((out / "hub_manifest.json").read_text(encoding="utf-8"))
        assert manifest["services"]["studio"]["url"] == "http://10.0.0.5:8111"
        assert manifest["network_policy"]["ingress_scope"] == "private_network"
        assert manifest["network_policy"]["offline_environment"]["SC_NEUROCORE_HUB_OFFLINE"] == "0"
        assert "image: sc-neurocore:test" in (out / "docker-compose.yml").read_text(
            encoding="utf-8"
        )
        assert "hub bundle generated" in capsys.readouterr().out

    def test_hub_init_rejects_invalid_port(self, tmp_path, capsys):
        from sc_neurocore.cli import _cmd_hub_init

        rc = _cmd_hub_init(str(tmp_path / "hub"), port=0)

        assert rc == 1
        assert "studio_port must be in the range" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Serve command
# ---------------------------------------------------------------------------


class TestServeCommand:
    """Tests for `sc-neurocore serve ...` and the underlying _cmd_serve."""

    def test_serve_without_model_arg_returns_1(self, capsys):
        """`sc-neurocore serve` with no model argument prints usage and exits 1."""
        rc = _run_main("serve")
        assert rc == 1
        out = capsys.readouterr().out
        assert "serve requires a model file" in out

    def test_serve_rejects_non_nir_extension(self, capsys):
        """`.pt` and other extensions are not yet supported by serve; exit 1."""
        from sc_neurocore.cli import _cmd_serve

        rc = _cmd_serve("model.pt", port=8001, dt=1.0)
        assert rc == 1
        assert "supports .nir files only" in capsys.readouterr().out

    def test_serve_loads_nir_and_blocks_in_server(self, tmp_path, capsys):
        """Successful path: read NIR, build Network, start blocking SpikeServer."""
        from sc_neurocore.cli import _cmd_serve

        nir_path = tmp_path / "model.nir"
        nir_path.write_bytes(b"")  # contents are mocked away

        # Fake graph and Network — we only need topo_order length
        fake_graph = mock.MagicMock()
        fake_network = mock.MagicMock()
        fake_network.topo_order = ["a", "b", "c"]

        # Build a fake SpikeServer that records start() and returns immediately.
        fake_server_instance = mock.MagicMock()
        fake_server_cls = mock.MagicMock(return_value=fake_server_instance)

        # Patch the lazy imports inside _cmd_serve via sys.modules
        fake_nir = _fake_module("nir", read=mock.MagicMock(return_value=fake_graph))
        fake_bridge = _fake_module(
            "sc_neurocore.nir_bridge",
            from_nir=mock.MagicMock(return_value=fake_network),
        )
        fake_serve_mod = _fake_module(
            "sc_neurocore.serve",
            SpikeServer=fake_server_cls,
        )

        with mock.patch.dict(
            "sys.modules",
            {
                "nir": fake_nir,
                "sc_neurocore.nir_bridge": fake_bridge,
                "sc_neurocore.serve": fake_serve_mod,
            },
        ):
            rc = _cmd_serve(str(nir_path), port=8123, dt=1.0)

        assert rc == 0
        # SpikeServer was constructed with the fake network and the given port
        fake_server_cls.assert_called_once_with(fake_network, port=8123)
        # And started in blocking mode
        fake_server_instance.start.assert_called_once_with(blocking=True)
        # Confirmation print mentions the node count
        assert "Loaded NIR graph with 3 nodes" in capsys.readouterr().out

    def test_serve_via_main_dispatcher_routes_to_cmd_serve(self, tmp_path):
        """`sc-neurocore serve model.nir --port N` reaches _cmd_serve with the right args."""
        nir_path = tmp_path / "x.nir"
        nir_path.write_bytes(b"")
        with mock.patch("sc_neurocore.cli._cmd_serve", return_value=0) as m_serve:
            rc = _run_main("serve", str(nir_path), "--port", "9000", "--dt", "1.0")
        assert rc == 0
        m_serve.assert_called_once_with(str(nir_path), 9000, 1.0)
