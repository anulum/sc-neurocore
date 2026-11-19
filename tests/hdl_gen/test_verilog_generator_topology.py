# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verilog generator topology and dense-core elaboration tests

"""Module, layer, wire, bus-width, Verilator lint, and dense-core simulation contracts."""

from __future__ import annotations

from pathlib import Path
import json
import re
import shutil
import subprocess

from sc_neurocore.hdl_gen.verilog_generator import VerilogGenerator


_REPO_ROOT = Path(__file__).resolve().parents[2]
_HDL_DIR = _REPO_ROOT / "hdl"
_DENSE_STACK = (
    _HDL_DIR / "sc_dense_layer_core.v",
    _HDL_DIR / "sc_bitstream_encoder.v",
    _HDL_DIR / "sc_bitstream_synapse.v",
    _HDL_DIR / "sc_dotproduct_to_current.v",
    _HDL_DIR / "sc_lif_neuron.v",
)


def live_dense_core_lint_stub() -> str:
    """Build a lint stub using the exact module header from hdl/sc_dense_layer_core.v."""
    raw = (_HDL_DIR / "sc_dense_layer_core.v").read_text(encoding="utf-8")
    raw_no_formal = re.sub(r"`ifdef\s+FORMAL.*?`endif\s*", "", raw, flags=re.DOTALL)
    match = re.search(r"(module\s+sc_dense_layer_core\b.*?\);)", raw_no_formal, flags=re.DOTALL)
    assert match is not None
    header = match.group(1)
    return (
        f"{header}\n"
        "    wire _unused_dense_in = &{1'b0, clk, rst_n, start_pulse, stream_len,\n"
        "        x_input_fp, weight_fp, y_min_fp, y_max_fp, cfg_leak, cfg_gain, 1'b0};\n"
        "    assign I_t = {DATA_WIDTH{1'b0}};\n"
        "    genvar g;\n"
        "    generate\n"
        "        for (g = 0; g < N_NEURONS; g = g + 1) begin : g_spk\n"
        "            assign spikes[g] = |x_input_fp[(g % N_INPUTS)*DATA_WIDTH +: DATA_WIDTH];\n"
        "        end\n"
        "    endgenerate\n"
        "    assign step_valid = 1'b1;\n"
        "    always @(posedge clk or negedge rst_n) begin\n"
        "        if (!rst_n) begin\n"
        "            run_done <= 1'b0;\n"
        "            running  <= 1'b0;\n"
        "        end else begin\n"
        "            run_done <= start_pulse;\n"
        "            running  <= 1'b1;\n"
        "        end\n"
        "    end\n"
        "endmodule\n"
    )


def test_verilog_generator_source_has_no_suppressions() -> None:
    """verilog_generator.py and aer_emitter.py must carry zero lint/type/coverage suppressions."""
    for rel in (
        "src/sc_neurocore/hdl_gen/verilog_generator.py",
        "src/sc_neurocore/hdl_gen/aer_emitter.py",
    ):
        text = (_REPO_ROOT / rel).read_text(encoding="utf-8")
        assert ("type:" + " ignore") not in text, rel
        assert ("no" + "qa") not in text, rel
        assert ("pragma:" + " no cover") not in text, rel


def test_verilog_generator_module_name() -> None:
    """Generated code contains the sanitised module name."""
    gen = VerilogGenerator(module_name="my_top")
    code = gen.generate()
    assert "module my_top" in code


def test_verilog_generator_add_layer() -> None:
    """add_layer registers layers in declaration order."""
    gen = VerilogGenerator()
    gen.add_layer("Dense", "layer0", {"n_neurons": 4})
    assert len(gen.layers) == 1
    assert gen.layers[0]["name"] == "layer0"


def test_verilog_generator_single_layer_wiring() -> None:
    """Single Dense layer binds sc_dense_layer_core parameters, x_input_fp, and spikes."""
    gen = VerilogGenerator(bus_width=8)
    gen.add_layer("Dense", "dense0", {"n_neurons": 4})
    code = gen.generate()
    assert ".N_INPUTS(8)" in code
    assert ".N_NEURONS(4)" in code
    assert ".DATA_WIDTH(16)" in code
    assert ".x_input_fp(dense0_x_input_fp)" in code
    assert ".spikes(output_bus)" in code
    assert "input_bus[sc_in_idx]" in code


def test_verilog_generator_two_layers_wires() -> None:
    """Two Dense layers connect via inter-layer spike wire layer_0_to_1."""
    gen = VerilogGenerator()
    gen.add_layer("Dense", "dense0", {"n_neurons": 4})
    gen.add_layer("Dense", "dense1", {"n_neurons": 4})
    code = gen.generate()
    assert "wire [3:0] layer_0_to_1;" in code
    assert ".spikes(layer_0_to_1)" in code
    assert "layer_0_to_1[sc_in_idx]" in code


def test_verilog_generator_derives_declared_bus_widths() -> None:
    """Sync generator honours custom top-level bus_width and layer output_width."""
    gen = VerilogGenerator(module_name="wide_top", bus_width=16)
    gen.add_layer("Dense", "dense0", {"n_neurons": 12, "output_width": 12})
    code = gen.generate()
    assert "input wire [15:0] input_bus" in code
    assert "output wire [11:0] output_bus" in code
    assert ".N_INPUTS(16)" in code
    assert ".N_NEURONS(12)" in code


def test_verilog_generator_dense_instances() -> None:
    """Dense layers instantiate sc_dense_layer_core with optional per-synapse weights."""
    gen = VerilogGenerator(bus_width=3)
    gen.add_layer(
        "Dense",
        "dense0",
        {
            "n_neurons": 3,
            "weights": [0x4000, 0x8000, 0xFFFF],
            "y_min_fp": -16,
            "y_max_fp": 512,
            "cfg_leak": 16,
            "cfg_gain": 256,
            "stream_len": 64,
        },
    )
    code = gen.generate()
    assert "sc_dense_layer_core" in code
    assert "{16'hFFFF, 16'h8000, 16'h4000}" in code
    assert ".y_min_fp(16'hFFF0)" in code


def test_verilog_generator_no_layers_still_valid_and_lints_clean(tmp_path: Path) -> None:
    """Empty generator emits a driven zero output_bus and passes Verilator -Wall."""
    assert shutil.which("verilator") is not None
    gen = VerilogGenerator(module_name="empty_top")
    code = gen.generate()
    assert "module empty_top" in code
    assert "assign output_bus = 8'b0;" in code
    rtl_path = tmp_path / "empty_top.v"
    rtl_path.write_text(code, encoding="utf-8")
    lint = subprocess.run(
        ["verilator", "--lint-only", "-Wall", "--top-module", "empty_top", str(rtl_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert lint.returncode == 0, lint.stderr


def test_verilog_generator_verilator_clean_and_simulates_with_real_dense_core(
    tmp_path: Path,
) -> None:
    """Verify multi-layer sync output under Verilator -Wall and Icarus with sc_dense_layer_core."""
    assert shutil.which("verilator") is not None
    assert shutil.which("iverilog") is not None
    assert shutil.which("vvp") is not None

    gen = VerilogGenerator(module_name="sync_two_layer", bus_width=4)
    gen.add_layer("Dense", "dense0", {"n_neurons": 3, "stream_len": 64})
    gen.add_layer("Dense", "dense1", {"n_neurons": 2, "stream_len": 64})
    top_rtl = gen.generate()

    stub_path = tmp_path / "sc_dense_layer_core.v"
    top_path = tmp_path / "sync_two_layer.v"
    stub_path.write_text(live_dense_core_lint_stub(), encoding="utf-8")
    top_path.write_text(top_rtl, encoding="utf-8")

    lint = subprocess.run(
        [
            "verilator",
            "--lint-only",
            "-Wall",
            "--top-module",
            "sync_two_layer",
            str(stub_path),
            str(top_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert lint.returncode == 0, lint.stderr

    tb_sv = """\
`timescale 1ns/1ps
module tb_sync_real;
    reg clk = 1'b0;
    always #2 clk = ~clk;
    reg rst_n = 1'b0;
    reg [3:0] input_bus = 4'b0000;
    wire [1:0] output_bus;

    sync_two_layer dut (
        .clk(clk),
        .rst_n(rst_n),
        .input_bus(input_bus),
        .output_bus(output_bus)
    );

    integer zero_spikes = 0;
    integer active_spikes = 0;

    initial begin
        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        repeat (20) begin
            @(posedge clk);
            if (|output_bus) zero_spikes = zero_spikes + 1;
        end
        if (zero_spikes != 0) $fatal(1, "unexpected spikes on zero input: %0d", zero_spikes);

        input_bus = 4'b1111;
        repeat (80) begin
            @(posedge clk);
            if (|output_bus) active_spikes = active_spikes + 1;
        end
        if (active_spikes == 0)
            $fatal(1, "expected multi-layer spikes from real sc_dense_layer_core");
        $display("SYNC_DENSE_CORE_PASS spikes=%0d", active_spikes);
        $finish;
    end
endmodule
"""
    tb_path = tmp_path / "tb_sync_real.sv"
    vvp_path = tmp_path / "sync_real.vvp"
    tb_path.write_text(tb_sv, encoding="utf-8")

    comp = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-o",
            str(vvp_path),
            *(str(p) for p in _DENSE_STACK),
            str(top_path),
            str(tb_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert comp.returncode == 0, comp.stderr

    sim = subprocess.run(
        ["vvp", "-n", str(vvp_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert sim.returncode == 0, f"{sim.stdout}\n{sim.stderr}"
    assert "SYNC_DENSE_CORE_PASS" in sim.stdout


def test_a_dense_layer_s_neurons_share_one_weight_vector_and_spike_together(
    tmp_path: Path,
) -> None:
    """With distinct per-input weights, the real core still drives all of a layer's bits alike."""
    assert shutil.which("iverilog") is not None
    assert shutil.which("vvp") is not None

    gen = VerilogGenerator(module_name="one_layer", bus_width=4)
    gen.add_layer(
        "Dense",
        "dense0",
        {"n_neurons": 4, "stream_len": 64, "weights": [0xFFFF, 0x8000, 0x2000, 0x0400]},
    )
    top_path = tmp_path / "one_layer.v"
    top_path.write_text(gen.generate(), encoding="utf-8")
    tb_sv = """\
`timescale 1ns/1ps
module tb_one_layer;
    reg clk = 1'b0;
    always #2 clk = ~clk;
    reg rst_n = 1'b0;
    reg [3:0] input_bus = 4'b0000;
    wire [3:0] output_bus;

    one_layer dut (.clk(clk), .rst_n(rst_n), .input_bus(input_bus), .output_bus(output_bus));

    integer pattern;
    integer spiking = 0;
    integer differing = 0;

    initial begin
        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        for (pattern = 0; pattern < 4; pattern = pattern + 1) begin
            input_bus = (pattern == 0) ? 4'b1111 : (pattern == 1) ? 4'b0001
                : (pattern == 2) ? 4'b1000 : 4'b0110;
            repeat (500) begin
                @(posedge clk);
                if (|output_bus) spiking = spiking + 1;
                if ((|output_bus) && !(&output_bus)) differing = differing + 1;
            end
        end
        if (spiking == 0) $fatal(1, "expected spikes from the real core");
        if (differing != 0) $fatal(1, "neurons differed on %0d cycles", differing);
        $display("SHARED_WEIGHTS_PASS spiking=%0d", spiking);
        $finish;
    end
endmodule
"""
    tb_path = tmp_path / "tb_one_layer.sv"
    vvp_path = tmp_path / "one_layer.vvp"
    tb_path.write_text(tb_sv, encoding="utf-8")
    comp = subprocess.run(
        ["iverilog", "-g2012", "-o", str(vvp_path), *(str(p) for p in _DENSE_STACK)]
        + [str(top_path), str(tb_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert comp.returncode == 0, comp.stderr
    sim = subprocess.run(
        ["vvp", "-n", str(vvp_path)], capture_output=True, text=True, check=False, timeout=60
    )
    assert sim.returncode == 0, f"{sim.stdout}\n{sim.stderr}"
    assert "SHARED_WEIGHTS_PASS" in sim.stdout


def test_verilog_generator_python_and_rust_mirror_emit_byte_identical_verilog(
    tmp_path: Path,
) -> None:
    """Python VerilogGenerator and Rust safety mirror emit byte-identical sync and AER RTL."""
    assert shutil.which("rustc") is not None
    rust_src = _REPO_ROOT / "src/sc_neurocore/accel/rust/safety/verilog_generator.rs"
    driver_rs = tmp_path / "parity_driver.rs"
    driver_bin = tmp_path / "parity_driver"
    sync_out = tmp_path / "rust_sync.v"
    aer_out = tmp_path / "rust_aer.v"
    driver_rs.write_text(
        f"""\
#[path = {json.dumps(str(rust_src))}]
mod verilog_generator;
use verilog_generator::{{DenseLayerParams, StochasticSourceKind, VerilogGenerator}};

fn main() {{
    let mut sync_gen = VerilogGenerator::try_new("parity_top", 6).unwrap();
    sync_gen.add_dense_layer("dense0", DenseLayerParams::new(4)).unwrap();
    sync_gen.add_dense_layer("dense1", DenseLayerParams::new(2)).unwrap();
    sync_gen.add_stochastic_source("rng_lfsr", StochasticSourceKind::Lfsr16, 0xBEEF).unwrap();
    sync_gen.add_stochastic_source("rng_sobol", StochasticSourceKind::Sobol16, 0x0042).unwrap();
    sync_gen.add_stochastic_source("rng_halton", StochasticSourceKind::Halton16, 0).unwrap();
    std::fs::write({json.dumps(str(sync_out))}, sync_gen.generate("sync").unwrap()).unwrap();

    let mut aer_gen = VerilogGenerator::try_new("parity_aer", 6).unwrap();
    aer_gen.add_dense_layer("dense0", DenseLayerParams::new(4)).unwrap();
    aer_gen.add_dense_layer("dense1", DenseLayerParams::new(2)).unwrap();
    std::fs::write({json.dumps(str(aer_out))}, aer_gen.generate("async_aer").unwrap()).unwrap();
}}
""",
        encoding="utf-8",
    )
    comp = subprocess.run(
        ["rustc", "--edition=2021", str(driver_rs), "-o", str(driver_bin)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert comp.returncode == 0, comp.stderr
    run = subprocess.run(
        [str(driver_bin)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert run.returncode == 0, run.stderr

    py_sync = VerilogGenerator(module_name="parity_top", bus_width=6)
    py_sync.add_layer("Dense", "dense0", {"n_neurons": 4})
    py_sync.add_layer("Dense", "dense1", {"n_neurons": 2})
    py_sync.add_layer("StochasticSource", "rng_lfsr", {"source_type": "LFSR", "seed": 0xBEEF})
    py_sync.add_layer("StochasticSource", "rng_sobol", {"source_type": "Sobol", "seed": 0x0042})
    py_sync.add_layer("StochasticSource", "rng_halton", {"source_type": "Halton"})
    assert sync_out.read_text(encoding="utf-8") == py_sync.generate("sync")

    py_aer = VerilogGenerator(module_name="parity_aer", bus_width=6)
    py_aer.add_layer("Dense", "dense0", {"n_neurons": 4})
    py_aer.add_layer("Dense", "dense1", {"n_neurons": 2})
    assert aer_out.read_text(encoding="utf-8") == py_aer.generate("async_aer")
