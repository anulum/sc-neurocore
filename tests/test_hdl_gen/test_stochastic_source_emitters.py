# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for standalone LFSR-16 and Sobol-16 HDL emitters

from __future__ import annotations

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


def test_lfsr16_emitter_generates_software_parity_module():
    verilog = Lfsr16Emitter(module_name="lfsr16_source", seed=0xBEEF).generate()

    assert "module lfsr16_source" in verilog
    assert "assign bit_out = (state < threshold);" in verilog
    assert "state[0] ^ state[2] ^ state[3] ^ state[5]" in verilog
    assert "localparam [15:0] FIRST_SAMPLE" in verilog
    assert "state <= {feedback, state[15:1]};" in verilog
    assert "16'hBEEF" in verilog


def test_lfsr16_reference_formula_matches_python_encoder():
    lfsr = Lfsr16(seed=0xACE1)
    state = 0xACE1

    for _ in range(128):
        state = _lfsr16_step(state)
        assert lfsr.step() == state


def test_sobol16_emitter_generates_direction_table_and_software_parity_module():
    verilog = Sobol16Emitter(module_name="sobol16_source", seed=0x1234).generate()

    assert "module sobol16_source" in verilog
    assert "assign bit_out = (value < threshold);" in verilog
    assert "16'h8000" in verilog
    assert "16'h0001" in verilog
    assert "value <= value ^ direction;" in verilog
    assert "index <= 16'd1;" in verilog
    assert "16'h1234" in verilog


def test_sobol16_reference_formula_matches_python_generator():
    sobol = SobolGenerator(seed=0x0000)
    value = 0x0000
    index = 0

    for _ in range(128):
        value, index = _sobol16_step(value, index)
        assert sobol.step() == value


def test_lfsr16_emitted_rtl_matches_reference_sequence(tmp_path):
    seed = 0xBEEF
    threshold = 0x8000
    verilog = Lfsr16Emitter(module_name="lfsr16_parity", seed=seed).generate()
    samples = _simulate_source(verilog, _lfsr_testbench("lfsr16_parity", threshold), tmp_path)

    state = _lfsr16_step(seed)
    expected = []
    for idx in range(_RTL_SAMPLE_COUNT):
        expected.append((idx, state, int(state < threshold)))
        state = _lfsr16_step(state)

    assert samples == expected
    assert _pack_sample_bits(samples, 32) == Lfsr16(seed=seed).encode(threshold, _RTL_SAMPLE_COUNT)


def test_sobol16_emitted_rtl_matches_reference_sequence(tmp_path):
    seed = 0x0042
    threshold = 0x4000
    verilog = Sobol16Emitter(module_name="sobol16_parity", seed=seed).generate()
    samples = _simulate_source(verilog, _sobol_testbench("sobol16_parity", threshold), tmp_path)

    value, index = _sobol16_step(seed, 0)
    expected = []
    for idx in range(_RTL_SAMPLE_COUNT):
        expected.append((idx, value, int(value < threshold)))
        value, index = _sobol16_step(value, index)

    assert samples == expected
    assert _pack_sample_bits(samples, 64) == [
        int(word) for word in SobolGenerator(seed=seed).encode(threshold, _RTL_SAMPLE_COUNT)
    ]


def test_verilog_generator_exposes_stochastic_source_helpers():
    generator = VerilogGenerator()

    lfsr_verilog = generator.emit_lfsr16_source(seed=0xACE1)
    sobol_verilog = generator.emit_sobol16_source(seed=0x0042)

    assert "module sc_lfsr16_source" in lfsr_verilog
    assert "module sc_sobol16_source" in sobol_verilog
    assert "16'h0042" in sobol_verilog


def test_emit_sources_from_ir_accepts_mapping_nodes():
    verilog = emit_sources_from_ir(
        {
            "nodes": [
                {
                    "name": "rng_lfsr",
                    "type": "StochasticSource",
                    "params": {"source_type": "LFSR", "seed": 0xBEEF},
                },
                {
                    "id": "rng_sobol",
                    "node_type": "StochasticSource",
                    "decorrelator": "Sobol",
                    "seed": 0x0042,
                },
                {"name": "dense0", "type": "Dense"},
            ]
        }
    )

    assert "module rng_lfsr" in verilog
    assert "16'hBEEF" in verilog
    assert "module rng_sobol" in verilog
    assert "16'h0042" in verilog
    assert "dense0" not in verilog


def test_verilog_generator_routes_stochastic_sources_from_ir():
    generator = VerilogGenerator()

    verilog = generator.emit_sources_from_ir(
        {
            "nodes": {
                "source-a": {
                    "type": "lfsr16",
                    "module_name": "source_a",
                    "seed": 0,
                }
            }
        }
    )

    assert "module source_a" in verilog
    assert "16'hACE1" in verilog


def test_emit_sources_from_ir_accepts_object_nodes():
    node = SimpleNamespace(
        module_name="object_sobol",
        node_type="StochasticSource",
        params={"decorrelator": "sobol16", "seed": 0x0017},
    )

    verilog = emit_sources_from_ir(SimpleNamespace(nodes=[node]))

    assert "module object_sobol" in verilog
    assert "16'h0017" in verilog


def test_emit_sources_from_ir_rejects_unknown_explicit_source_kind():
    with pytest.raises(ValueError, match="unsupported stochastic source type"):
        emit_sources_from_ir(
            {
                "nodes": [
                    {
                        "type": "StochasticSource",
                        "params": {"source_type": "NonexistentSource"},
                    }
                ]
            }
        )


def test_emit_sources_from_ir_rejects_duplicate_module_names():
    with pytest.raises(ValueError, match="duplicate stochastic source module name"):
        emit_sources_from_ir(
            {
                "nodes": [
                    {"type": "lfsr16", "module_name": "shared_source"},
                    {"type": "sobol16", "module_name": "shared_source"},
                ]
            }
        )


def test_stochastic_source_emitters_reject_invalid_module_names():
    with pytest.raises(ValueError, match="Invalid module name"):
        Lfsr16Emitter(module_name="lfsr-16 source")

    with pytest.raises(ValueError, match="Invalid module name"):
        Sobol16Emitter(module_name="sobol-16 source")


def test_lfsr16_emitter_zero_seed_falls_back_to_default():
    """Seed 0 is an absorbing state for the LFSR; emitter must reject it."""
    emitter = Lfsr16Emitter(module_name="lfsr16_zero", seed=0x0000)
    assert emitter.seed == 0xACE1
    verilog = emitter.generate()
    assert "16'hACE1" in verilog


def test_lfsr16_emitter_masks_seed_to_16_bits():
    """Seeds wider than 16 bits are silently masked to preserve module contract."""
    emitter = Lfsr16Emitter(module_name="lfsr16_mask", seed=0x1234_BEEF)
    assert emitter.seed == 0xBEEF


def test_sobol16_emitter_masks_seed_to_16_bits():
    """Same 16-bit mask guarantee for the Sobol emitter."""
    emitter = Sobol16Emitter(module_name="sobol16_mask", seed=0x00FF_0042)
    assert emitter.seed == 0x0042
    verilog = emitter.generate()
    assert "16'h0042" in verilog


def test_require_positive_int_rejects_non_positive_value():
    with pytest.raises(ValueError, match="must be a positive integer"):
        VerilogGenerator._require_positive_int(0, "width")


def test_emit_async_aer_wraps_declared_dense_layers():
    generator = VerilogGenerator(module_name="async_route")
    generator.add_layer("Dense", "dense0", {"n_neurons": 4})
    verilog = generator.emit_async_aer()
    assert "module async_route" in verilog


def test_emit_quasirandom_source_rejects_unknown_method():
    with pytest.raises(ValueError, match="method must be 'sobol' or 'halton'"):
        VerilogGenerator().emit_quasirandom_source(method="mt19937")


def test_emit_sources_from_ir_rejects_non_collection_payload():
    with pytest.raises(TypeError, match="mapping or sequence of nodes"):
        emit_sources_from_ir(42)


def test_emit_sources_from_ir_rejects_source_without_generator():
    with pytest.raises(ValueError, match="missing source_type/decorrelator"):
        emit_sources_from_ir([{"type": "stochastic_source"}])


def test_emit_sources_from_ir_defaults_unnamed_source_module() -> None:
    verilog = emit_sources_from_ir([{"type": "stochastic_source", "source_type": "sobol"}])
    assert "sc_stochastic_source_0" in verilog
