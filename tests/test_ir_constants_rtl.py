# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Typed constants and encoding across the native boundary

"""Exercise typed constant words and probability encoding through actual PyO3 RTL."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import ScGraphBuilder
from sc_neurocore.hdl.resources import baseline_primitive_path
from sc_neurocore.neurons.fixed_point_lif import FixedPointBitstreamEncoder


@pytest.mark.parametrize("width,fraction", [(8, 0), (16, 8), (24, 16), (32, 24), (64, 40)])
def test_public_constant_vector_uses_declared_width_and_fraction(
    tmp_path: Path, width: int, fraction: int
) -> None:
    """Read the emitted packed words after quantizing through the installed builder."""
    values = [-0.375, 0.0, 0.625]
    builder = ScGraphBuilder("typed_constants")
    value = builder.constant_f64_vec(values, f"fixed<{width},{fraction}>")
    builder.output("words", value)
    graph = builder.build()
    assert graph.verify() is None
    sv = tmp_path / "constant.sv"
    sv.write_text(graph.emit_sv(), encoding="utf-8")
    tb = tmp_path / "tb.sv"
    displays = "\n".join(f'$display("WORD %h", words[{i * width} +: {width}]);' for i in range(3))
    tb.write_text(
        f"module tb; wire [{width * 3 - 1}:0] words; "
        "typed_constants dut(.clk(1'b0), .rst_n(1'b1), .words(words)); "
        f"initial begin #1; {displays} $finish; end endmodule\n",
        encoding="utf-8",
    )
    binary = tmp_path / "const.vvp"
    subprocess.run(
        ["iverilog", "-g2012", "-s", "tb", "-o", str(binary), str(sv), str(tb)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    run = subprocess.run(
        ["vvp", str(binary)], check=True, capture_output=True, text=True, timeout=30
    )
    actual = [
        int(line.removeprefix("WORD "), 16)
        for line in run.stdout.splitlines()
        if line.startswith("WORD ")
    ]
    assert actual == [int(value * (1 << fraction)) & ((1 << width) - 1) for value in values]


@pytest.mark.parametrize("width,fraction", [(1, 0), (8, 4), (16, 8), (24, 16), (32, 24), (64, 63)])
def test_encode_scales_dynamic_probability_to_fixed_lfsr_threshold(
    tmp_path: Path, width: int, fraction: int
) -> None:
    """Compare every emitted bit to the bit-true Python encoder after Q conversion."""
    builder = ScGraphBuilder("typed_encode")
    value = builder.input("probability", f"fixed<{width},{fraction}>")
    encoded = builder.encode(value, seed=0x4321)
    builder.output("bit_out", encoded)
    graph = builder.build()
    assert graph.verify() is None
    sv = tmp_path / "encode.sv"
    sv.write_text(graph.emit_sv(), encoding="utf-8")
    reference = FixedPointBitstreamEncoder(seed_init=0x4321)
    one = 1 << fraction
    maximum = (1 << (width - 1)) - 1
    values = [-one // 2, 0, one // 8, one // 2, min(one, maximum), maximum]
    expected: list[int] = []
    drive: list[str] = []
    for cycle in range(256):
        raw = values[(cycle // 5) % len(values)]
        threshold = max(0, min(65535, raw * 65535 // one))
        expected.append(reference.step(threshold))
        drive.append(
            f"probability={width}'h{raw & ((1 << width) - 1):x}; #2; clk=1; #2; "
            '$display("BIT %0d", bit_out); clk=0; #1;'
        )
    tb = tmp_path / "tb.sv"
    tb.write_text(
        f"module tb; reg clk=0, rst_n=0; reg [{width - 1}:0] probability=0; wire bit_out; "
        "typed_encode dut(.clk(clk),.rst_n(rst_n),.probability(probability),.bit_out(bit_out)); "
        "initial begin #1; clk=1; #1; clk=0; rst_n=1;\n"
        + "\n".join(drive)
        + "\n$finish; end endmodule\n",
        encoding="utf-8",
    )
    binary = tmp_path / "encode.vvp"
    subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-Wall",
            "-s",
            "tb",
            "-o",
            str(binary),
            str(sv),
            str(tb),
            str(baseline_primitive_path("sc_bitstream_encoder.v")),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    run = subprocess.run(
        ["vvp", str(binary)], check=True, capture_output=True, text=True, timeout=30
    )
    actual = [
        int(line.removeprefix("BIT "))
        for line in run.stdout.splitlines()
        if line.startswith("BIT ")
    ]
    assert actual == expected


@pytest.mark.parametrize(
    "ty,value", [("i8", 128), ("u8", -1), ("fixed<8,8>", 1), ("fixed<0,0>", 1), ("vec<rate,2>", 1)]
)
def test_public_integer_constant_rejects_invalid_word_or_shape(ty: str, value: int) -> None:
    """Report invalid declarations at both graph verification and HDL emission."""
    builder = ScGraphBuilder("invalid_word")
    result = builder.constant_i64(value, ty)
    builder.output("value", result)
    graph = builder.build()
    assert graph.verify()
    with pytest.raises(ValueError):
        graph.emit_sv()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 128.0, -129.0])
def test_public_float_constant_rejects_nonfinite_or_unrepresentable_values(value: float) -> None:
    """Reject float saturation instead of silently changing the emitted constant."""
    builder = ScGraphBuilder("invalid_float")
    result = builder.constant_f64(value)
    builder.output("value", result)
    graph = builder.build()
    assert graph.verify()
    with pytest.raises(ValueError):
        graph.emit_sv()
