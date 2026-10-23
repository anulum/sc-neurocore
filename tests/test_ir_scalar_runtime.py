# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed scalar word graph circuits

"""Exercise the public Python facade, PyO3 methods, persisted IR and real RTL."""

from __future__ import annotations

import math
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import ScGraphBuilder, parse_ir


def _rtl_observations(tmp_path: Path, source: str, bench: str) -> list[str]:
    """Compile real emitted RTL and return its ordered public output observations."""
    top = tmp_path / "top.sv"
    tb = tmp_path / "tb.sv"
    binary = tmp_path / "scalar.vvp"
    top.write_text(source, encoding="utf-8")
    tb.write_text(bench, encoding="utf-8")
    subprocess.run(
        ["iverilog", "-g2012", "-s", "tb", "-o", str(binary), str(top), str(tb)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    trace = subprocess.run(
        ["vvp", str(binary)], capture_output=True, text=True, check=True, timeout=30
    )
    return [line for line in trace.stdout.splitlines() if line.startswith("OBS ")]


def test_public_scalar_pipeline_matches_encoded_reference(tmp_path: Path) -> None:
    """Multiply with explicit width, rescale, saturate and select through every builder family."""
    b = ScGraphBuilder("python_scalar")
    x = b.input("x", "i16")
    fixed = b.word_bitcast(x, "fixed<16,8>")
    raw = b.word_bitcast(fixed, "i16")
    wide = b.word_resize(raw, "i32")
    squared = b.word_binary("mul", wide, wide, "i32")
    rescaled = b.word_round(squared, 8, "half_even", "i32")
    bounded = b.word_clamp(rescaled, -32768, 32767, "i32")
    result = b.word_resize(bounded, "i16")
    negative = b.word_unary("neg", raw, "i16")
    zero = b.constant_i64(0, "i16")
    condition = b.word_compare("lt", raw, zero)
    selected = b.word_select(condition, negative, result, "i16")
    b.output("result", selected)
    graph = b.build()
    assert graph.verify() is None
    assert ": fixed<16,8>" in graph.to_text()
    restored = parse_ir(graph.to_text())
    assert restored.to_text() == graph.to_text()
    assert restored.emit_sv() == graph.emit_sv()
    assert restored.emit_mlir() == graph.emit_mlir()
    words = [-32768, -257, -1, 0, 1, 127, 128, 129, 257, 32767]
    expected = []
    drives = []
    for word in words:
        quotient, residue = divmod(word * word, 256)
        quotient += int(residue > 128 or (residue == 128 and quotient % 2 == 1))
        output = -word if word < 0 else min(quotient, 32767)
        expected.append(f"OBS {output & 65535:04x}")
        drives.append(f'x=16\'h{word & 65535:x}; #1; $display("OBS %h",result);')
    top = tmp_path / "top.sv"
    top.write_text(graph.emit_sv(), encoding="utf-8")
    bench = tmp_path / "tb.sv"
    bench.write_text(
        "module tb; reg clk=0,rst_n=1; reg [15:0] x=0; wire [15:0] result; "
        "python_scalar dut(.*); initial begin " + "\n".join(drives) + "$finish; end endmodule",
        encoding="utf-8",
    )
    binary = tmp_path / "scalar.vvp"
    subprocess.run(
        ["iverilog", "-g2012", "-s", "tb", "-o", str(binary), str(top), str(bench)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    trace = subprocess.run(
        ["vvp", str(binary)], capture_output=True, text=True, check=True, timeout=30
    )
    assert [line for line in trace.stdout.splitlines() if line.startswith("OBS ")] == expected


@pytest.mark.parametrize("kind", ["plus", "", "signed_divide"])
def test_public_scalar_builder_rejects_unknown_operation(kind: str) -> None:
    """Unknown spellings fail at the native boundary without consuming the builder."""
    b = ScGraphBuilder("invalid_operation")
    x = b.input("x", "i16")
    with pytest.raises(ValueError, match="unknown WordBinaryOp"):
        b.word_binary(kind, x, x, "i16")
    b.output("value", b.word_binary("add", x, x, "i16"))
    assert b.build().verify() is None


def test_public_scalar_emission_refuses_implicit_type_coercion() -> None:
    """Both emitters retain the verifier's mismatch instead of silently resizing a word."""
    b = ScGraphBuilder("invalid_word_type")
    x = b.input("x", "i16")
    b.output("value", b.word_binary("add", x, x, "i32"))
    graph = b.build()
    assert graph.verify()
    with pytest.raises(ValueError, match="requires i32, got i16"):
        graph.emit_sv()
    with pytest.raises(ValueError, match="requires i32, got i16"):
        graph.emit_mlir()


@pytest.mark.parametrize("bound", [-(1 << 63) - 1, 1 << 63, 1 << 128])
def test_public_scalar_clamp_refuses_unrepresentable_literal_bounds(bound: int) -> None:
    """A wide data type cannot silently narrow a bound through the native i64 interface."""
    b = ScGraphBuilder("clamp_literal_range")
    x = b.input("x", "u128")
    with pytest.raises(OverflowError):
        b.word_clamp(x, 0, bound, "u128")
    b.output("value", b.word_clamp(x, 0, 32767, "u128"))
    assert b.build().verify() is None


def test_public_dense_namespace_collision_is_an_authored_refusal() -> None:
    """Normal installed native calls reject the reproduced Dense conversion-bus collision."""
    b = ScGraphBuilder("dense_collision")
    inputs = b.input("dense_0_inputs", "vec<rate,1>")
    weights = b.constant_f64_vec([0.5], "rate")
    leak = b.constant_i64(16, "i16")
    gain = b.constant_i64(256, "i16")
    b.output("spikes", b.dense_forward(inputs, weights, leak, gain, n_inputs=1, n_neurons=1))
    graph = b.build()
    assert graph.verify()
    with pytest.raises(ValueError, match="reserved generated namespace"):
        graph.emit_sv()
    with pytest.raises(ValueError, match="reserved generated namespace"):
        graph.emit_mlir()


@pytest.mark.parametrize(
    ("kind", "left", "right", "expected"),
    [
        ("add", 120, 120, 240),
        ("sub", 3, 7, 252),
        ("mul", -17, 9, 103),
        ("div", -7, 3, 254),
        ("rem", -7, 3, 255),
        ("div", 7, 0, 0),
        ("rem", -7, 0, 0),
        ("and", 0xA5, 0x3C, 0x24),
        ("or", 0xA5, 0x3C, 0xBD),
        ("xor", 0xA5, 0x3C, 0x99),
        ("shl", 0x81, 1, 2),
        ("lshr", 0x81, 1, 0x40),
        ("ashr", 0x81, 1, 0xC0),
    ],
)
def test_public_binary_kind_reaches_its_integer_circuit(
    tmp_path: Path, kind: str, left: int, right: int, expected: int
) -> None:
    """The Python operation name chooses the corresponding native arithmetic, not just a valid type."""
    b = ScGraphBuilder("python_binary")
    x = b.input("x", "i8")
    y = b.input("y", "i8")
    b.output("value", b.word_binary(kind, x, y, "i8"))
    graph = parse_ir(b.build().to_text())
    bench = (
        f"module tb; reg clk=0,rst_n=1; reg [7:0] x=8'h{left & 255:x},y=8'h{right & 255:x}; "
        "wire [7:0] value; python_binary dut(.*); initial begin #1; "
        '$display("OBS %h",value); $finish; end endmodule'
    )
    assert _rtl_observations(tmp_path, graph.emit_sv(), bench) == [f"OBS {expected:02x}"]


@pytest.mark.parametrize(("kind", "word", "expected"), [("neg", 7, 249), ("not", 0x55, 0xAA)])
def test_public_unary_kind_preserves_encoded_bits(
    tmp_path: Path, kind: str, word: int, expected: int
) -> None:
    """Exercise both native unary dispatches through installed Python and actual output ports."""
    b = ScGraphBuilder("python_unary")
    x = b.input("x", "i8")
    b.output("value", b.word_unary(kind, x, "i8"))
    bench = (
        f"module tb; reg clk=0,rst_n=1; reg [7:0] x=8'h{word:x}; wire [7:0] value; "
        'python_unary dut(.*); initial begin #1; $display("OBS %h",value); $finish; end endmodule'
    )
    assert _rtl_observations(tmp_path, b.build().emit_sv(), bench) == [f"OBS {expected:02x}"]


def test_public_comparison_predicates_preserve_signed_ordering(tmp_path: Path) -> None:
    """Each predicate crosses the Python/native boundary and produces its own Boolean result."""
    b = ScGraphBuilder("python_predicates")
    x = b.input("x", "i8")
    y = b.input("y", "i8")
    for i, predicate in enumerate(["eq", "ne", "lt", "le", "gt", "ge"]):
        b.output(f"o{i}", b.word_compare(predicate, x, y))
    bench = (
        "module tb; reg clk=0,rst_n=1; reg [7:0] x=1,y=2; wire o0,o1,o2,o3,o4,o5; "
        'python_predicates dut(.*); initial begin #1; $display("OBS %b%b%b%b%b%b",o0,o1,o2,o3,o4,o5); '
        'x=255; y=255; #1; $display("OBS %b%b%b%b%b%b",o0,o1,o2,o3,o4,o5); '
        'x=127; y=128; #1; $display("OBS %b%b%b%b%b%b",o0,o1,o2,o3,o4,o5); $finish; end endmodule'
    )
    assert _rtl_observations(tmp_path, b.build().emit_sv(), bench) == [
        "OBS 011100",
        "OBS 100101",
        "OBS 010011",
    ]


def test_public_round_modes_preserve_signed_ties(tmp_path: Path) -> None:
    """Check every rounding-mode argument against exact rational arithmetic, including negative ties."""
    b = ScGraphBuilder("python_round_modes")
    x = b.input("x", "i16")
    modes = ["floor", "toward_zero", "half_up", "half_away", "half_even"]
    for i, mode in enumerate(modes):
        b.output(f"o{i}", b.word_round(x, 1, mode, "i16"))
    words = [-32768, -32767, -7, -6, -5, -3, -2, -1, 0, 1, 2, 3, 5, 6, 7, 32767]
    expected = []
    drive = []
    for word in words:
        value = Fraction(word, 2)
        answers = [
            math.floor(value),
            math.trunc(value),
            math.floor(value + Fraction(1, 2)),
            (-1 if word < 0 else 1) * math.floor(abs(value) + Fraction(1, 2)),
            round(value),
        ]
        expected.append("OBS " + " ".join(map(str, answers)))
        drive.append(
            f'x=16\'h{word & 65535:x}; #1; $display("OBS %0d %0d %0d %0d %0d",'
            "$signed(o0),$signed(o1),$signed(o2),$signed(o3),$signed(o4));"
        )
    bench = (
        "module tb; reg clk=0,rst_n=1; reg [15:0] x=0; wire [15:0] o0,o1,o2,o3,o4; "
        "python_round_modes dut(.*); initial begin " + "\n".join(drive) + "$finish; end endmodule"
    )
    assert _rtl_observations(tmp_path, b.build().emit_sv(), bench) == expected


def test_public_clamp_preserves_both_signed_literal_bounds(tmp_path: Path) -> None:
    """A caller's negative lower bound and positive upper bound both reach the saturation circuit."""
    b = ScGraphBuilder("python_bounds")
    x = b.input("x", "i16")
    b.output("value", b.word_clamp(x, -3, 5, "i16"))
    words = [-32768, -4, -3, -1, 0, 5, 6, 32767]
    drive = "\n".join(
        f'x=16\'h{word & 65535:x}; #1; $display("OBS %0d",$signed(value));' for word in words
    )
    bench = (
        "module tb; reg clk=0,rst_n=1; reg [15:0] x=0; wire [15:0] value; "
        "python_bounds dut(.*); initial begin " + drive + "$finish; end endmodule"
    )
    assert _rtl_observations(tmp_path, b.build().emit_sv(), bench) == [
        "OBS -3",
        "OBS -3",
        "OBS -3",
        "OBS -1",
        "OBS 0",
        "OBS 5",
        "OBS 5",
        "OBS 5",
    ]


@pytest.mark.parametrize("type_name", ["u0", "i0", "fixed<16,16>", "bitstream<0>", "vec<bool,0>"])
def test_unused_invalid_declared_input_refuses_emission(type_name: str) -> None:
    """Unused invalid declarations cannot bypass normal installed graph verification."""
    b = ScGraphBuilder("unused_bad_type")
    b.input("unused", type_name)
    x = b.input("x", "i8")
    b.output("value", b.word_binary("add", x, x, "i8"))
    graph = b.build()
    assert graph.verify()
    for emit in [graph.emit_sv, graph.emit_mlir]:
        with pytest.raises(ValueError, match="invalid declared"):
            emit()


def test_public_type_constructor_refuses_excessive_nesting_before_consumption(
    tmp_path: Path,
) -> None:
    """The Python type parser rejects deep recursion while leaving the builder usable."""
    type_name = "vec<" * 1000 + "u8" + ",1>" * 1000
    b = ScGraphBuilder("type_depth")
    with pytest.raises(ValueError, match="nesting exceeds 64"):
        b.input("deep", type_name)
    x = b.input("x", "u8")
    b.output("value", x)
    assert b.build().verify() is None
    valid_type = "vec<" * 64 + "u8" + ",1>" * 64
    b = ScGraphBuilder("python_nested")
    bits = b.input("bits", valid_type)
    b.output("value", bits)
    graph = parse_ir(b.build().to_text())
    assert graph.verify() is None
    bench = (
        "module tb; reg clk=0,rst_n=1; reg [7:0] bits=8'd174; wire [7:0] value; "
        'python_nested dut(.*); initial begin #1; $display("OBS %h",value); $finish; end endmodule'
    )
    assert _rtl_observations(tmp_path, graph.emit_sv(), bench) == ["OBS ae"]
    with pytest.raises(ValueError, match="line 2.*nesting exceeds 64"):
        parse_ir(f'sc.graph @deep {{\n  %0 = sc.input "x" : {type_name}\n}}\n')


def test_public_text_parser_refuses_exhausted_value_allocator() -> None:
    """A maximum SSA identifier produces a located Python error instead of wrapping or aborting."""
    with pytest.raises(ValueError, match="line 2.*value ID overflow"):
        parse_ir('sc.graph @max_id {\n  %4294967295 = sc.input "x" : u8\n}\n')
