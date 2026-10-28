# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native function and RNG integer-program execution

"""Persist public integer programs through native graph builders and execute real RTL."""

from __future__ import annotations

import inspect
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from sc_neurocore.compiler import function_units as fu
from sc_neurocore.compiler import random_units as ru
from sc_neurocore.compiler.function_unit_ir import Program, Step
from sc_neurocore.compiler.function_unit_programs import program
from sc_neurocore.compiler.native_program import NativeWordBuilder, lower_program
from sc_neurocore.compiler.random_unit_programs import program as random_program
from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import ScGraph, ScGraphBuilder, parse_ir


def _observe(
    tmp_path: Path,
    label: str,
    graph: ScGraph,
    names: list[str],
    cases: Sequence[tuple[int, ...]],
    expected: list[int],
) -> None:
    """Execute a persisted native circuit and retain its real CIRCT inputs on request."""
    assert graph.verify() is None
    restored = parse_ir(graph.to_text())
    assert restored.to_text() == graph.to_text()
    assert restored.emit_sv() == graph.emit_sv()
    assert restored.emit_mlir() == graph.emit_mlir()
    observations = [f"OBS {value & ((1 << 64) - 1):016x}" for value in expected]
    declarations = " ".join(f"reg [63:0] {name}=0;" for name in names)
    stimulus = []
    for values in cases:
        drive = " ".join(
            f"{name}=64'h{value & ((1 << 64) - 1):016x};"
            for name, value in zip(names, values, strict=True)
        )
        stimulus.append(drive + ' #1; $display("OBS %h",result);')
    bench = (
        f"module tb; reg clk=0,rst_n=1; {declarations} wire [63:0] result; "
        "native_program dut(.*); initial begin\n"
        + "\n".join(stimulus)
        + "\n$finish; end endmodule\n"
    )
    if root := os.environ.get("SCN_IR_CIRCT_RECEIPT_DIR"):
        case = Path(root) / label
        case.mkdir(parents=True, exist_ok=True)
        (case / "graph.scir").write_text(restored.to_text(), encoding="utf-8")
        (case / "graph.mlir").write_text(restored.emit_mlir(), encoding="utf-8")
        (case / "tb.sv").write_text(bench, encoding="utf-8")
        (case / "expected.txt").write_text("\n".join(observations) + "\n", encoding="utf-8")
    top, tb, binary = tmp_path / "top.sv", tmp_path / "tb.sv", tmp_path / "unit.vvp"
    top.write_text(restored.emit_sv(), encoding="utf-8")
    tb.write_text(bench, encoding="utf-8")
    subprocess.run(
        ["iverilog", "-g2012", "-s", "tb", "-o", str(binary), str(top), str(tb)],
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    actual = subprocess.run(
        ["vvp", str(binary)], capture_output=True, text=True, check=True, timeout=90
    ).stdout
    assert [line for line in actual.splitlines() if line.startswith("OBS ")] == observations


@pytest.mark.parametrize("function", fu.FUNCTIONS)
@pytest.mark.parametrize(("width", "fraction"), [(8, 4), (16, 8), (32, 16)])
def test_native_function_program_matches_independent_reference(
    function: str, width: int, fraction: int, tmp_path: Path
) -> None:
    """Every existing unary/binary function executes from a genuine persisted native graph."""
    unit = program(function, width, fraction)
    low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
    codes = [low, -1, 0, 1, high]
    count = fu.FUNCTION_ARITY[function]
    cases = [
        (code,) if count == 1 else (code, codes[-index - 1]) for index, code in enumerate(codes)
    ]
    expected = [
        fu.evaluate(function, values[0], width, fraction, additional_inputs=values[1:])
        for values in cases
    ]
    assert [unit.evaluate(values[0], additional_inputs=values[1:]) for values in cases] == expected
    b = ScGraphBuilder("native_program")
    names = [f"arg{index}" for index in range(count)]
    inputs = [b.input(name, "i64") for name in names]
    b.output("result", lower_program(unit, b, inputs))
    _observe(tmp_path, f"function-{function}-{width}-{fraction}", b.build(), names, cases, expected)


@pytest.mark.parametrize("kind", ["rand", "randn", "poisson", "poisson_grid", "randn_exact"])
def test_native_rng_program_preserves_counter_seed_stream_and_site(
    kind: str, tmp_path: Path
) -> None:
    """Lower the existing RNG program including signed seed bits and 32-bit counter rollover."""
    site, width, fraction = 7, 16, 8
    unit = random_program(kind, site, width, fraction, 3.5 if kind == "poisson" else None)
    b = ScGraphBuilder("native_program")
    names = ["arg", "step", "seed", "stream"]
    ids = [b.input(name, "i64") for name in names]
    b.output("result", lower_program(unit, b, ids[:1], dict(zip(names[1:], ids[1:], strict=True))))
    cases = [(0, 0, 0, 0), (256, (1 << 32) - 1, -2, 3), (896, (1 << 32) + 5, -17, 19)]
    expected = []
    for code, step, seed, stream in cases:
        value = unit.evaluate(code, {"step": step, "seed": seed, "stream": stream})
        word0, word1, _, _ = ru.draw(site, step, stream, seed & ((1 << 64) - 1))
        if kind == "rand":
            assert value == ru.uniform_code(word0, fraction)
        elif kind == "randn":
            assert value == ru.normal_code(word0, word1, width, fraction)
        elif kind == "poisson":
            assert value == ru.poisson_code(word0, 3.5, width, fraction)
        expected.append(value)
    _observe(tmp_path, f"rng-{kind}", b.build(), names, cases, expected)


@pytest.mark.parametrize("coefficient", [0, 1, 2])
def test_native_program_table_endpoints_and_repeated_lookup(
    coefficient: int, tmp_path: Path
) -> None:
    """Use real log2 coefficients below, inside and above the table, including a repeated lookup."""
    rows = program("log", 16, 8).tables["log2"]
    p = Program(tables={"log2": rows})
    index = p.input()
    first = p.lookup("log2", index, coefficient)
    p.output = p.add(first, p.lookup("log2", index, coefficient))
    cases = [(value,) for value in [-9, -1, 0, 1, len(rows) - 1, len(rows), len(rows) + 9]]
    expected = [2 * rows[max(0, min(len(rows) - 1, value))][coefficient] for (value,) in cases]
    b = ScGraphBuilder("native_program")
    b.output("result", lower_program(p, b, [b.input("arg", "i64")]))
    _observe(tmp_path, f"table-{coefficient}", b.build(), ["arg"], cases, expected)


def test_native_program_keeps_zero_division_and_clamped_shifts(tmp_path: Path) -> None:
    """Real signed division and variable shift edges preserve the shared integer semantics."""
    p = Program()
    x, amount = p.input(), p.input(1)
    quotient = p.div(p.const(-7), x)
    shifted = p.shr_var(p.const(-19), amount)
    fixed = p.add(p.shl(p.const(0), 100), p.shr(p.const(-19), 100))
    p.output = p.add(p.add(quotient, shifted), fixed)
    cases = [(-3, -1), (-2, 0), (-1, 1), (0, 62), (1, 63), (2, 64), (3, 100)]
    expected = [p.evaluate(x, additional_inputs=(amount,)) for x, amount in cases]
    b = ScGraphBuilder("native_program")
    operands = [b.input("arg", "i64"), b.input("amount", "i64")]
    b.output("result", lower_program(p, b, operands))
    _observe(tmp_path, "division-shifts", b.build(), ["arg", "amount"], cases, expected)


@pytest.mark.parametrize(
    ("steps", "output", "message"),
    [
        ([Step("bad", "unknown")], "bad", "instruction"),
        ([Step("bad", "add", ("x",))], "bad", "instruction"),
        ([Step("bad", "neg", ("missing",))], "bad", "undefined"),
        ([Step("x", "const"), Step("x", "const")], "x", "unique"),
        ([Step("bad", "input", value=-1), Step("x", "input")], "x", "input index"),
        ([Step("x", "input"), Step("bad", "shl", ("x",), -1)], "bad", "non-negative"),
        ([Step("bad", "const", value=1 << 63)], "bad", "constant"),
        ([Step("bad", "extern", table="seed")], "bad", "external binding"),
        ([Step("x", "input"), Step("bad", "table", ("x",), table="empty")], "bad", "table"),
        ([Step("x", "input")], "missing", "output"),
    ],
)
def test_invalid_program_is_refused_before_native_graph_mutation(
    steps: list[Step], output: str, message: str
) -> None:
    """Public native lowering rejects malformed programs without appending partial instructions."""
    b = ScGraphBuilder("invalid_program")
    x = b.input("arg", "i64")
    with pytest.raises(ValueError, match=message):
        lower_program(Program(steps=steps, output=output), b, [x])
    b.output("result", x)
    assert len(b.build()) == 2


def test_engine_builder_matches_the_compiler_builder_protocol() -> None:
    """Every method the compiler types against exists on the engine builder with the same signature."""
    members = [
        name
        for name, value in vars(NativeWordBuilder).items()
        if not name.startswith("_") and inspect.isfunction(value)
    ]
    assert "word_select" in members and "word_delay" in members
    for name in members:
        expected = inspect.signature(getattr(NativeWordBuilder, name))
        actual = inspect.signature(getattr(ScGraphBuilder, name))
        assert list(actual.parameters) == list(expected.parameters), name
        for parameter in expected.parameters.values():
            assert actual.parameters[parameter.name].default == parameter.default, (name, parameter)


@pytest.mark.parametrize("row", [(0, 1 << 63, 0), (0, -(1 << 63), 0)])
def test_native_program_refuses_invalid_table_coefficients(row: tuple[int, int, int]) -> None:
    """Signed coefficient overflow refuses before native instructions append."""
    builder = ScGraphBuilder("invalid_table")
    source = builder.input("arg", "i64")
    unit = Program(tables={"coefficients": (row,)})
    unit.output = unit.lookup("coefficients", unit.input(), 1)
    with pytest.raises(ValueError, match="coefficient does not fit signed 64 bits"):
        lower_program(unit, builder, [source])
    builder.output("result", source)
    assert len(builder.build()) == 2
