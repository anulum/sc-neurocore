# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Standalone function hardware public contract

"""Exercise the public standalone unit API through real RTL and integer oracles."""

from __future__ import annotations

import math
from pathlib import Path
import random
import subprocess
from typing import cast

import pytest

from sc_neurocore.compiler import emit_function_unit_verilog, function_unit_functions
from sc_neurocore.compiler import expr_lut_tables, function_units, lut_accuracy

FORMATS = [(16, 8), (16, 12), (32, 16), (18, 9), (32, 24), (16, 15)]
UNITS = [
    (lowering, function)
    for lowering in ("lut-v1", "v2")
    for function in function_unit_functions(lowering)
]


def _operands(width: int, fraction: int, arity: int) -> list[tuple[int, ...]]:
    """Cover signed endpoints, branch boundaries and independent binary operands."""
    low, high, one = -(1 << (width - 1)), (1 << (width - 1)) - 1, 1 << fraction
    values = {low, low + 1, -1, 0, 1, high - 1, high}
    for value in (-16.0, -2.0, -1.0, -0.125, 0.125, 1.0, 2.0, 16.0, math.pi / 2, math.pi):
        code = round(value * one)
        values.update(x for x in (code - 1, code, code + 1) if low <= x <= high)
    rng = random.Random(716 + width + fraction)
    values.update(rng.randint(low, high) for _ in range(24))
    ordered = sorted(values)
    if arity == 1:
        return [(value,) for value in ordered]
    pairs = {
        (value, ordered[(index * 7 + 3) % len(ordered)]) for index, value in enumerate(ordered)
    }
    edges = [low, -1, 0, 1, high]
    pairs.update((left, right) for left in edges for right in edges)
    pairs.update((0, value) for value in ordered)
    return sorted(pairs)


def _simulate(
    tmp_path: Path, source: str, width: int, operands: list[tuple[int, ...]]
) -> list[int]:
    """Compile and execute an emitted module with genuine independent signed ports."""
    arity = len(operands[0])
    declarations = [f"reg signed [{width - 1}:0] argument{i};" for i in range(arity)]
    ports = [f".argument{i}(argument{i})" for i in range(arity)] + [".result(result)"]
    statements: list[str] = []
    mask = (1 << width) - 1
    for values in operands:
        statements.extend(
            f"argument{i} = {width}'h{value & mask:x};" for i, value in enumerate(values)
        )
        statements.append('#1; $display("%0d", $signed(result));')
    bench = (
        "module tb;\n"
        + "\n".join(declarations)
        + f"\nwire signed [{width - 1}:0] result;\n"
        + f"function_unit dut({', '.join(ports)});\ninitial begin\n"
        + "\n".join(statements)
        + "\n$finish(0);\nend\nendmodule\n"
    )
    (tmp_path / "unit.v").write_text(source, encoding="utf-8")
    (tmp_path / "tb.v").write_text(bench, encoding="utf-8")
    try:
        subprocess.run(
            ["iverilog", "-g2012", "-s", "tb", "-o", "sim", "unit.v", "tb.v"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        result = subprocess.run(
            ["vvp", "sim"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return [int(line) for line in result.stdout.splitlines()]
    finally:
        for name in ("sim", "unit.v", "tb.v"):
            (tmp_path / name).unlink(missing_ok=True)


def test_catalogue_matches_the_existing_accuracy_and_exact_function_surfaces() -> None:
    """Keep the new standalone catalogue complete against independent public inventories."""
    assert set(function_unit_functions()) == set(lut_accuracy.table_functions("v2"))
    assert set(function_unit_functions("lut-v1")) == (
        set(lut_accuracy.table_functions()) | expr_lut_tables.EXACT_FUNCTIONS
    )
    assert "expit" not in function_unit_functions()
    assert not expr_lut_tables.RANDOM_FUNCTIONS.intersection(function_unit_functions())


@pytest.mark.parametrize("width,fraction", [(2, 0), (2, 1), (3, 0), (3, 2), (4, 0), (4, 3)])
@pytest.mark.parametrize("lowering,function", UNITS)
def test_narrow_rtl_exhausts_independent_signed_inputs(
    tmp_path: Path, width: int, fraction: int, lowering: str, function: str
) -> None:
    """Exhaust small words, including LUT indices wider than their input ports."""
    values = range(-(1 << (width - 1)), 1 << (width - 1))
    operands: list[tuple[int, ...]] = (
        [(value,) for value in values]
        if function_units.FUNCTION_ARITY[function] == 1
        else [(left, right) for left in values for right in values]
    )
    expected = [
        lut_accuracy.lut_read_int(function, arguments[0], width, fraction)
        if lowering == "lut-v1" and function not in expr_lut_tables.EXACT_FUNCTIONS
        else function_units.evaluate(
            function, arguments[0], width, fraction, additional_inputs=arguments[1:]
        )
        for arguments in operands
    ]
    source = emit_function_unit_verilog(
        function, data_width=width, fraction=fraction, function_lowering=lowering
    )
    assert _simulate(tmp_path, source, width, operands) == expected


@pytest.mark.parametrize("width,fraction", FORMATS)
@pytest.mark.parametrize("lowering,function", UNITS)
def test_standalone_rtl_matches_existing_integer_semantics(
    tmp_path: Path,
    width: int,
    fraction: int,
    lowering: str,
    function: str,
) -> None:
    """Check every supported function/format/lowering on actual generated hardware."""
    source = emit_function_unit_verilog(
        function,
        data_width=width,
        fraction=fraction,
        function_lowering=lowering,
    )
    operands = _operands(width, fraction, function_units.FUNCTION_ARITY[function])
    expected = [
        lut_accuracy.lut_read_int(function, values[0], width, fraction)
        if lowering == "lut-v1" and function not in expr_lut_tables.EXACT_FUNCTIONS
        else function_units.evaluate(
            function,
            values[0],
            width,
            fraction,
            additional_inputs=values[1:],
        )
        for values in operands
    ]
    assert _simulate(tmp_path, source, width, operands) == expected


@pytest.mark.parametrize("lowering", ["lut-v1", "v2"])
def test_expit_alias_is_an_executable_sigmoid_unit(tmp_path: Path, lowering: str) -> None:
    """Preserve the model namespace alias through the actual standalone RTL surface."""
    operands = _operands(16, 8, 1)
    source = emit_function_unit_verilog("expit", function_lowering=lowering)
    expected = [
        lut_accuracy.lut_read_int("sigmoid", values[0], 16, 8)
        if lowering == "lut-v1"
        else function_units.evaluate("sigmoid", values[0], 16, 8)
        for values in operands
    ]
    assert _simulate(tmp_path, source, 16, operands) == expected


@pytest.mark.parametrize("width,fraction", [(2, 0), (32, 0), (31, 30), (32, 29)])
@pytest.mark.parametrize("lowering,function", [("v2", "sign"), ("v2", "sin"), ("lut-v1", "sqrt")])
def test_format_limits_execute_the_selected_integer_semantics(
    tmp_path: Path, width: int, fraction: int, lowering: str, function: str
) -> None:
    """Exercise zero fractional bits and admitted signed-width limits in actual RTL."""
    source = emit_function_unit_verilog(
        function, data_width=width, fraction=fraction, function_lowering=lowering
    )
    operands = _operands(width, fraction, 1)
    expected = [
        lut_accuracy.lut_read_int(function, values[0], width, fraction)
        if lowering == "lut-v1"
        else function_units.evaluate(function, values[0], width, fraction)
        for values in operands
    ]
    assert _simulate(tmp_path, source, width, operands) == expected


@pytest.mark.parametrize("function", ["rand", "abs", "clip", "max", "unknown", "sin); bad()"])
def test_refuses_constructs_outside_the_canonical_standalone_units(function: str) -> None:
    """Reject unsupported constructs before parsing any caller-authored expression text."""
    with pytest.raises(ValueError, match="has no"):
        emit_function_unit_verilog(function)


@pytest.mark.parametrize("function", sorted(expr_lut_tables.V2_ONLY_FUNCTIONS))
def test_refuses_v2_only_functions_in_legacy_mode(function: str) -> None:
    """Never invent a legacy lowering or publish a zero-cost unsupported function."""
    with pytest.raises(ValueError, match="has no"):
        emit_function_unit_verilog(function, function_lowering="lut-v1")


def test_refuses_unknown_lowering_in_both_public_entrypoints() -> None:
    """Fail closed for a misspelled mode in discovery and hardware generation."""
    with pytest.raises(ValueError, match="lowering"):
        function_unit_functions("invalid")
    with pytest.raises(ValueError, match="lowering"):
        emit_function_unit_verilog("sin", function_lowering="invalid")


@pytest.mark.parametrize("function", [None, 12, ["sin"]])
def test_refuses_non_string_function_at_the_public_boundary(function: object) -> None:
    """Reject malformed deserialised function names with a stable admission error."""
    with pytest.raises(TypeError, match="function must"):
        emit_function_unit_verilog(cast(str, function))


@pytest.mark.parametrize("width,fraction", [(1, 0), (33, 8), (16, -1), (16, 16), (32, 30)])
def test_refuses_formats_outside_the_existing_integer_unit_contract(
    width: int, fraction: int
) -> None:
    """Apply the same representability limits as the shared numerical function units."""
    with pytest.raises(ValueError):
        emit_function_unit_verilog("sin", data_width=width, fraction=fraction)


@pytest.mark.parametrize("width,fraction", [(True, 8), (16, False), (16.0, 8), (16, 8.0)])
def test_refuses_non_integer_geometry(width: object, fraction: object) -> None:
    """Preserve strict integer geometry rather than accepting coerced or Boolean widths."""
    with pytest.raises(TypeError):
        emit_function_unit_verilog("sin", data_width=cast(int, width), fraction=cast(int, fraction))
