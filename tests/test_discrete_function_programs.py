# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Discontinuous integer-program semantics
"""Execute the maintained public program factory against independent word results."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from sc_neurocore.compiler.function_unit_ir import emit_c, emit_verilog
from tests.test_function_unit_programs_atan2 import _run, _literal

import pytest

from sc_neurocore.compiler.function_unit_programs import program


@pytest.mark.parametrize(
    ("width", "fraction"), [(2, 0), (8, 4), (8, 7), (16, 8), (16, 12), (32, 16), (32, 24)]
)
def test_round_program_keeps_negative_ties_and_saturation(width: int, fraction: int) -> None:
    """Public programs round signed halves to even before clamping the final code."""
    unit = program("round", width, fraction)
    scale = 1 << fraction
    low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
    codes = {low, high, -1, 0, 1}
    for integer in (-4, -3, -2, -1, 0, 1, 2, 3):
        codes.update(integer * scale + scale // 2 + delta for delta in (-1, 0, 1))
    for code in sorted(value for value in codes if low <= value <= high):
        expected = max(low, min(high, round(Fraction(code, scale)) * scale))
        assert unit.evaluate(code) == expected


@pytest.mark.parametrize(
    ("width", "fraction"), [(2, 0), (8, 4), (8, 7), (16, 8), (16, 12), (32, 16), (32, 24)]
)
def test_heaviside_program_reads_the_second_operand_only_at_zero(width: int, fraction: int) -> None:
    """Changing the second word affects zero inputs and preserves the whole word there."""
    unit = program("heaviside", width, fraction)
    low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
    for code in (low, -1, 0, 1, high):
        for at_zero in (low, -1, 0, 1, high):
            expected = 0 if code < 0 else at_zero if code == 0 else min(1 << fraction, high)
            assert unit.evaluate(code, additional_inputs=(at_zero,)) == expected


@pytest.mark.parametrize(
    "dw,fr", [(2, 0), (8, 4), (8, 7), (16, 8), (16, 12), (32, 16), (32, 24), (31, 30)]
)
@pytest.mark.parametrize("function", ["round", "heaviside"])
@pytest.mark.parametrize("target", ["c", "rust", "verilog"])
def test_executed_discrete_targets_equal_reference(
    function: str, target: str, dw: int, fr: int, tmp_path: Path
) -> None:
    """Real generated code preserves half ties, both operand extrema and fractional-only saturation."""
    low, high = -(1 << (dw - 1)), (1 << (dw - 1)) - 1
    scale = 1 << fr
    axis = {low, high, -1, 0, 1}
    axis.update(n * scale + scale // 2 + d for n in range(-4, 4) for d in (-1, 0, 1))
    axis = {x for x in axis if low <= x <= high}
    pairs = [(x, z) for x in sorted(axis) for z in (low, -1, 0, 1, high)]
    expected = [
        max(low, min(high, round(Fraction(y, scale)) * scale))
        if function == "round"
        else 0
        if y < 0
        else x
        if y == 0
        else min(scale, high)
        for y, x in pairs
    ]
    unit = program(function, dw, fr)
    additional = ("x",) if function == "heaviside" else ()
    assert [
        unit.evaluate(y, additional_inputs=(x,) if additional else ()) for y, x in pairs
    ] == expected
    executable = tmp_path / "unit"
    if target == "verilog":
        lines, output = emit_verilog(unit, "u", "y", additional_arguments=additional)
        stimulus = "\n".join(
            f'y={_literal(y)}; x={_literal(x)}; #1; $display("%0d", {output});' for y, x in pairs
        )
        source = tmp_path / "unit.v"
        source.write_text(
            "module tb; reg signed [63:0] y, x;\n"
            + "\n".join(lines)
            + "\ninitial begin\n"
            + stimulus
            + "\n$finish(0); end endmodule\n"
        )
        _run(["iverilog", "-g2012", "-s", "tb", "-o", str(executable), str(source)])
        result = _run(["vvp", str(executable)])
    else:
        lines, arrays, output = emit_c(unit, "u", "y", target, additional_arguments=additional)
        body = "\n".join(lines)
        if target == "c":
            tables = "\n".join(
                f"static const int64_t {name}[] = {{" + ",".join(f"{x}LL" for x in values) + "};"
                for name, values in arrays.items()
            )
            calls = "\n".join(f'printf("%lld\\n", (long long)f({y}LL,{x}LL));' for y, x in pairs)
            source = tmp_path / "unit.c"
            source.write_text(
                "#include <stdint.h>\n#include <stdio.h>\n"
                + tables
                + f"\nint64_t f(int64_t y,int64_t x){{\n(void)x;\n{body}\nreturn {output};}}\n"
                + f"int main(void){{\n{calls}\nreturn 0;}}\n"
            )
            _run(
                [
                    "gcc",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-fsanitize=undefined",
                    "-fno-sanitize-recover=all",
                    str(source),
                    "-o",
                    str(executable),
                ]
            )
        else:
            tables = "\n".join(
                f"let {name}: [i64; {len(values)}] = [" + ",".join(f"{x}i64" for x in values) + "];"
                for name, values in arrays.items()
            )
            calls = "\n".join(f'println!("{{}}", f({y}i64,{x}i64));' for y, x in pairs)
            source = tmp_path / "unit.rs"
            source.write_text(
                f"fn f(y:i64,x:i64)->i64{{\nlet _ = x;\n{tables}\n{body}\n{output}\n}}\n"
                + f"fn main(){{\n{calls}\n}}\n"
            )
            _run(
                [
                    "rustc",
                    "-D",
                    "warnings",
                    "-C",
                    "overflow-checks=on",
                    "-O",
                    str(source),
                    "-o",
                    str(executable),
                ]
            )
        result = _run([str(executable)])
    assert [int(value) for value in result.split()] == expected
