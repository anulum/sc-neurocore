# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — binary atan2 integer programs on every target

"""Execute quadrant-aware integer programs in Python, checked C, Rust and RTL."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sc_neurocore.compiler import function_units as fu
from sc_neurocore.compiler.function_unit_ir import emit_c, emit_verilog
from sc_neurocore.compiler.function_unit_programs import program


def _cases(width: int) -> list[tuple[int, int]]:
    low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
    axis = [low, low + 1, -257, -256, -2, -1, 0, 1, 2, 255, 256, high - 1, high]
    axis = sorted({c for c in axis if low <= c <= high})
    return [(y, x) for y in axis for x in axis]


def _run(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=120).stdout


def _literal(value: int) -> str:
    return f"-64'sd{-value}" if value < 0 else f"64'sd{value}"


def test_all_eight_bit_pairs_equal_the_integer_reference() -> None:
    """The IR, with every intermediate checked, equals the independent reference."""
    unit = program("atan2", 8, 4)
    for y in range(-128, 128):
        for x in range(-128, 128):
            assert unit.evaluate(y, additional_inputs=(x,)) == fu.evaluate(
                "atan2", y, 8, 4, additional_inputs=(x,)
            ), (y, x)


def test_unknown_binary_factory_name_is_refused() -> None:
    """The shared unit factory does not accept an unregistered operation."""
    with pytest.raises(ValueError, match="no v2 unit"):
        program("not_a_function", 16, 8)


@pytest.mark.parametrize("dw,fr", [(16, 8), (16, 12), (16, 15), (18, 9), (32, 16), (32, 24)])
@pytest.mark.parametrize("target", ["c", "rust", "verilog"])
def test_executed_targets_equal_reference(target: str, dw: int, fr: int, tmp_path: Path) -> None:
    """Actual native code handles all quadrants, axes, equal ratios and extrema."""
    pairs = _cases(dw)
    expected = [fu.evaluate("atan2", y, dw, fr, additional_inputs=(x,)) for y, x in pairs]
    unit = program("atan2", dw, fr)
    assert [unit.evaluate(y, additional_inputs=(x,)) for y, x in pairs] == expected
    executable = tmp_path / "unit"
    if target == "verilog":
        lines, output = emit_verilog(unit, "u", "y", additional_arguments=("x",))
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
        lines, arrays, output = emit_c(unit, "u", "y", target, additional_arguments=("x",))
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
                + f"\nint64_t f(int64_t y,int64_t x){{\n{body}\nreturn {output};}}\n"
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
                f"fn f(y:i64,x:i64)->i64{{\n{tables}\n{body}\n{output}\n}}\n"
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
