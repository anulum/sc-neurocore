# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — softplus integer programs executed in C, Rust and Verilog

"""Execute bounded exponent/logarithm composition on native targets and all input codes."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import pytest

from sc_neurocore.compiler import function_units as fu
from sc_neurocore.compiler.function_unit_ir import emit_c, emit_verilog
from sc_neurocore.compiler.function_unit_programs import program


def _run(command: list[str]) -> str:
    """Run a real native tool and retain its diagnostics when it refuses the input."""
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return result.stdout


def _codes(width: int, fraction: int) -> list[int]:
    """Choose signed extrema and both sides of negative-exponential reductions."""
    low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
    values = {low, low + 1, high - 1, high, -1, 0, 1}
    values.update(range(max(low, -1024), min(high, 1024) + 1, 17))
    for k in range(34):
        for sign in (-1, 1):
            centre = round(sign * k * math.log(2.0) * (1 << fraction))
            values.update(centre + d for d in (-1, 0, 1))
    return sorted(x for x in values if low <= x <= high)


def test_every_q8_8_code_matches_the_reference_without_overflow() -> None:
    """The actual integer program checks all intermediates at every signed 16-bit input."""
    unit = program("softplus", 16, 8)
    for code in range(-(1 << 15), 1 << 15):
        assert unit.evaluate(code) == fu.evaluate("softplus", code, 16, 8), code


@pytest.mark.parametrize(
    "dw,fr",
    [(2, 0), (8, 4), (16, 8), (16, 12), (18, 9), (32, 0), (32, 16), (32, 24), (32, 29), (31, 30)],
)
@pytest.mark.parametrize("target", ["c", "rust", "verilog"])
def test_executed_units_match_reference(target: str, dw: int, fr: int, tmp_path: Path) -> None:
    """Real target compilers preserve signed minima, transitions and final saturation."""
    codes = _codes(dw, fr)
    expected = [fu.evaluate("softplus", x, dw, fr) for x in codes]
    unit = program("softplus", dw, fr)
    assert [unit.evaluate(x) for x in codes] == expected
    executable = tmp_path / "unit"
    if target == "verilog":
        lines, output = emit_verilog(unit, "u", "x")
        literals = [f"-64'sd{-x}" if x < 0 else f"64'sd{x}" for x in codes]
        calls = "\n".join(f'x={x}; #1; $display("%0d", {output});' for x in literals)
        source = tmp_path / "unit.v"
        source.write_text(
            "module tb; reg signed [63:0] x;\n"
            + "\n".join(lines)
            + "\ninitial begin\n"
            + calls
            + "\n$finish(0); end endmodule\n"
        )
        _run(["iverilog", "-g2012", "-s", "tb", "-o", str(executable), str(source)])
        actual = _run(["vvp", str(executable)])
    else:
        lines, arrays, output = emit_c(unit, "u", "x", target)
        body = "\n".join(lines)
        if target == "c":
            tables = "\n".join(
                f"static const int64_t {name}[] = {{" + ",".join(f"{x}LL" for x in values) + "};"
                for name, values in arrays.items()
            )
            calls = "\n".join(f'printf("%lld\\n", (long long)f({x}LL));' for x in codes)
            source = tmp_path / "unit.c"
            source.write_text(
                "#include <stdint.h>\n#include <stdio.h>\n"
                + tables
                + f"\nint64_t f(int64_t x){{\n{body}\nreturn {output};}}\n"
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
            calls = "\n".join(f'println!("{{}}", f({x}i64));' for x in codes)
            source = tmp_path / "unit.rs"
            source.write_text(
                f"fn f(x:i64)->i64{{\n{tables}\n{body}\n{output}\n}}\n"
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
        actual = _run([str(executable)])
    assert [int(row) for row in actual.split()] == expected
