# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Euclidean magnitude across integer program targets
"""Execute norm programs in checked C, overflow-checked Rust and real RTL."""

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.compiler.function_unit_ir import emit_c, emit_verilog
from sc_neurocore.compiler.function_unit_programs import program
from tests.test_function_unit_programs_atan2 import _literal, _run
from tests.test_magnitude_function_units import decimal_norm


@pytest.mark.parametrize("width", [2, 3, 4, 5])
def test_small_norm_programs_exhaustively_preserve_integer_geometry(width: int) -> None:
    """All input pairs agree with a decimal oracle and every intermediate fits signed64."""
    unit = program("hypot", width, width - 1)
    for first in range(-(1 << (width - 1)), 1 << (width - 1)):
        for second in range(-(1 << (width - 1)), 1 << (width - 1)):
            assert unit.evaluate(first, additional_inputs=(second,)) == decimal_norm(
                first, second, width
            )


@pytest.mark.parametrize("width,fraction", [(2, 0), (8, 4), (16, 8), (18, 9), (32, 16), (31, 30)])
@pytest.mark.parametrize("target", ["c", "rust", "verilog"])
def test_executed_norm_targets_round_and_saturate_exactly(
    width: int, fraction: int, target: str, tmp_path: Path
) -> None:
    """Actual target execution preserves both minima, near-saturation roots and symmetry."""
    low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
    axis = {low, low + 1, high, high - 1, -1, 0, 1, 2, 3, 4, 5, 12, 13}
    coordinates = sorted(code for code in axis if low <= code <= high)
    pairs = [(a, b) for a in coordinates for b in coordinates]
    expected = [decimal_norm(a, b, width) for a, b in pairs]
    unit = program("hypot", width, fraction)
    assert [unit.evaluate(a, additional_inputs=(b,)) for a, b in pairs] == expected
    binary = tmp_path / "norm"
    if target == "verilog":
        lines, output = emit_verilog(unit, "norm", "a", additional_arguments=("b",))
        calls = "\n".join(
            f'a={_literal(a)}; b={_literal(b)}; #1; $display("%0d",{output});' for a, b in pairs
        )
        source = tmp_path / "norm.v"
        source.write_text(
            "module tb; reg signed [63:0] a,b;\n"
            + "\n".join(lines)
            + "\ninitial begin\n"
            + calls
            + "\n$finish(0);end endmodule\n"
        )
        _run(["iverilog", "-g2012", "-s", "tb", "-o", str(binary), str(source)])
        result = _run(["vvp", str(binary)])
    else:
        lines, arrays, output = emit_c(unit, "norm", "a", target, additional_arguments=("b",))
        assert arrays == {}
        if target == "c":
            calls = "\n".join(f'printf("%lld\\n",(long long)norm({a}LL,{b}LL));' for a, b in pairs)
            source = tmp_path / "norm.c"
            source.write_text(
                "#include <stdint.h>\n#include <stdio.h>\nint64_t norm(int64_t a,int64_t b){\n"
                + "\n".join(lines)
                + f"\nreturn {output};}}\nint main(void){{{calls}return 0;}}\n"
            )
            command = [
                "gcc",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-fsanitize=undefined",
                "-fno-sanitize-recover=all",
            ]
        else:
            calls = "\n".join(f'println!("{{}}",norm({a}i64,{b}i64));' for a, b in pairs)
            source = tmp_path / "norm.rs"
            source.write_text(
                "fn norm(a:i64,b:i64)->i64{\n"
                + "\n".join(lines)
                + f"\n{output}\n}}\nfn main(){{{calls}}}\n"
            )
            command = ["rustc", "-O", "-D", "warnings", "-C", "overflow-checks=on"]
        _run([*command, str(source), "-o", str(binary)])
        result = _run([str(binary)])
    assert [int(code) for code in result.split()] == expected
