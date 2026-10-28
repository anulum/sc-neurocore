# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reciprocal roots in checked integer targets
"""Execute the shared reciprocal-root program in C, Rust and real RTL."""

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.compiler.function_unit_ir import emit_c, emit_verilog
from sc_neurocore.compiler.function_unit_programs import program
from tests.test_function_unit_programs_atan2 import _literal, _run
from tests.test_reciprocal_root_units import decimal_rsqrt


@pytest.mark.parametrize("width", range(2, 9))
def test_all_small_formats_check_every_signed_input(width: int) -> None:
    """Every eager intermediate stays within signed64 across all small word geometries."""
    for fraction in range(width):
        unit = program("rsqrt", width, fraction)
        for code in range(-(1 << (width - 1)), 1 << (width - 1)):
            assert unit.evaluate(code) == decimal_rsqrt(code, width, fraction)


@pytest.mark.parametrize(
    "width,fraction", [(2, 0), (8, 0), (16, 8), (32, 0), (32, 16), (32, 29), (31, 30)]
)
@pytest.mark.parametrize("target", ["c", "rust", "verilog"])
def test_executed_targets_match_decimal(
    width: int, fraction: int, target: str, tmp_path: Path
) -> None:
    """Sanitized C, checked Rust and Icarus agree at word, limb and rounding boundaries."""
    low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
    codes = {low, low + 1, -1, 0, 1, 2, 3, 4, 5, high, high - 1}
    codes.update((1 << bit) + delta for bit in range(width) for delta in (-1, 0, 1))
    samples = sorted(code for code in codes if low <= code <= high)
    expected = [decimal_rsqrt(code, width, fraction) for code in samples]
    unit = program("rsqrt", width, fraction)
    assert [unit.evaluate(code) for code in samples] == expected
    binary = tmp_path / "root"
    if target == "verilog":
        lines, output = emit_verilog(unit, "root", "a")
        calls = "\n".join(f'a={_literal(code)}; #1; $display("%0d",{output});' for code in samples)
        source = tmp_path / "root.v"
        source.write_text(
            "module tb; reg signed [63:0] a;\n"
            + "\n".join(lines)
            + "\ninitial begin\n"
            + calls
            + "\n$finish(0);end endmodule\n"
        )
        _run(["iverilog", "-g2012", "-s", "tb", "-o", str(binary), str(source)])
        result = _run(["vvp", str(binary)])
    else:
        lines, arrays, output = emit_c(unit, "root", "a", target)
        assert arrays == {}
        if target == "c":
            calls = "\n".join(f'printf("%lld\\n",(long long)root({a}LL));' for a in samples)
            source = tmp_path / "root.c"
            source.write_text(
                "#include <stdint.h>\n#include <stdio.h>\nint64_t root(int64_t a){\n"
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
            calls = "\n".join(f'println!("{{}}",root({a}i64));' for a in samples)
            source = tmp_path / "root.rs"
            source.write_text(
                "fn root(a:i64)->i64{\n"
                + "\n".join(lines)
                + f"\n{output}\n}}\nfn main(){{{calls}}}\n"
            )
            command = ["rustc", "-O", "-D", "warnings", "-C", "overflow-checks=on"]
        _run([*command, str(source), "-o", str(binary)])
        result = _run([str(binary)])
    assert [int(code) for code in result.split()] == expected
