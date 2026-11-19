# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Circular CORDIC native parity

"""Compile the C and Rust public emissions and compare them with the integer oracle."""

from __future__ import annotations

import itertools
import random
import subprocess
from pathlib import Path

import pytest

from sc_neurocore.compiler import emit_cordic_c, emit_cordic_rust, evaluate_cordic


def _operands(width: int, fraction: int) -> list[tuple[int, int]]:
    """Exercise signed endpoints, axial inputs and unrelated seeded pairs."""
    low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
    edges = range(low, high + 1) if width <= 4 else (low, low + 1, -1, 0, 1, high - 1, high)
    rng = random.Random(width * 31 + fraction)
    return [
        *itertools.product(edges, edges),
        *((rng.randint(low, high), rng.randint(low, high)) for _ in range(80)),
    ]


def _emit_harnesses(
    tmp_path: Path,
    function: str,
    width: int,
    fraction: int,
    operands: list[tuple[int, int]],
) -> None:
    """Write builds that call the external native entry points and check their refusals."""
    binary = function == "atan2"
    c_cases = ",\n".join(f"    {{ {left}, {right} }}" for left, right in operands)
    rust_cases = ",\n".join(f"    ({left}_i32, {right}_i32)" for left, right in operands)
    c_second = ", operands[i][1]" if binary else ""
    rust_second = ", right" if binary else ""
    zero_second = ", 0" if binary else ""
    invalid_code = 1 << (width - 1)
    reject_c = ""
    reject_rust = ""
    if width < 32:
        reject_c = f"if (cordic_evaluate({invalid_code}{zero_second}, &result)) return 2;"
        reject_rust = f"assert!(cordic_evaluate({invalid_code}{zero_second}).is_none());"
        if binary:
            reject_c += f"\n    if (cordic_evaluate(0, {invalid_code}, &result)) return 3;"
            reject_rust += f"\n    assert!(cordic_evaluate(0, {invalid_code}).is_none());"
    (tmp_path / "unit.c").write_text(
        emit_cordic_c(function, data_width=width, fraction=fraction), encoding="utf-8"
    )
    (tmp_path / "main.c").write_text(
        f"""#include <inttypes.h>
#include <stdio.h>
#include "unit.c"

int main(void) {{
    const int32_t operands[][2] = {{
{c_cases}
    }};
    int32_t result;
    if (cordic_evaluate(0{zero_second}, NULL)) return 1;
    {reject_c}
    for (size_t i = 0; i < sizeof operands / sizeof operands[0]; ++i) {{
        if (!cordic_evaluate(operands[i][0]{c_second}, &result)) return 4;
        printf("%" PRId32 "\\n", result);
    }}
    return 0;
}}
""",
        encoding="utf-8",
    )
    (tmp_path / "unit.rs").write_text(
        emit_cordic_rust(function, data_width=width, fraction=fraction), encoding="utf-8"
    )
    (tmp_path / "main.rs").write_text(
        f"""mod unit;
use unit::cordic_evaluate;

fn main() {{
    {reject_rust}
    let operands = [
{rust_cases}
    ];
    for (left, {"right" if binary else "_right"}) in operands {{
        println!("{{}}", cordic_evaluate(left{rust_second}).unwrap());
    }}
}}
""",
        encoding="utf-8",
    )


def _build_and_read(tmp_path: Path, command: list[str], output: str) -> list[int]:
    """Build a native executable and decode its actual result codes."""
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True, timeout=90)
    result = subprocess.run(
        [str(tmp_path / output)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return [int(line) for line in result.stdout.splitlines()]


@pytest.mark.parametrize("function", ("sin", "cos", "arctan", "atan2"))
@pytest.mark.parametrize("width,fraction", ((4, 3), (16, 8), (32, 0), (32, 29), (31, 30)))
def test_emitted_c_and_rust_match_the_public_integer_oracle(
    tmp_path: Path, function: str, width: int, fraction: int
) -> None:
    """Check portable output, wide-angle reduction and quadrant behaviour in native code."""
    operands = _operands(width, fraction)
    expected = [
        evaluate_cordic(
            function,
            left,
            argument1=right if function == "atan2" else None,
            data_width=width,
            fraction=fraction,
        )
        for left, right in operands
    ]
    _emit_harnesses(tmp_path, function, width, fraction, operands)
    actual_c = _build_and_read(
        tmp_path,
        ["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic", "main.c", "-o", "test-c"],
        "test-c",
    )
    actual_rust = _build_and_read(
        tmp_path,
        [
            "rustc",
            "--edition=2021",
            "-Dwarnings",
            "-C",
            "overflow-checks=yes",
            "main.rs",
            "-o",
            "test-rust",
        ],
        "test-rust",
    )
    assert actual_c == expected
    assert actual_rust == expected
