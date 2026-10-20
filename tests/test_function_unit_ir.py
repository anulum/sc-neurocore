# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — indexed function-unit operand execution and emission

"""Execute distinct operands through the shared integer IR and every target."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import cast

import pytest

from sc_neurocore.compiler.function_unit_ir import Program, emit_c, emit_verilog


def _program() -> Program:
    p = Program()
    x, y, z = p.input(), p.input(1), p.input(2)
    difference = p.sub(x, y)
    p.output = p.add(p.add(p.mul(difference, z), p.input(1)), p.extern("seed"))
    return p


def test_indexed_operands_and_named_externals_are_independent() -> None:
    """Operand order, repeated indices and the seed binding remain distinct."""
    p = _program()
    for x, y, z, seed in [(-7, 2, 3, 11), (9, -3, -5, -2), (0, 0, 0, 42)]:
        assert p.evaluate(x, {"seed": seed}, additional_inputs=(y, z)) == (x - y) * z + y + seed


@pytest.mark.parametrize("index", [-1, True, 1.5, "1"])
def test_invalid_operand_index_is_refused(index: object) -> None:
    """An operand index cannot silently become a negative or fractional lookup."""
    with pytest.raises(ValueError, match="non-negative integer"):
        Program().input(cast(int, index))


@pytest.mark.parametrize("extra", [(), (2,), (2, 3, 4)])
def test_missing_and_extra_operands_are_refused(extra: tuple[int, ...]) -> None:
    """Evaluation and every emitter reject incomplete or discarded operands."""
    p = _program()
    with pytest.raises(ValueError, match="exactly 3 input code"):
        p.evaluate(1, {"seed": 0}, additional_inputs=extra)
    names = tuple(str(x) for x in extra)
    with pytest.raises(ValueError, match="exactly 3 input code"):
        emit_verilog(p, "u", "x", {"seed": "s"}, additional_arguments=names)
    for lang in ("c", "rust"):
        with pytest.raises(ValueError, match="exactly 3 input code"):
            emit_c(p, "u", "x", lang, {"seed": "s"}, additional_arguments=names)


def test_sparse_indices_keep_their_positions() -> None:
    """Unreferenced intervening operands do not renumber the final operand."""
    p = Program()
    p.output = p.sub(p.input(2), p.input())
    assert p.evaluate(7, additional_inputs=(999, -3)) == -10
    with pytest.raises(ValueError, match="exactly 3"):
        p.evaluate(7, additional_inputs=(-3,))


def test_additional_input_overflow_is_detected() -> None:
    """Every referenced operand is checked even when a later select discards it."""
    p = Program()
    p.output = p.select(p.const(0), p.input(1), p.input())
    assert p.evaluate(-4, additional_inputs=(23,)) == -4
    for value in (1 << 63, -(1 << 63)):
        with pytest.raises(OverflowError, match="input.*leaves 64 bits"):
            p.evaluate(-4, additional_inputs=(value,))


def test_unary_text_and_positional_external_bindings_are_preserved() -> None:
    """The additive API keeps previous unary output and positional seed bindings."""
    p = Program()
    p.output = p.add(p.input(), p.extern("seed"))
    assert p.evaluate(7, {"seed": -3}) == 4
    assert emit_verilog(p, "u", "x", {"seed": "s"}) == (
        [
            "wire signed [63:0] u_t0 = $signed(x);",
            "wire signed [63:0] u_t1 = $signed(s);",
            "wire signed [63:0] u_t2 = u_t0 + u_t1;",
        ],
        "u_t2",
    )
    assert emit_c(p, "u", "x", "c", {"seed": "s"}) == (
        [
            "int64_t u_t0 = (int64_t)(x);",
            "int64_t u_t1 = (int64_t)(s);",
            "int64_t u_t2 = u_t0 + u_t1;",
        ],
        {},
        "u_t2",
    )
    assert emit_c(p, "u", "x", "rust", {"seed": "s"}) == (
        [
            "let u_t0: i64 = (x) as i64;",
            "let u_t1: i64 = (s) as i64;",
            "let u_t2: i64 = u_t0 + u_t1;",
        ],
        {},
        "u_t2",
    )


def test_constant_program_keeps_its_original_argument() -> None:
    """Programs that ignore their original input still refuse extra arguments."""
    p = Program()
    p.output = p.const(42)
    assert p.evaluate(-100) == 42
    with pytest.raises(ValueError, match="exactly 1"):
        p.evaluate(0, additional_inputs=(1,))


def _run(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=60).stdout


def _v_literal(value: int) -> str:
    return f"-64'sd{-value}" if value < 0 else f"64'sd{value}"


@pytest.mark.parametrize("lang", ["c", "rust", "verilog"])
def test_native_targets_execute_independent_operands(lang: str, tmp_path: Path) -> None:
    """Actual C, checked Rust and RTL agree with independent signed arithmetic."""
    p = _program()
    cases = [
        (-7, 2, 3, 11),
        (9, -3, -5, -2),
        (0, 0, 0, 42),
        (-(1 << 31), (1 << 31) - 1, -2, 0),
        ((1 << 31) - 1, -(1 << 31), 2, -1),
    ]
    cases += [(x, 5 - 2 * x, x % 7 - 3, x + 17) for x in range(-100, 101)]
    expected = [(x - y) * z + y + seed for x, y, z, seed in cases]
    assert [
        p.evaluate(x, {"seed": s}, additional_inputs=(y, z)) for x, y, z, s in cases
    ] == expected
    executable = tmp_path / "unit"
    if lang == "verilog":
        lines, output = emit_verilog(p, "u", "x", {"seed": "s"}, additional_arguments=("y", "z"))
        stimulus = "\n".join(
            f"x={_v_literal(x)}; y={_v_literal(y)}; z={_v_literal(z)}; s={_v_literal(s)}; "
            f'#1; $display("%0d", {output});'
            for x, y, z, s in cases
        )
        source = tmp_path / "unit.v"
        source.write_text(
            "module tb; reg signed [63:0] x, y, z, s;\n"
            + "\n".join(lines)
            + "\ninitial begin\n"
            + stimulus
            + "\n$finish(0); end endmodule\n"
        )
        _run(["iverilog", "-g2012", "-s", "tb", "-o", str(executable), str(source)])
        result = _run(["vvp", str(executable)])
    else:
        statements, arrays, output = emit_c(
            p, "u", "x", lang, {"seed": "s"}, additional_arguments=("y", "z")
        )
        assert not arrays
        body = "\n".join(statements)
        if lang == "c":
            source = tmp_path / "unit.c"
            calls = "\n".join(
                f'printf("%lld\\n", (long long)unit({x}LL,{y}LL,{z}LL,{s}LL));'
                for x, y, z, s in cases
            )
            source.write_text(
                "#include <stdint.h>\n#include <stdio.h>\n"
                f"int64_t unit(int64_t x,int64_t y,int64_t z,int64_t s) {{\n{body}\nreturn {output};}}\n"
                f"int main(void) {{\n{calls}\nreturn 0;}}\n"
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
            source = tmp_path / "unit.rs"
            calls = "\n".join(
                f'println!("{{}}", unit({x}i64,{y}i64,{z}i64,{s}i64));' for x, y, z, s in cases
            )
            source.write_text(
                f"fn unit(x:i64,y:i64,z:i64,s:i64)->i64 {{\n{body}\n{output}\n}}\n"
                f"fn main() {{\n{calls}\n}}\n"
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
    assert [int(x) for x in result.split()] == expected
