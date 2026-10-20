# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — function lowering v2: unit programs equal the independent reference

"""Tests of :mod:`sc_neurocore.compiler.function_unit_programs` and :mod:`function_unit_ir`.

The integer programs (emitted to Verilog, C and Rust) and the reference
:mod:`sc_neurocore.compiler.function_units` are written separately; here the
program's own evaluator, which checks that every step stays within 64 bits,
must equal the reference at every input code of Q8.8 and at sampled codes of
Q16.16 and Q8.24. The emitted C and Rust are compiled and compared on sampled
codes (skipped without gcc or rustc); the Verilog path is covered by
``tests/test_function_lowering_v2_cosim.py``.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.compiler import function_units as fu
from sc_neurocore.compiler.function_unit_ir import Program, emit_c, emit_verilog
from sc_neurocore.compiler.function_unit_programs import program


class TestProgramsEqualReference:
    """Program evaluation equals the reference and never leaves 64 bits."""

    @pytest.mark.parametrize("function", fu.UNARY_FUNCTIONS)
    def test_every_code_q8_8(self, function: str) -> None:
        """All 65 536 codes of Q8.8."""
        unit = program(function, 16, 8)
        for code in range(-(1 << 15), 1 << 15):
            assert unit.evaluate(code) == fu.evaluate(function, code, 16, 8), code

    @pytest.mark.parametrize(("dw", "fr"), [(32, 16), (32, 24), (8, 4)])
    @pytest.mark.parametrize("function", fu.UNARY_FUNCTIONS)
    def test_sampled_other_formats(self, function: str, dw: int, fr: int) -> None:
        """Extremes, a grid over the word and a dense window around zero."""
        unit = program(function, dw, fr)
        low, high = -(1 << (dw - 1)), (1 << (dw - 1)) - 1
        codes = {low, high, 0, 1, -1} | set(range(low, high, max(1, (high - low) // 3000)))
        codes |= {c for c in range(-2000, 2000) if low <= c <= high}
        for code in sorted(codes):
            assert unit.evaluate(code) == fu.evaluate(function, code, dw, fr), code


class TestIr:
    """The intermediate representation's own contract."""

    def test_overflow_is_detected(self) -> None:
        """A step that leaves 64 bits raises instead of wrapping silently."""
        p = Program()
        p.output = p.mul(p.const(1 << 40), p.input())
        assert p.evaluate(3) == 3 << 40
        with pytest.raises(OverflowError):
            p.evaluate(1 << 30)

    def test_division_truncates_and_tolerates_an_unselected_zero(self) -> None:
        """Division truncates towards zero; a zero divisor yields 0 (only in unselected branches)."""
        p = Program()
        x = p.input()
        p.output = p.div(p.const(-7), x)
        assert p.evaluate(2) == -3 and p.evaluate(0) == 0

    def test_variable_shifts_are_clamped(self) -> None:
        """Shift amounts outside [0, 63] are clamped, so no negative shift is ever formed."""
        p = Program()
        x = p.input()
        p.output = p.add(p.shl_var(p.const(1), x), p.shr_var(p.const(1 << 10), p.neg(x)))
        assert p.evaluate(-5) == 1 + (1 << 5)  # 1 << -5 is taken as 1 << 0; 1024 >> 5 = 32
        assert p.evaluate(3) == 8 + (1 << 10)  # 1 << 3 = 8; 1024 >> -3 is taken as 1024 >> 0

    def test_unknown_table_and_language_are_refused(self) -> None:
        """Lookups name a known table; emission knows only C and Rust."""
        p = Program()
        with pytest.raises(ValueError, match="unknown table"):
            p.lookup("nope", p.input(), 0)
        with pytest.raises(ValueError, match="unknown language"):
            emit_c(p, "u", "x", lang="go")

    def test_constants_must_fit(self) -> None:
        """A constant outside 64 bits is refused when the program is built."""
        with pytest.raises(ValueError, match="does not fit"):
            Program().const(1 << 63)

    def test_verilog_emission_declares_every_step_once(self) -> None:
        """One wire or table register per step, all under the call-site prefix."""
        unit = program("sigmoid", 16, 8)
        lines, out = emit_verilog(unit, "_fu7", "arg")
        declared = [
            line for line in lines if line.startswith(("wire signed [63:0]", "reg signed [63:0]"))
        ]
        assert len(declared) == len(unit.steps)
        assert out.startswith("_fu7_") and all("_fu7_" in line for line in declared)


_CODES = [-(1 << 15), -1000, -257, -1, 0, 1, 255, 256, 1000, (1 << 15) - 1]


def _run_c(function: str, lang: str) -> list[int]:
    statements, arrays, out = emit_c(program(function, 16, 8), "u", "x", lang)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        if lang == "c":
            tables = "\n".join(
                f"static const int64_t {k}[{len(v)}] = {{{', '.join(f'{x}LL' for x in v)}}};"
                for k, v in arrays.items()
            )
            body = "\n    ".join(statements)
            src = (
                f"#include <stdio.h>\n#include <stdint.h>\n{tables}\n"
                f"static int64_t unit(int64_t x) {{\n    {body}\n    return {out};\n}}\n"
                "int main(void) { static const int64_t cs[] = {"
                + ", ".join(f"{c}LL" for c in _CODES)
                + '}; for (int i = 0; i < 10; i++) printf("%lld\\n", (long long)unit(cs[i])); return 0; }\n'
            )
            (tmp / "u.c").write_text(src)
            subprocess.run(
                [
                    "gcc",
                    "-O1",
                    "-fsanitize=undefined",
                    "-fno-sanitize-recover=all",
                    "-o",
                    str(tmp / "u"),
                    str(tmp / "u.c"),
                ],
                check=True,
                capture_output=True,
            )
        else:
            tables = "\n".join(
                f"static {k}: [i64; {len(v)}] = [{', '.join(f'{x}i64' for x in v)}];"
                for k, v in arrays.items()
            )
            body = "\n    ".join(statements)
            src = (
                f"#![allow(warnings)]\n{tables}\nfn unit(x: i64) -> i64 {{\n    {body}\n    {out}\n}}\n"
                "fn main() { for c in ["
                + ", ".join(f"{c}i64" for c in _CODES)
                + '].iter() { println!("{}", unit(*c)); } }\n'
            )
            (tmp / "u.rs").write_text(src)
            subprocess.run(
                [
                    "rustc",
                    "-C",
                    "overflow-checks=on",
                    "-O",
                    "-o",
                    str(tmp / "u"),
                    str(tmp / "u.rs"),
                ],
                check=True,
                capture_output=True,
            )
        result = subprocess.run([str(tmp / "u")], check=True, capture_output=True, text=True).stdout
    return [int(v) for v in result.split()]


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not available")
@pytest.mark.parametrize("function", fu.UNARY_FUNCTIONS)
def test_emitted_c_equals_reference(function: str) -> None:
    """Compiled C (with undefined-behaviour checks) returns the reference codes."""
    assert _run_c(function, "c") == [fu.evaluate(function, c, 16, 8) for c in _CODES]


@pytest.mark.skipif(shutil.which("rustc") is None, reason="rustc not available")
@pytest.mark.parametrize("function", ["exp", "log", "sin", "exprel"])
def test_emitted_rust_equals_reference(function: str) -> None:
    """Compiled Rust (with overflow checks) returns the reference codes."""
    assert _run_c(function, "rust") == [fu.evaluate(function, c, 16, 8) for c in _CODES]
