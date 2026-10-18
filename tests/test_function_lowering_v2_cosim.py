# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — function lowering v2 through the compiler: RTL, bit-true kernel and reference agree

"""Whole-neuron co-simulation of ``function_lowering="v2"``.

A map neuron ``v = f(I)`` is compiled with :func:`compile_to_verilog` and with
:func:`generate_bittrue_kernel_from_neuron`, both with ``function_lowering="v2"``,
driven with constant inputs in Icarus Verilog and gcc, and compared bit for bit
with each other and with the integer reference
(:func:`sc_neurocore.compiler.function_units.evaluate`). An Euler neuron with a
transcendental drift checks the unit inside the accumulate path. The cases skip
when Icarus Verilog or gcc is absent.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.compiler import function_units as fu
from sc_neurocore.compiler.intelligence.bit_true_kernel import generate_bittrue_kernel_from_neuron
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron, from_equations
from tests.bit_true_cosim_support import HAS_COSIM, _c_main, _q_input, _rows, _testbench

_EXPRESSIONS = {
    "exp": "exp(I)",
    "log": "log(I)",
    "sqrt": "sqrt(I)",
    "sin": "sin(I)",
    "cos": "cos(I)",
    "sigmoid": "sigmoid(I)",
    "tanh": "tanh(I)",
    "cosh": "cosh(I)",
    "sinh": "sinh(I)",
    "exprel": "exprel(I)",
    "cbrt": "I ** (1.0 / 3.0)",
}
_POSITIVE = {"log", "sqrt", "cbrt"}


def _traces(
    neuron: EquationNeuron, x: float, dw: int, fr: int, steps: int
) -> tuple[list[list[str]], list[list[str]]]:
    module = "fu_v2"
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / f"{module}.v").write_text(
            compile_to_verilog(neuron, module, dw, fr, function_lowering="v2")
        )
        (tmp / "tb.v").write_text(_testbench(neuron, module, x, steps, dw, fr))
        subprocess.run(
            [
                "iverilog",
                "-g2012",
                "-o",
                str(tmp / "sim"),
                str(tmp / f"{module}.v"),
                str(tmp / "tb.v"),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        rtl = _rows(
            subprocess.run(
                ["vvp", str(tmp / "sim")], capture_output=True, text=True, timeout=120
            ).stdout
        )
        kernel = generate_bittrue_kernel_from_neuron(
            neuron, module, data_width=dw, fraction=fr, function_lowering="v2"
        )
        (tmp / f"{module}.c").write_text(
            kernel + "\n" + _c_main(neuron, module, _q_input(x, dw, fr), steps, dw)
        )
        subprocess.run(
            [
                "gcc",
                "-O2",
                "-fsanitize=undefined",
                "-fno-sanitize-recover=all",
                "-o",
                str(tmp / "k"),
                str(tmp / f"{module}.c"),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        c = _rows(
            subprocess.run([str(tmp / "k")], capture_output=True, text=True, timeout=60).stdout
        )
    return rtl, c


@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog / gcc not available")
class TestMapNeurons:
    """Each unit as a map neuron: RTL = kernel = reference."""

    @pytest.mark.parametrize(
        ("dw", "fr"), [(16, 8), (16, 12), (32, 16)], ids=["Q8.8", "Q4.12", "Q16.16"]
    )
    @pytest.mark.parametrize("function", sorted(_EXPRESSIONS))
    def test_unit_output(self, function: str, dw: int, fr: int) -> None:
        """For inputs across the format, the neuron outputs the reference unit's code."""
        neuron = EquationNeuron(
            equations={"v": _EXPRESSIONS[function]}, state={"v": 0.0}, dt=1.0, method="map"
        )
        top = min(20.0, (1 << (dw - fr - 1)) - 1.0)
        inputs = [0.3, 1.7, top] if function in _POSITIVE else [-top, -1.25, 0.0, 0.0625, 2.5, top]
        for x in inputs:
            rtl, c = _traces(neuron, x, dw, fr, 2)
            expected = fu.evaluate(function, _q_input(x, dw, fr), dw, fr)
            assert int(rtl[-1][1]) == expected, f"RTL {function}({x}) at Q{dw - fr}.{fr}"
            assert rtl == c, f"kernel differs from RTL for {function}({x})"


@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog / gcc not available")
def test_euler_neuron_with_units_in_the_drift() -> None:
    """An Euler neuron whose drift uses tanh and exp runs bit-identically in RTL and kernel."""
    neuron = from_equations(
        "dv/dt = 0.2*(tanh(v) - v) + 0.05*exp(-v) + I", init=dict(v=0.0), dt=0.5
    )
    rtl, c = _traces(neuron, 0.75, 16, 8, 40)
    assert len(rtl) == 40
    assert rtl == c


def test_lut_v1_default_is_unchanged() -> None:
    """Without ``function_lowering`` the RTL is the lut-v1 RTL, byte for byte."""
    neuron = EquationNeuron(equations={"v": "tanh(I)"}, state={"v": 0.0}, dt=1.0, method="map")
    assert compile_to_verilog(neuron, "m") == compile_to_verilog(
        neuron, "m", function_lowering="lut-v1"
    )
    assert "v2 function unit" in compile_to_verilog(neuron, "m", function_lowering="v2")


def test_unknown_lowering_is_refused() -> None:
    """A lowering name outside ``lut-v1``/``v2`` is refused by the format."""
    neuron = EquationNeuron(equations={"v": "tanh(I)"}, state={"v": 0.0}, dt=1.0, method="map")
    with pytest.raises(ValueError, match="function_lowering must be one of"):
        compile_to_verilog(neuron, "m", function_lowering="v3")


def test_sinh_needs_v2() -> None:
    """``sinh`` has no lut-v1 table and is refused there with the remedy named."""
    neuron = EquationNeuron(equations={"v": "sinh(I)"}, state={"v": 0.0}, dt=1.0, method="map")
    with pytest.raises(ValueError, match="needs function_lowering='v2'"):
        compile_to_verilog(neuron, "m")


def test_wide_words_are_refused_for_v2() -> None:
    """A 64-bit word cannot be mirrored by the 64-bit units and is refused, not wrapped."""
    neuron = EquationNeuron(equations={"v": "exp(I)"}, state={"v": 0.0}, dt=1.0, method="map")
    with pytest.raises(ValueError, match="words of 2 to 32 bits"):
        compile_to_verilog(neuron, "m", 64, 32, function_lowering="v2")
