# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verilog emitter: call arguments, n-ary max/min and named constants

"""Calls and named constants in :mod:`sc_neurocore.compiler.verilog_expr_emitter`.

Until 2026-10-18 ``max(a, b, c)`` was lowered as ``max(a, b)``, ``clip(x)`` and
``max(x)`` as ``x``, keyword arguments were ignored, and ``pi`` was emitted as an
undeclared net. The Python model computes something else in each case (or
raises), and the bit-true C kernel shared the same lowering, so co-simulation
could not see it. The simulation cases compare RTL and kernel with the Python
model's value, quantised once; they skip when Icarus Verilog or gcc is absent.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.compiler.verilog_expr_emitter import _emit_expr
from sc_neurocore.neurons.equation_builder import EquationNeuron
from tests.bit_true_cosim_support import HAS_COSIM, _c_trace, _q_input, _verilog_trace

Q = Q88(data_width=16, fraction=8)


def _emit(text: str, params: dict[str, str] | None = None) -> tuple[str, list[str]]:
    result, intermediates, *_ = _emit_expr(text, {"v": "v"}, params or {}, Q)
    return result, intermediates


class TestArguments:
    """Every argument is lowered, or the call is refused."""

    def test_max_with_three_arguments_compares_all(self) -> None:
        """The third argument appears in the lowered select."""
        result, _ = _emit("max(v, 0.5, 3.0)")
        inner = _emit("max(v, 0.5)")[0]
        literal = Q.encode_signed_literal(3.0)
        # Folded left, max(max(v, 0.5), 3.0), in the same select form as the two-argument case.
        assert result == f"(((({inner}) + 32'sd0) > (({literal}) + 32'sd0)) ? {inner} : {literal})"

    def test_two_argument_max_is_unchanged(self) -> None:
        """The two-argument form keeps its earlier text, so existing RTL is byte-identical."""
        result, _ = _emit("max(v, 0.5)")
        assert result == "((((v_reg) + 32'sd0) > ((16'sd128) + 32'sd0)) ? v_reg : 16'sd128)"

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("max(v)", "at least 2"),
            ("min(v)", "at least 2"),
            ("clip(v)", "exactly 3"),
            ("clip(v, 1.0)", "exactly 3"),
            ("exp(v, 2.0)", "exactly 1"),
            ("clip(v, a_min=0.0, a_max=1.0)", "positional arguments only"),
        ],
    )
    def test_refusals(self, text: str, message: str) -> None:
        """Calls the model would not evaluate as lowered are refused."""
        with pytest.raises(ValueError, match=message):
            _emit(text)

    def test_unsupported_function_message_is_unchanged(self) -> None:
        """An unknown function still names itself in the refusal."""
        with pytest.raises(ValueError, match="Unsupported function 'gamma'"):
            _emit("gamma(v)")


class TestNamedConstants:
    """``pi`` becomes a literal unless a parameter takes its name."""

    def test_pi_is_a_literal(self) -> None:
        """``2*pi`` multiplies by the encoded constant, not by a net called ``pi``."""
        _, intermediates = _emit("2.0 * pi")
        assert any(Q.encode_signed_literal(math.pi) in line for line in intermediates)
        assert not any(line.rstrip(";").endswith("* pi") for line in intermediates)

    def test_parameter_named_pi_takes_precedence(self) -> None:
        """A parameter ``pi`` is read as that parameter, as in the Python model."""
        result, _ = _emit("pi", params={"pi": "P_PI"})
        assert result == "P_PI"


@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog / gcc not available")
class TestAgainstThePythonModel:
    """RTL and C kernel equal the Python model's value for the repaired constructs."""

    @pytest.mark.parametrize(
        ("expression", "x"),
        [
            ("max(I, 0.5, 3.0)", 0.0),
            ("max(I, 0.5, 3.0)", 4.0),
            ("min(I, -0.5, 1.0, -2.0)", 0.0),
            ("2.0 * pi * I", 0.5),
            ("clip(I, 1.0, 0.0)", 0.5),  # reversed bounds: numpy.clip gives the upper bound, 0
            ("clip(I, -1.0, 1.0)", 2.0),
            ("clip(I, -1.0, 1.0)", -3.0),
        ],
    )
    def test_map_output_equals_model(self, expression: str, x: float) -> None:
        """A map neuron ``v = expression`` outputs the model's value in Q8.8."""
        neuron = EquationNeuron(equations={"v": expression}, state={"v": 0.0}, dt=1.0, method="map")
        neuron.step(x)
        expected = _q_input(float(neuron.state["v"]), 16, 8)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            rtl = _verilog_trace(neuron, "calls_map", x, 2, 16, 8, tmp)
            kernel = _c_trace(neuron, "calls_map", _q_input(x, 16, 8), 2, 16, 8, tmp)
        # 2*pi*0.5 is one multiply of encoded operands: allow the truncation of that product.
        tolerance = 2 if "pi" in expression else 0
        assert abs(int(rtl[-1][1]) - expected) <= tolerance
        assert rtl[-1] == kernel[-1]
