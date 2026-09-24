# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — What the RTL and bit-true lowering refuse rather than approximate

"""Expressions the RTL or the bit-true kernel cannot lower exactly are refused."""

from __future__ import annotations

import pytest

from sc_neurocore.compiler.c_fixed_emitter import emit_c_fixed_expr
from sc_neurocore.compiler.intelligence.bit_true_kernel import generate_bittrue_kernel_from_neuron
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.neurons.equation_builder import EquationNeuron


def _map(expression: str) -> EquationNeuron:
    return EquationNeuron(equations={"x": expression}, method="map", dt=1.0)


@pytest.mark.parametrize("language", ["verilog", "c"])
def test_and_or_over_numbers_is_refused(language: str) -> None:
    """Python's ``a or b`` returns an operand, not a truth value."""
    neuron = _map("1.0 if (x or x > 1.0) else 0.0")
    with pytest.raises(ValueError, match="'and' / 'or' operands must be comparisons"):
        if language == "verilog":
            compile_to_verilog(neuron)
        else:
            generate_bittrue_kernel_from_neuron(neuron)


@pytest.mark.parametrize("language", ["verilog", "c"])
def test_a_truth_value_needs_a_format_that_holds_one(language: str) -> None:
    neuron = _map("x * (x > 0.0)")
    with pytest.raises(ValueError, match=r"needs 1\.0, which Q1\.15 cannot hold"):
        if language == "verilog":
            compile_to_verilog(neuron, data_width=16, fraction=15)
        else:
            generate_bittrue_kernel_from_neuron(neuron, data_width=16, fraction=15)


def _balanced_sum(terms: int) -> str:
    """Write ``x + x + ...`` as a balanced tree, within the schema depth limit."""
    if terms == 1:
        return "x"
    half = terms // 2
    return f"({_balanced_sum(half)} + {_balanced_sum(terms - half)})"


def test_a_sum_the_wide_datapath_cannot_hold_exactly_is_refused() -> None:
    """At 8 bits the 16-bit datapath holds at most 255 word-sized terms."""
    with pytest.raises(ValueError, match="sums 256 word-sized terms"):
        compile_to_verilog(_map(_balanced_sum(256)), data_width=8, fraction=4)
    compile_to_verilog(_map(_balanced_sum(255)), data_width=8, fraction=4)


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("x // 2.0", "positive integer power-of-two literal"),
        ("x // 3", "positive integer power-of-two literal"),
        ("x // 256", "exceeds fixed-point maximum"),
    ],
)
def test_the_kernel_refuses_the_floor_divisors_the_rtl_refuses(
    expression: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        emit_c_fixed_expr(expression, {"x": "s->x"}, {}, Q88(data_width=16, fraction=8))
    with pytest.raises(ValueError, match=message):
        compile_to_verilog(_map(expression))


def test_the_kernel_refuses_what_it_does_not_mirror() -> None:
    poisson = EquationNeuron(
        equations={"v": "-v + I"}, detection="poisson", probability_expression="sigmoid(v)"
    )
    with pytest.raises(ValueError, match="does not mirror stochastic spike detection"):
        generate_bittrue_kernel_from_neuron(poisson)
    substepped = EquationNeuron(
        equations={"v": "-v + I"},
        threshold="v > 1.0",
        detection="crossing",
        substeps=4,
        dt=0.25,
    )
    with pytest.raises(ValueError, match=r"does not mirror macro-step sub-stepping \(substeps=4\)"):
        generate_bittrue_kernel_from_neuron(substepped)


def test_operators_neither_lowering_supports_are_refused() -> None:
    with pytest.raises(ValueError, match="Unsupported binary op: LShift"):
        emit_c_fixed_expr("x << 1", {"x": "s->x"}, {}, Q88(data_width=16, fraction=8))
