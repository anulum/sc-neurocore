# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — atan2 reference and numerical-domain contract

"""Check the binary unit against real angles and independent high-precision coefficients."""

from __future__ import annotations

import math

import pytest
import sympy

from sc_neurocore.compiler import function_units as fu
from sc_neurocore.compiler.hardware_numeric_contract import hardware_numeric_contract
from sc_neurocore.compiler.lut_accuracy import measure
from sc_neurocore.compiler.q_format import QFormat
from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.neurons.equation_refusals import EquationRefusal


@pytest.mark.parametrize(
    "dw,fr",
    [
        (2, 0),
        (8, 4),
        (8, 7),
        (16, 8),
        (16, 12),
        (16, 15),
        (18, 9),
        (24, 12),
        (32, 16),
        (32, 24),
        (31, 30),
    ],
)
def test_measured_pair_domain_and_declared_error(dw: int, fr: int) -> None:
    """Every small pair or the declared wider sample respects the arithmetic bound."""
    result = measure("atan2", dw, fr, "v2")
    bound = 0.5 / (1 << fr) + 8 / (1 << fu.P)
    assert result.declared_error_bound == bound
    assert result.max_error <= bound
    assert result.exhaustive == (dw <= 8)
    assert (
        result.codes_evaluated == 1 << (2 * dw)
        if dw <= 8
        else result.codes_evaluated < 1 << (2 * dw)
    )
    low, high = -(1 << (dw - 1)), (1 << (dw - 1)) - 1
    assert result.operand_ranges == ((low / (1 << fr), high / (1 << fr)),) * 2
    y, x = (round(v * (1 << fr)) for v in result.worst_inputs)
    actual = fu.evaluate("atan2", y, dw, fr, additional_inputs=(x,)) / (1 << fr)
    reference = min(high / (1 << fr), max(low / (1 << fr), math.atan2(y, x)))
    assert abs(actual - reference) == pytest.approx(result.max_error, abs=1e-14)
    public = result.to_public_dict()
    assert public["input_arity"] == 2 and public["domain_cardinality"] == 1 << (2 * dw)
    assert public["max_error_at_inputs"] == list(result.worst_inputs)
    assert "table_domain" not in public


@pytest.mark.parametrize(
    "y,x",
    [
        (0, 0),
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
        (-(1 << 31), (1 << 31) - 1),
        (1, -(1 << 31)),
    ],
)
def test_axes_quadrants_and_word_extremes(y: int, x: int) -> None:
    """Order and signs survive reduction, including the canonical-zero convention."""
    actual = fu.evaluate("atan2", y, 32, 24, additional_inputs=(x,)) / (1 << 24)
    assert abs(actual - math.atan2(y, x)) <= 0.5 / (1 << 24) + 8 / (1 << fu.P)


def test_binary_arity_and_domain_are_explicit() -> None:
    """Unknown, missing, extra and unrepresentable operands are refused."""
    with pytest.raises(ValueError, match="exactly 2"):
        fu.evaluate("atan2", 0, 16, 8)
    with pytest.raises(ValueError, match="exactly 2"):
        fu.evaluate("atan2", 0, 16, 8, additional_inputs=(1, 2))
    with pytest.raises(ValueError, match="exactly 1"):
        fu.evaluate("sin", 0, 16, 8, additional_inputs=(1,))
    for y, x in [(-(1 << 15) - 1, 0), (1 << 15, 0), (0, -(1 << 15) - 1), (0, 1 << 15)]:
        with pytest.raises(ValueError, match="outside signed 16-bit domain"):
            fu.evaluate("atan2", y, 16, 8, additional_inputs=(x,))
    with pytest.raises(ValueError, match="words of 2 to 32"):
        fu.evaluate("atan2", 0, 64, 32, additional_inputs=(1,))


def test_atan_coefficients_match_high_precision_interpolation() -> None:
    """Actual coefficient errors satisfy the quantisation premise of the bound."""
    rows = fu.table("atan")
    for index, row in enumerate(rows):
        start = sympy.Rational(index, len(rows))
        middle = start + sympy.Rational(1, 2 * len(rows))
        end = start + sympy.Rational(1, len(rows))
        a, m, z = (sympy.atan(v).evalf(60) * (1 << fu.P) for v in (start, middle, end))
        for actual, expected in zip(row, (a, 4 * m - 3 * a - z, 2 * a + 2 * z - 4 * m)):
            assert abs(float(expected - actual)) <= 0.501


def test_python_neuron_and_hardware_contract_share_operand_order() -> None:
    """The real equation runtime and numeric inspector expose both coordinates."""
    neuron = EquationNeuron(
        equations={"v": "atan2(I, x)"},
        parameters={"x": -2.0},
        state={"v": 0.0},
        method="map",
        dt=1.0,
    )
    neuron.step(1.0)
    assert neuron.state["v"] == pytest.approx(math.atan2(1.0, -2.0))
    contract = hardware_numeric_contract(
        neuron, QFormat.from_string("Q8.8"), function_lowering="v2"
    )
    units = contract.to_public_dict()["function_units"]
    assert isinstance(units, list) and len(units) == 1
    assert units[0]["function"] == "atan2"
    assert units[0]["accuracy"]["input_arity"] == 2
    assert units[0]["accuracy"]["exhaustive"] is False


def test_python_zero_branch_matches_fixed_point_codes() -> None:
    """A negated zero in an actual equation cannot select an unrepresentable sign bit."""
    neuron = EquationNeuron(
        equations={"v": "atan2(-I, -1)"}, state={"v": 0.0}, method="map", dt=1.0
    )
    neuron.step(0.0)
    assert neuron.state["v"] == math.pi


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), -float("inf"), 1j])
def test_python_angle_refuses_nonreal_or_nonfinite_coordinates(coordinate: complex) -> None:
    """The equation runtime rejects coordinates outside the declared real finite domain."""
    from sc_neurocore.neurons.equation_namespace import build_eval_namespace

    angle = build_eval_namespace()["atan2"]
    for y, x in [(coordinate, 1.0), (1.0, coordinate)]:
        with pytest.raises(EquationRefusal, match="finite real coordinates"):
            angle(y, x)
