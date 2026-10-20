# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — softplus reference, domain and arithmetic error

"""Compare stable integer softplus with independent numerical and coefficient oracles."""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest
import sympy

from sc_neurocore.compiler import function_units as fu
from sc_neurocore.compiler.hardware_numeric_contract import hardware_numeric_contract
from sc_neurocore.compiler.lut_accuracy import measure
from sc_neurocore.compiler.q_format import QFormat
from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.neurons.equation_namespace import build_eval_namespace
from sc_neurocore.neurons.equation_refusals import EquationRefusal


@pytest.mark.parametrize(
    "dw,fr",
    [
        (2, 0),
        (8, 0),
        (8, 4),
        (8, 7),
        (16, 0),
        (16, 8),
        (16, 12),
        (16, 15),
        (18, 9),
        (24, 12),
        (32, 0),
        (32, 16),
        (32, 24),
        (32, 29),
        (31, 30),
    ],
)
def test_full_word_measurements_respect_the_arithmetic_bound(dw: int, fr: int) -> None:
    """Exhaustive small words and declared wider samples meet a separate analytic bound."""
    result = measure("softplus", dw, fr, "v2")
    bound = 0.5 / (1 << fr) + 16.0 / (1 << fu.P)
    assert result.max_error <= bound and result.declared_error_bound == bound
    assert result.exhaustive == (dw <= 16)
    assert result.codes_evaluated == 1 << dw if dw <= 16 else result.codes_evaluated < 1 << dw
    public = result.to_public_dict()
    low, high = -(1 << (dw - 1)), (1 << (dw - 1)) - 1
    assert public["input_arity"] == 1 and public["domain_cardinality"] == 1 << dw
    assert public["operand_ranges"] == [[low / (1 << fr), high / (1 << fr)]]
    assert public["declared_error_bound"] == bound and "table_domain" not in public
    code = round(result.worst_inputs[0] * (1 << fr))
    reference = min(high / (1 << fr), float(np.logaddexp(0.0, code / (1 << fr))))
    actual = fu.evaluate("softplus", code, dw, fr) / (1 << fr)
    assert actual >= 0.0 and abs(actual - reference) == pytest.approx(result.max_error, abs=1e-15)


@pytest.mark.parametrize(
    "dw,fr", [(16, 8), (16, 12), (32, 0), (32, 16), (32, 24), (32, 29), (31, 30)]
)
def test_reduction_and_logarithm_segment_boundaries(dw: int, fr: int) -> None:
    """Probe both sides of range-reduction and logarithm-grid crossings, plus word extrema."""
    scale = 1 << fr
    low, high = -(1 << (dw - 1)), (1 << (dw - 1)) - 1
    points = {low, low + 1, high - 1, high, -1, 0, 1}
    for value in [k * math.log(2.0) for k in range(65)] + [
        -math.log(k / 512) for k in range(1, 513)
    ]:
        for sign in (-1, 1):
            centre = round(sign * value * scale)
            points.update(centre + offset for offset in (-2, -1, 0, 1, 2))
    bound = 0.5 / scale + 16.0 / (1 << fu.P)
    for code in sorted(c for c in points if low <= c <= high):
        actual = fu.evaluate("softplus", code, dw, fr) / scale
        reference = min(high / scale, float(np.logaddexp(0.0, code / scale)))
        assert 0.0 <= actual <= high / scale and abs(actual - reference) <= bound, (dw, fr, code)


@pytest.mark.parametrize("table", ["exp2", "log2"])
def test_reused_coefficients_satisfy_the_interpolation_premise(table: str) -> None:
    """Sixty-digit interpolation coefficients independently bound the actual integer tables."""
    rows = fu.table(table)
    for index, actual in enumerate(rows):
        points = [sympy.Rational(2 * index + k, 2 * len(rows)) for k in range(3)]
        values = [
            (sympy.Pow(2, x) if table == "exp2" else sympy.log(1 + x, 2)).evalf(60) * (1 << fu.P)
            for x in points
        ]
        a, m, z = values
        for got, expected in zip(actual, (a, 4 * m - 3 * a - z, 2 * a + 2 * z - 4 * m)):
            assert abs(float(expected - got)) <= 0.501
    exact = 1 / sympy.log(2).evalf(60)
    actual_log2e = sympy.Rational((fu._LOG2E_HIGH << 31) + fu._LOG2E_LOW, 1 << (fu.P + 31))
    assert abs(float(exact - actual_log2e)) <= 2.0**-50
    assert abs(float(sympy.log(2).evalf(60) * (1 << fu.P) - fu._LN2)) <= 0.501


@pytest.mark.parametrize(
    "value", [-sys.float_info.max, -1000.0, -0.0, 0.0, 1000.0, sys.float_info.max]
)
def test_finite_python_tails_survive_an_ambient_float_trap(value: float) -> None:
    """The actual map remains finite when mathematically negligible tails underflow."""
    neuron = EquationNeuron({"v": "softplus(I)"}, state={"v": 0.0}, method="map", dt=1.0)
    with np.errstate(all="raise"):
        neuron.step(value)
    assert math.isfinite(neuron.state["v"])
    with np.errstate(under="ignore"):
        assert neuron.state["v"] == float(np.logaddexp(0.0, value))


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan, 1 + 2j])
def test_nonfinite_or_complex_coordinates_are_refused(value: float | complex) -> None:
    """Unsupported inputs fail through the public namespace with an authored refusal."""
    with pytest.raises(EquationRefusal, match="finite real inputs"):
        build_eval_namespace()["softplus"](value)


def test_python_array_shape_and_finite_domain_are_preserved() -> None:
    """The actual namespace evaluates a shaped vector and refuses any invalid element."""
    values = np.array([[-1000.0, -2.0, -0.0], [0.0, 2.0, 1000.0]])
    function = build_eval_namespace()["softplus"]
    with np.errstate(all="raise"):
        actual = function(values)
    with np.errstate(under="ignore"):
        np.testing.assert_array_equal(actual, np.logaddexp(0.0, values))
    assert actual.shape == (2, 3)
    with pytest.raises(EquationRefusal, match="finite real inputs"):
        function(np.array([0.0, math.inf]))


def test_numeric_contract_discloses_the_real_domain_and_bound() -> None:
    """A real neuron inspector reports the same scalar function selected by the compiler."""
    neuron = EquationNeuron({"v": "softplus(I)"}, state={"v": 0.0}, method="map", dt=1.0)
    contract = hardware_numeric_contract(
        neuron, QFormat.from_string("Q8.8"), function_lowering="v2"
    )
    assert contract.arithmetic is not None
    units = contract.to_public_dict()["function_units"]
    assert isinstance(units, list) and len(units) == 1
    assert units[0]["function"] == "softplus"
    assert units[0]["accuracy"]["input_arity"] == 1
    assert units[0]["accuracy"]["declared_error_bound"] == 2**-9 + 16 * 2**-30


def test_unit_domain_arity_and_old_lowering_are_explicit() -> None:
    """The new operation admits one signed code on the declared v2 formats only."""
    for code in [-(1 << 15) - 1, 1 << 15]:
        with pytest.raises(ValueError, match="outside signed 16-bit domain"):
            fu.evaluate("softplus", code, 16, 8)
    with pytest.raises(ValueError, match="exactly 1"):
        fu.evaluate("softplus", 0, 16, 8, additional_inputs=(1,))
    with pytest.raises(ValueError, match="words of 2 to 32"):
        fu.evaluate("softplus", 0, 64, 32)
    with pytest.raises(ValueError, match="no lut-v1 table"):
        measure("softplus", 16, 8)
