# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual compiled equation error boundaries

"""Exercise compiled values through EquationNeuron's public constructor and step."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.neurons.equation_refusals import EquationEvaluationFailure, EquationRefusal
from sc_neurocore.refusals import AuthoredRefusal


@pytest.mark.parametrize(
    ("expression", "category", "reason"),
    [
        ("1 / 0", ZeroDivisionError, "the equation divides by zero"),
        ("(1e15)**64", OverflowError, "the equation value exceeds its numeric range"),
        ("exp(1000)", FloatingPointError, "the equation arithmetic could not remain finite"),
        ("[v]", TypeError, "the equation value has an invalid type"),
        ("[v][2]", IndexError, "the equation index is outside its value"),
        ("v.missing", AttributeError, "the equation reads an unavailable attribute"),
        ("missing", NameError, "the equation reads an unavailable symbol"),
        ("min([])", ValueError, "the equation value is invalid"),
    ],
)
def test_expected_expression_failure_preserves_its_exception_category(
    expression: str, category: type[Exception], reason: str
) -> None:
    """Keep existing exception handlers while withholding interpreter text."""
    neuron = EquationNeuron(equations={"v": expression}, state={"v": 1.0}, dt=0.1)
    with np.errstate(over="raise"), pytest.raises(category) as refused:
        neuron.step()
    assert isinstance(refused.value, EquationEvaluationFailure)
    assert str(refused.value) == reason
    assert neuron.state == {"v": 1.0}
    assert refused.value.__cause__ is not None


def test_existing_domain_guard_passes_through_the_evaluation_boundary() -> None:
    """Preserve the square-root guard's original authored explanation."""
    neuron = EquationNeuron(equations={"v": "sqrt(v)"}, state={"v": -1.0})
    with pytest.raises(EquationRefusal, match="^sqrt domain error$") as refused:
        neuron.step()
    assert not isinstance(refused.value, EquationEvaluationFailure)


def test_a_non_scalar_threshold_is_an_authored_value_failure() -> None:
    """Apply the same value boundary to an actual bool conversion."""
    neuron = EquationNeuron(equations={"v": "0"}, state={"v": 1.0}, threshold="exprel([v, v])")
    with pytest.raises(EquationEvaluationFailure, match="^the equation value is invalid$"):
        neuron.step()


def test_a_non_scalar_reset_is_an_authored_type_failure() -> None:
    """Guard the actual reset conversion without changing its spike condition."""
    neuron = EquationNeuron(
        equations={"v": "0"}, state={"v": 1.0}, threshold="True", reset={"v": "[v]"}
    )
    with pytest.raises(TypeError, match="^the equation value has an invalid type$") as refused:
        neuron.step()
    assert isinstance(refused.value, EquationEvaluationFailure)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("euler", 0.98),
        ("map", -0.2),
        ("gauss_seidel", 0.98),
        ("rk4", 0.9801986733333333),
        ("exp_euler", math.exp(-0.02)),
    ],
)
def test_valid_method_recurrences_keep_their_numerical_results(
    method: str, expected: float
) -> None:
    """Run each integration method against its unchanged one-step recurrence."""
    neuron = EquationNeuron(equations={"v": "-0.2 * v"}, state={"v": 1.0}, dt=0.1, method=method)
    assert neuron.step() == 0
    assert neuron.state["v"] == pytest.approx(expected, abs=1e-14)


def test_engine_state_corruption_is_not_marked_as_an_expression_refusal() -> None:
    """Keep a real state-update fault outside the expression error boundary."""
    neuron = EquationNeuron(equations={"v": "1"}, state={"v": 1.0})
    neuron.state.clear()
    with pytest.raises(KeyError) as failure:
        neuron.step()
    assert not isinstance(failure.value, AuthoredRefusal)
