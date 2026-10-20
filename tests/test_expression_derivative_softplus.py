# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — stable softplus derivatives in the equation language

"""Run smooth softplus Jacobians without expanding an overflowing exponential."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.neurons.equation_namespace import build_eval_namespace
from sc_neurocore.neurons.equation_program import compile_expression
from sc_neurocore.neurons.expression_derivative import differentiate


@pytest.mark.parametrize("value", [-1000.0, -100.0, -30.0, -5.0, 0.0, 5.0, 30.0, 100.0, 1000.0])
def test_first_and_second_derivatives_are_stable(value: float) -> None:
    """Actual DSL derivative programs agree with a stable independent logistic formula."""
    derivative = differentiate("softplus(v)", "v")
    second = differentiate(derivative, "v")
    sigma = 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))
    env = build_eval_namespace() | {"v": value}
    with np.errstate(over="raise", invalid="raise"):
        assert float(compile_expression(derivative)(env)) == pytest.approx(sigma, abs=1e-15)
        assert float(compile_expression(second)(env)) == pytest.approx(
            sigma * (1 - sigma), abs=1e-15
        )
    assert "sigmoid" in derivative and "exp(" not in derivative


@pytest.mark.parametrize(
    "expression", ["softplus(2*v+I)", "v*softplus(v)", "softplus(v)*softplus(I)"]
)
@pytest.mark.parametrize("value", [-2.0, 0.0, 2.0])
def test_composed_derivatives_match_finite_differences(expression: str, value: float) -> None:
    """Product and chain rules retain the same real authored nonlinear expression."""
    original = compile_expression(expression)
    derivative = compile_expression(differentiate(expression, "v"))
    env = build_eval_namespace() | {"v": value, "I": 0.25}
    h = 1e-6
    estimate = (
        float(original(env | {"v": value + h})) - float(original(env | {"v": value - h}))
    ) / (2 * h)
    assert float(derivative(env)) == pytest.approx(estimate, abs=1e-8)


def test_state_independent_softplus_survives_without_expansion() -> None:
    """A large finite coefficient stays a softplus call in the returned equation."""
    text = differentiate("v*softplus(I)", "v")
    assert (
        "softplus" in text
        and float(compile_expression(text)(build_eval_namespace() | {"I": 1000.0})) == 1000.0
    )
