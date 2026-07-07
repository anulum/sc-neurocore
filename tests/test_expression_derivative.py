# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Symbolic partial-derivative tests

"""Tests for the in-grammar symbolic partial derivative used by exponential Euler.

Correctness is checked against a central finite difference (independent of the
symbolic engine's formatting), and the grammar contract is checked by evaluating
the returned derivative string in the same restricted namespace the golden model
uses — so a derivative that is not a valid DSL expression fails loudly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sc_neurocore.neurons.expression_derivative import (
    ExpressionDifferentiationError,
    differentiate,
)

_NAMESPACE: dict[str, Any] = {
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "tanh": np.tanh,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "abs": abs,
    "min": min,
    "max": max,
    "exprel": lambda x: 1.0 if x == 0 else np.expm1(x) / x,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
}


def _evaluate(expr: str, env: dict[str, float]) -> float:
    return float(eval(expr, {"__builtins__": {}}, {**_NAMESPACE, **env}))  # noqa: S307


def _assert_matches_finite_difference(expr: str, wrt: str, env: dict[str, float]) -> str:
    """Differentiate, then check the result against a central finite difference."""
    derivative = differentiate(expr, wrt)
    x = env[wrt]
    step = 1e-6
    forward = _evaluate(expr, {**env, wrt: x + step})
    backward = _evaluate(expr, {**env, wrt: x - step})
    finite_difference = (forward - backward) / (2 * step)
    analytic = _evaluate(derivative, env)
    assert analytic == pytest.approx(finite_difference, abs=1e-4), (
        expr,
        wrt,
        derivative,
        analytic,
        finite_difference,
    )
    return derivative


def test_gating_form_is_exact_reciprocal_time_constant() -> None:
    """The gating derivative ∂/∂x (x_inf - x)/tau is exactly -1/tau."""
    assert differentiate("(x_inf - x)/tau", "x") == "-1/tau"


def test_polynomial_and_rational_derivatives() -> None:
    _assert_matches_finite_difference(
        "a*x**2 + b*x + c", "x", {"x": 1.3, "a": 2.0, "b": -1.0, "c": 5.0}
    )
    _assert_matches_finite_difference(
        "g*(E_l - v) - v/tau", "v", {"v": -55.0, "g": 0.3, "E_l": -70.0, "tau": 10.0}
    )


@pytest.mark.parametrize(
    "expr",
    [
        "exp(-v/k)",
        "log(v)",
        "sqrt(v)",
        "sin(v)",
        "cos(v)",
        "tanh(v)",
        "sinh(v)",
        "cosh(v)",
        "exprel(-v/k)",
        "sigmoid(v)",
    ],
)
def test_smooth_functions_differentiate_in_grammar(expr: str) -> None:
    """Every smooth grammar function differentiates and matches a finite diff."""
    _assert_matches_finite_difference(expr, "v", {"v": 0.7, "k": 3.0})


def test_conductance_rate_expression() -> None:
    """A Hodgkin–Huxley-style gating rhs differentiates cleanly wrt the gate."""
    expr = "0.1*exprel(-(V + 40)/10)*(1 - m) - 4*exp(-(V + 65)/18)*m"
    derivative = _assert_matches_finite_difference(expr, "m", {"m": 0.3, "V": -50.0})
    # The rate rhs is linear in the gate, so A = ∂/∂m is independent of m.
    at_low = _evaluate(derivative, {"m": 0.1, "V": -50.0})
    at_high = _evaluate(derivative, {"m": 0.9, "V": -50.0})
    assert at_low == pytest.approx(at_high)


def test_no_dependence_returns_zero() -> None:
    assert differentiate("I/C", "v") == "0"
    assert differentiate("g*(E_l - w) + I", "v") == "0"


def test_unary_plus_and_negation() -> None:
    assert differentiate("+x", "x") == "1"
    _assert_matches_finite_difference("-x/tau", "x", {"x": 2.0, "tau": 4.0})


def test_integer_and_float_constants_are_preserved() -> None:
    _assert_matches_finite_difference("2*x + 0.5*x**2", "x", {"x": 1.1})


def test_opaque_term_not_depending_on_variable_survives_as_coefficient() -> None:
    """A non-smooth factor independent of the variable round-trips verbatim."""
    derivative = differentiate("abs(V)*x + (x_inf - x)/tau", "x")
    assert "abs(V)" in derivative
    value = _evaluate(derivative, {"V": -3.0, "x_inf": 0.0, "tau": 5.0})
    assert value == pytest.approx(abs(-3.0) - 1.0 / 5.0)


@pytest.mark.parametrize(
    ("factor", "env"),
    [
        ("(a % 3)", {"a": 7.0}),  # unsupported binary operator, independent of x
        ("(a > b)", {"a": 5.0, "b": 1.0}),  # comparison, independent of x
        ("(not flag)", {"flag": False}),  # unary-not, independent of x
        ("clip(V, -80, 40)", {"V": -3.0}),  # non-smooth call, independent of x
    ],
)
def test_opaque_constructs_independent_of_variable_are_admitted(
    factor: str, env: dict[str, float]
) -> None:
    """A non-smooth factor independent of the variable survives as its coefficient."""
    # clip is not in the golden namespace, so provide it for evaluation here.
    local = {**env}
    derivative = differentiate(f"{factor}*x", "x")
    namespace = {**_NAMESPACE, "clip": np.clip}
    got = float(eval(derivative, {"__builtins__": {}}, {**namespace, **local}))  # noqa: S307
    want = float(eval(factor, {"__builtins__": {}}, {**namespace, **local}))  # noqa: S307
    assert got == pytest.approx(want)


@pytest.mark.parametrize(
    "expr",
    [
        "abs(x)",
        "clip(x, -80, 40)",
        "min(x, y)",
        "max(x, y)",
        "x % 3",
        "x // 2",
        "x > 3",
        "unknown_fn(x)",
    ],
)
def test_non_smooth_dependence_on_variable_raises(expr: str) -> None:
    """Differentiating through a non-smooth dependence on the variable is refused."""
    with pytest.raises(ExpressionDifferentiationError):
        differentiate(expr, "x")


@pytest.mark.parametrize("expr", ["True + x", "None + x"])
def test_non_numeric_constant_raises(expr: str) -> None:
    with pytest.raises(ExpressionDifferentiationError):
        differentiate(expr, "x")


def test_call_through_non_name_target_is_opaque_when_independent() -> None:
    """A call whose target is not a plain name is opaque (here, independent of x)."""
    # (a + b)(y) parses as a Call with a BinOp func — not a plain name — and is
    # independent of x, so it survives verbatim as a coefficient rather than
    # raising. It is not evaluable, so only the round-trip is checked.
    derivative = differentiate("(a + b)(y)*x", "x")
    assert "a + b" in derivative
