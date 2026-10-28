# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reciprocal root differentiation domains
"""Run symbolic reciprocal-root derivatives through the real equation evaluator."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.equation_namespace import build_eval_namespace
from sc_neurocore.neurons.equation_program import compile_expression
from sc_neurocore.neurons.expression_derivative import ExpressionDifferentiationError, differentiate


@pytest.mark.parametrize("expression", ["rsqrt(v*v+1)", "v*rsqrt(v*v+2)", "rsqrt(rsqrt(v*v+1)+1)"])
def test_chain_rule_matches_finite_differences(expression: str) -> None:
    """Positive real arguments retain correct product and nested chain rules."""
    source = compile_expression(expression)
    derivative = compile_expression(differentiate(expression, "v"))
    for at in [-3.0, -0.1, 0.0, 0.1, 3.0]:
        env = build_eval_namespace() | {"v": at}
        h = 1e-5
        estimate = (float(source(env | {"v": at + h})) - float(source(env | {"v": at - h}))) / (
            2 * h
        )
        assert float(derivative(env)) == pytest.approx(estimate, abs=1e-8)


@pytest.mark.parametrize("expression", ["rsqrt(v)", "rsqrt(v*v)", "rsqrt(-v*v-1)"])
def test_unproven_positive_domains_refuse(expression: str) -> None:
    """The Jacobian never crosses zero or invents positivity from a squared zero."""
    with pytest.raises(ExpressionDifferentiationError, match="provably positive"):
        differentiate(expression, "v")


@pytest.mark.parametrize("expression", ["rsqrt()", "rsqrt(v,1)", "rsqrt(v,x=1)"])
def test_wrong_call_signatures_refuse(expression: str) -> None:
    """Differentiation preserves the unary positional-only function contract."""
    with pytest.raises(ExpressionDifferentiationError, match="exactly one"):
        differentiate(expression, "v")


def test_independent_root_remains_an_opaque_coefficient() -> None:
    """A parameter coefficient keeps its reciprocal root in the generated grammar."""
    derivative = compile_expression(differentiate("v*rsqrt(I)", "v"))
    assert derivative(build_eval_namespace() | {"I": 4.0}) == 0.5
