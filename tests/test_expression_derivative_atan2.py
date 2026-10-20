# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — atan2 derivatives and branch-cut refusal

"""Differentiate actual equation expressions only on proven smooth angle domains."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.equation_namespace import build_eval_namespace
from sc_neurocore.neurons.equation_program import compile_expression
from sc_neurocore.neurons.expression_derivative import ExpressionDifferentiationError, differentiate


@pytest.mark.parametrize(
    "expr", ["atan2(v, 1)", "atan2(1, v)", "atan2(-1, v)", "atan2(2*v + I, 3)", "v*atan2(v, 1)"]
)
@pytest.mark.parametrize("value", [-2.0, 0.0, 2.0])
def test_smooth_regions_match_finite_differences(expr: str, value: float) -> None:
    """The returned derivative runs in the real DSL and matches a numerical oracle."""
    derivative = compile_expression(differentiate(expr, "v"))
    original = compile_expression(expr)
    env = build_eval_namespace() | {"v": value, "I": 0.25}
    h = 1e-6
    numerical = (
        float(original(env | {"v": value + h})) - float(original(env | {"v": value - h}))
    ) / (2 * h)
    assert float(derivative(env)) == pytest.approx(numerical, abs=1e-7)


@pytest.mark.parametrize(
    "expr", ["atan2(v, -1)", "atan2(0, v)", "atan2(v, x)", "atan2(v, v)", "atan2(v, 0)"]
)
def test_unproven_branch_cut_or_origin_is_refused(expr: str) -> None:
    """A singular or unproven domain cannot silently become a smooth Jacobian."""
    with pytest.raises(ExpressionDifferentiationError, match="provably positive x or nonzero y"):
        differentiate(expr, "v")


def test_state_independent_angle_survives_as_a_coefficient() -> None:
    """A coefficient needs no angle derivative when differentiating another state."""
    derivative = compile_expression(differentiate("v*atan2(I, x)", "v"))
    reference = compile_expression("atan2(I, x)")
    env = build_eval_namespace() | {"v": 2.0, "I": -1.0, "x": -3.0}
    assert derivative(env) == reference(env)


@pytest.mark.parametrize("expr", ["atan2(v)", "atan2(v, 1, 2)", "atan2(v, x=1)"])
def test_derivative_arity_is_not_silently_changed(expr: str) -> None:
    """Symbolic conversion rejects the same unsupported call shapes as lowering."""
    with pytest.raises(ExpressionDifferentiationError, match="two positional"):
        differentiate(expr, "v")
