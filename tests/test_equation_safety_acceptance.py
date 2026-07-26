# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation safety acceptance tests

"""Allowed expressions, attributes, depth configuration, and model regression contracts."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.equation_safety import ExpressionSafetyValidator


def test_valid_expression_passes() -> None:
    """A whitelisted arithmetic/transcendental expression validates cleanly."""
    validator = ExpressionSafetyValidator()
    validator.validate("tanh(v) + a * exp(-w / tau)")


def test_benign_attribute_access_is_permitted() -> None:
    """A non-dunder, non-blocked attribute passes the Attribute node checks."""
    ExpressionSafetyValidator().validate("m.real + 1.0")


def test_custom_max_depth_is_honoured() -> None:
    """The configured depth limit governs the rejection threshold."""
    expr = "1 + 1 + 1"  # depth 5, comfortably under the default
    ExpressionSafetyValidator(max_depth=20).validate(expr)
    with pytest.raises(ValueError, match="AST depth .* exceeds limit 2"):
        ExpressionSafetyValidator(max_depth=2).validate(expr)


def test_bounded_powers_and_nested_bases_are_permitted() -> None:
    """Small integer/rational powers and nested *bases* stay valid dynamics."""
    validator = ExpressionSafetyValidator()
    validator.validate("v**2")
    validator.validate("x**0.5")
    validator.validate("(a**2)**3")


def test_montbrio_rate_equation_still_validates() -> None:
    """Regression guard: the MPR firing-rate expression is not caught by the new caps."""
    ExpressionSafetyValidator().validate("delta/(3.141592653589793*tau**2) + 2.0*r*v/tau")
