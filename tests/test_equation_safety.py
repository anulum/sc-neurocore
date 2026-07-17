# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the equation expression safety gate

"""Unit tests for the AST-level equation safety validator.

This is the security boundary that makes every ``# nosec B307`` eval site in
the equation runner sound, so the tests exercise each rejection path (syntax,
depth, disallowed node type, blocked name, dunder attribute, blocked attribute)
as well as the accepting path for a benign attribute access.
"""

from __future__ import annotations

import ast

import pytest

from sc_neurocore.neurons.equation_safety import EVAL_GLOBALS, ExpressionSafetyValidator


def test_valid_expression_passes() -> None:
    """A whitelisted arithmetic/transcendental expression validates cleanly."""
    validator = ExpressionSafetyValidator()
    validator.validate("tanh(v) + a * exp(-w / tau)")


def test_benign_attribute_access_is_permitted() -> None:
    """A non-dunder, non-blocked attribute passes the Attribute node checks."""
    ExpressionSafetyValidator().validate("m.real + 1.0")


def test_syntax_error_is_rejected() -> None:
    """An unparsable string raises with the invalid-syntax message."""
    with pytest.raises(ValueError, match="Invalid equation syntax"):
        ExpressionSafetyValidator().validate("v +")


def test_excessively_deep_ast_is_rejected() -> None:
    """A tree deeper than the limit is refused before the node walk."""
    deep = "1" + "+1" * 25
    with pytest.raises(ValueError, match="AST depth .* exceeds limit"):
        ExpressionSafetyValidator().validate(deep)


def test_custom_max_depth_is_honoured() -> None:
    """The configured depth limit governs the rejection threshold."""
    expr = "1 + 1 + 1"  # depth 5, comfortably under the default
    ExpressionSafetyValidator(max_depth=20).validate(expr)
    with pytest.raises(ValueError, match="AST depth .* exceeds limit 2"):
        ExpressionSafetyValidator(max_depth=2).validate(expr)


def test_disallowed_node_type_is_rejected() -> None:
    """A node type outside the allowlist (a lambda) is refused."""
    with pytest.raises(ValueError, match="Unsafe AST node"):
        ExpressionSafetyValidator().validate("lambda: 1")


def test_blocked_name_is_rejected() -> None:
    """A bare blocked identifier is refused as a blocked function."""
    with pytest.raises(ValueError, match="Blocked function 'os'"):
        ExpressionSafetyValidator().validate("os")


def test_dunder_attribute_access_is_rejected() -> None:
    """Any double-underscore attribute access is refused outright."""
    with pytest.raises(ValueError, match="Dunder attribute access"):
        ExpressionSafetyValidator().validate("v.__class__")


def test_blocked_attribute_is_rejected() -> None:
    """A non-dunder attribute whose name is on the block list is refused."""
    with pytest.raises(ValueError, match="Blocked attribute 'subprocess'"):
        ExpressionSafetyValidator().validate("v.subprocess")


def test_ast_depth_of_a_leaf_is_one() -> None:
    """The depth helper reports 1 for a childless node."""
    leaf = ast.parse("1", mode="eval").body
    assert ExpressionSafetyValidator._ast_depth(leaf) == 1


def test_eval_globals_is_an_empty_builtins_sandbox() -> None:
    """The eval globals carry a truly empty ``__builtins__`` — no builtin is reachable."""
    assert set(EVAL_GLOBALS) == {"__builtins__"}
    assert EVAL_GLOBALS["__builtins__"] == {}


def test_empty_builtins_sandbox_denies_import_even_without_the_name_block() -> None:
    """Defence in depth: the empty ``__builtins__`` is an independent second barrier.

    The AST name-block rejects ``__import__`` before compilation, but the empty
    ``__builtins__`` in :data:`EVAL_GLOBALS` means even a compiled expression that
    slipped the validator cannot resolve ``__import__`` — or any other builtin —
    at eval time; the vector no longer rests on the name-block alone.
    """
    for expression in ("__import__", "abs", "eval"):
        compiled = compile(expression, "<sandbox-test>", "eval")
        with pytest.raises(NameError):
            eval(compiled, EVAL_GLOBALS, {})  # noqa: S307 - deliberately exercising the sandbox


def test_nested_exponentiation_is_rejected() -> None:
    """Chained exponents (``9**9**9**9``) blow up under eval and are refused."""
    with pytest.raises(ValueError, match="Nested exponentiation blocked"):
        ExpressionSafetyValidator().validate("9**9**9**9")


def test_oversized_literal_exponent_is_rejected() -> None:
    """A single ``**`` with a huge literal exponent is refused before eval."""
    with pytest.raises(ValueError, match="Exponent .* exceeds limit"):
        ExpressionSafetyValidator().validate("9 ** 999999")


def test_oversized_numeric_constant_is_rejected() -> None:
    """A literal whose magnitude overflows the sandbox ceiling is refused."""
    with pytest.raises(ValueError, match="exceeds magnitude limit"):
        ExpressionSafetyValidator().validate("1e400")


def test_bounded_powers_and_nested_bases_are_permitted() -> None:
    """Small integer/rational powers and nested *bases* stay valid dynamics."""
    validator = ExpressionSafetyValidator()
    validator.validate("v**2")
    validator.validate("x**0.5")
    validator.validate("(a**2)**3")


def test_montbrio_rate_equation_still_validates() -> None:
    """Regression guard: the MPR firing-rate expression is not caught by the new caps."""
    ExpressionSafetyValidator().validate("delta/(3.141592653589793*tau**2) + 2.0*r*v/tau")
