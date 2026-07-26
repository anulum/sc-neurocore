# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation safety AST structure tests

"""Syntax, depth, node allowlist, and leaf-depth contracts."""

from __future__ import annotations

import ast

import pytest

from sc_neurocore.neurons.equation_safety import ExpressionSafetyValidator


def test_syntax_error_is_rejected() -> None:
    """An unparsable string raises with the invalid-syntax message."""
    with pytest.raises(ValueError, match="Invalid equation syntax"):
        ExpressionSafetyValidator().validate("v +")


def test_excessively_deep_ast_is_rejected() -> None:
    """A tree deeper than the limit is refused before the node walk."""
    deep = "1" + "+1" * 25
    with pytest.raises(ValueError, match="AST depth .* exceeds limit"):
        ExpressionSafetyValidator().validate(deep)


def test_disallowed_node_type_is_rejected() -> None:
    """A node type outside the allowlist (a lambda) is refused."""
    with pytest.raises(ValueError, match="Unsafe AST node"):
        ExpressionSafetyValidator().validate("lambda: 1")


def test_ast_depth_of_a_leaf_is_one() -> None:
    """The depth helper reports 1 for a childless node."""
    leaf = ast.parse("1", mode="eval").body
    assert ExpressionSafetyValidator._ast_depth(leaf) == 1
