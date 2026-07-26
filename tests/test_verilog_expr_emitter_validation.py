# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Focused Verilog expression lowering contracts

"""Focused fixed-point Verilog expression lowering contracts."""

from .verilog_expr_emitter_support import *


def test_unsupported_unary_operator_fails_closed() -> None:
    """Unsupported unary operators are rejected instead of lowered silently."""
    with pytest.raises(ValueError, match="Unsupported unary op"):
        _emit_expr("~v", _STATE_VARS, {}, Q88())


def test_unsupported_ast_node_fails_closed() -> None:
    """Unsupported AST nodes are rejected by the generic visitor."""
    with pytest.raises(ValueError, match="Unsupported AST node"):
        _emit_expr("[v]", _STATE_VARS, {}, Q88())


def test_if_expression_lowers_to_a_verilog_ternary() -> None:
    """Piecewise expressions retain their comparison, true branch, and false branch."""
    result, *_ = _emit_expr("v if v > 0.0 else -v", _STATE_VARS, {}, Q88())

    assert result.startswith("(((v_reg >")
    assert ") ? (v_reg) : ((-v_reg)))" in result
