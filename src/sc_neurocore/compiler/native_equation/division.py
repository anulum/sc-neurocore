# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Authored equation division admission

"""Keep user divisions distinct from totalized structured function-program divisions."""

from __future__ import annotations

import ast
import math

from ...neurons.equation_builder import EquationNeuron
from ..equation_lowering import equation_expressions
from ..verilog_compiler_config import Q88


def validate_division(neuron: EquationNeuron, q: Q88) -> None:
    """Require an encoded nonzero denominator or a preserved nonzero source branch.

    Parameters
    ----------
    neuron : EquationNeuron
        Original expressions before introducing totalized compiler intrinsics.
    q : Q88
        Word geometry used to encode constants and comparison boundaries.

    Raises
    ------
    ValueError
        A source divisor may encode zero. Parameter division is retained as a
        quotient; its AST is never rewritten to literal reciprocal arithmetic.
    """
    fixed = {**neuron.parameters, **neuron.constants}
    changing = {*neuron.equations, *(f"{name}_prev" for name in neuron.equations), "I"}
    nonzero = {
        name for name, value in fixed.items() if name not in changing and q.encode(value) != 0
    }

    def literal_code(node: ast.AST) -> int | None:
        if isinstance(node, ast.Constant):
            value = node.value
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                code = q.encode(value)
                return code - (1 << q.data_width) if code >= 1 << (q.data_width - 1) else code
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            nested_code = literal_code(node.operand)
            return (
                None
                if nested_code is None
                else -nested_code
                if isinstance(node.op, ast.USub)
                else nested_code
            )
        return None

    def guarded(node: ast.AST, truth: bool) -> set[str]:
        if isinstance(node, ast.Compare) and len(node.ops) > 1:
            lefts = [node.left, *node.comparators[:-1]]
            branches = [
                guarded(ast.Compare(left=left, ops=[op], comparators=[right]), truth)
                for left, op, right in zip(lefts, node.ops, node.comparators, strict=True)
            ]
            return set().union(*branches) if truth else set.intersection(*branches)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right, operation = node.left, node.comparators[0], node.ops[0]
            if isinstance(right, ast.Name) and literal_code(left) is not None:
                left, right = right, left
                operation = {
                    ast.Lt: ast.Gt,
                    ast.LtE: ast.GtE,
                    ast.Gt: ast.Lt,
                    ast.GtE: ast.LtE,
                }.get(type(operation), ast.Eq)()
            if isinstance(left, ast.Name):
                code = literal_code(right)
                if code is not None:
                    positive = (
                        isinstance(operation, ast.Gt)
                        and code >= 0
                        or isinstance(operation, ast.GtE)
                        and code > 0
                    )
                    negative = (
                        isinstance(operation, ast.Lt)
                        and code <= 0
                        or isinstance(operation, ast.LtE)
                        and code < 0
                    )
                    false_positive = (
                        isinstance(operation, ast.Lt)
                        and code > 0
                        or isinstance(operation, ast.LtE)
                        and code >= 0
                    )
                    false_negative = (
                        isinstance(operation, ast.Gt)
                        and code < 0
                        or isinstance(operation, ast.GtE)
                        and code <= 0
                    )
                    if positive or negative if truth else false_positive or false_negative:
                        return {left.id}
        if isinstance(node, ast.BoolOp):
            branches = [guarded(child, truth) for child in node.values]
            if isinstance(node.op, ast.And) == truth:
                return set().union(*branches)
            return set.intersection(*branches)
        return set()

    def visit(node: ast.AST, known: set[str], where: str) -> None:
        if isinstance(node, ast.IfExp):
            visit(node.test, known, where)
            visit(node.body, known | guarded(node.test, True), where)
            visit(node.orelse, known | guarded(node.test, False), where)
            return
        if isinstance(node, ast.BoolOp):
            active = set(known)
            for child in node.values:
                visit(child, active, where)
                active.update(guarded(child, isinstance(node.op, ast.And)))
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            denominator = node.right
            literal = (
                isinstance(denominator, ast.Constant)
                and isinstance(denominator.value, (int, float))
                and not isinstance(denominator.value, bool)
                and math.isfinite(denominator.value)
                and denominator.value != 0
            )
            bound = isinstance(denominator, ast.Name) and denominator.id in known
            unary = isinstance(denominator, ast.UnaryOp) and literal_code(denominator) not in (
                None,
                0,
            )
            # A rounded integer norm is at least one code when either encoded
            # coordinate is nonzero. A real constant rounded to zero proves nothing.
            norm = (
                isinstance(denominator, ast.Call)
                and isinstance(denominator.func, ast.Name)
                and denominator.func.id == "hypot"
                and len(denominator.args) == 2
                and not denominator.keywords
                and any(
                    literal_code(coordinate) not in (None, 0)
                    or isinstance(coordinate, ast.Name)
                    and coordinate.id in known
                    for coordinate in denominator.args
                )
            )
            if not literal and not bound and not unary and not norm:
                raise ValueError(
                    f"{where}: divisor {ast.unparse(denominator)!r} may encode zero; "
                    "provide an encoded nonzero constant or preserve an explicit nonzero branch"
                )
        for descendant in ast.iter_child_nodes(node):
            visit(descendant, known, where)

    for where, expression in equation_expressions(neuron):
        visit(ast.parse(expression, mode="eval").body, nonzero, where)
