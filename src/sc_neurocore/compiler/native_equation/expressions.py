# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation expressions as native scalar graphs

"""Lower the maintained equation AST with explicit context sizing and collapse points."""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping

from .. import expr_lut_tables as tables
from ..verilog_expr_emitter import _require_wide_headroom
from .arithmetic import EquationArithmetic
from .functions import lower_call, lower_lookup
from .scheduling import EquationSchedule


class EquationExpressions:
    """Translate equation expressions into typed native word instructions.

    Parameters
    ----------
    arithmetic : EquationArithmetic
        Destination graph and the existing fixed-point configuration.
    bindings : mapping of str to int
        State, parameter, constant and input SSA identifiers, all signed Q words.
        Runtime state and previous-state aliases are supplied by the integrator.
    externs : mapping of str to int, optional
        Signed i64 RNG step, seed and stream bindings.
    schedule : EquationSchedule or None, optional
        Shared source-compatible unit numbering and selected product registers.
        None leaves every expression product combinational.

    Notes
    -----
    Additive expressions inherit their enclosing width. Products, quotients,
    powers and table/program results collapse at the same points as the existing
    Verilog compiler. Inline clamps compare wide values and retain the caller's
    branch width. The surrounding frontend owns state commits and events.
    """

    def __init__(
        self,
        arithmetic: EquationArithmetic,
        bindings: Mapping[str, int],
        externs: Mapping[str, int] | None = None,
        schedule: EquationSchedule | None = None,
    ) -> None:
        self.a = arithmetic
        self.bindings = bindings
        self.externs = {} if externs is None else externs
        self.schedule = schedule
        self._values: dict[tuple[ast.AST, int], int] = {}
        self._collapsed: dict[ast.AST, int] = {}

    def lower(self, expression: str, width: int | None = None) -> int:
        """Compile a numeric expression at a declared signed context width.

        Parameters
        ----------
        expression : str
            Existing equation expression, after intrinsic/site preparation.
        width : int or None, optional
            Result width; defaults to the configured Q data width.

        Returns
        -------
        int
            Signed native SSA value at the requested width.

        Raises
        ------
        ValueError
            If the expression, name, numeric headroom or function is unsupported.
        SyntaxError
            If the expression cannot be parsed.
        """
        tree = ast.parse(expression, mode="eval")
        _require_wide_headroom(tree.body, self.a.q)
        return self.value(tree.body, self.a.q.data_width if width is None else width)

    def condition(self, node: ast.AST) -> int:
        """Lower an equation predicate to Bool, preserving wide chained comparisons.

        Parameters
        ----------
        node : ast.AST
            Comparison, boolean expression or numeric condition.

        Returns
        -------
        int
            Native Bool SSA identifier.

        Raises
        ------
        ValueError
            If a boolean operand is not itself a comparison or boolean expression.
        """
        if self.schedule is not None:
            self.schedule.plan(node)
        a, b = self.a, self.a.builder
        wide = 2 * a.q.data_width
        if isinstance(node, ast.Compare):
            left = self.value(node.left, wide)
            conditions = []
            predicates = {ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge"}
            for operation, other in zip(node.ops, node.comparators, strict=True):
                if type(operation) not in predicates:
                    raise ValueError("native equation comparisons require <, <=, > or >=")
                right = self.value(other, wide)
                conditions.append(b.word_compare(predicates[type(operation)], left, right))
                left = right
            result = conditions[0]
            for condition in conditions[1:]:
                result = b.word_select(result, condition, result, "bool")
            return result
        if isinstance(node, ast.BoolOp):
            if any(not isinstance(child, (ast.Compare, ast.BoolOp)) for child in node.values):
                raise ValueError("'and' / 'or' operands must be comparisons in native equations")
            result = self.condition(node.values[0])
            for child in node.values[1:]:
                other_condition = self.condition(child)
                result = (
                    b.word_select(result, other_condition, result, "bool")
                    if isinstance(node.op, ast.And)
                    else b.word_select(result, result, other_condition, "bool")
                )
            return result
        return b.word_compare("ne", self.value(node, wide), a.constant(0, wide))

    def value(self, node: ast.AST, width: int) -> int:
        """Compile a numeric AST node under its exact enclosing word width.

        Parameters
        ----------
        node : ast.AST
            Numeric equation subtree, including comparisons used as Q numbers.
        width : int
            Enclosing signed word width.

        Returns
        -------
        int
            Native signed-word SSA identifier of the requested width.

        Raises
        ------
        ValueError
            If a name, operator, function or numeric boolean cannot be represented.
        """
        key = (node, width)
        if self.schedule is not None:
            self.schedule.plan(node)
        if key not in self._values:
            self._values[key] = self._value(node, width)
        return self._values[key]

    def _value(self, node: ast.AST, width: int) -> int:
        """Append one subtree; memoisation shares nodes across inline comparison branches."""
        a, q, b, ty = self.a, self.a.q, self.a.builder, f"i{width}"
        dw, wide = q.data_width, 2 * q.data_width
        if isinstance(node, (ast.Compare, ast.BoolOp)):
            if q.fraction >= dw - 1:
                raise ValueError("a numeric comparison requires a Q format that represents 1.0")
            return b.word_select(
                self.condition(node), a.constant(1 << q.fraction, width), a.constant(0, width), ty
            )
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return a.literal(float(node.value), width)
        if isinstance(node, ast.Name):
            if node.id in self.bindings:
                return b.word_resize(self.bindings[node.id], ty)
            if node.id in tables.NAMED_CONSTANTS:
                return a.literal(tables.NAMED_CONSTANTS[node.id], width)
            raise ValueError(f"unbound native equation identifier: {node.id!r}")
        if isinstance(node, ast.UnaryOp):
            operand = self.value(node.operand, width)
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.USub):
                return b.word_unary("neg", operand, ty)
            raise ValueError("native equations support only unary + and -")
        if isinstance(node, ast.IfExp):
            return b.word_select(
                self.condition(node.test),
                self.value(node.body, width),
                self.value(node.orelse, width),
                ty,
            )
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"abs", "clip", "max", "min"}:
                return self._inline_call(node, width)
            if node not in self._collapsed:
                self._collapsed[node] = lower_call(node, a, self.value, self.externs)
            return b.word_resize(self._collapsed[node], ty)
        if not isinstance(node, ast.BinOp):
            raise ValueError(f"unsupported native equation AST: {type(node).__name__}")
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return b.word_binary(
                "add" if isinstance(node.op, ast.Add) else "sub",
                self.value(node.left, width),
                self.value(node.right, width),
                ty,
            )
        if isinstance(node.op, (ast.Mult, ast.Div)):
            if node in self._collapsed:
                return b.word_resize(self._collapsed[node], ty)
            left = self.value(node.left, wide)
            if (
                isinstance(node.op, ast.Div)
                and isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, (int, float))
            ):
                right = a.literal(1.0 / float(node.right.value), wide)
            else:
                right = self.value(node.right, wide)
            if isinstance(node.op, ast.Div) and not isinstance(node.right, ast.Constant):
                shifted = b.word_binary("shl", left, a.constant(q.fraction, wide), f"i{wide}")
                result = b.word_resize(b.word_binary("div", shifted, right, f"i{wide}"), f"i{dw}")
            else:
                product = b.word_binary("mul", left, right, f"i{wide}")
                if isinstance(node.op, ast.Mult) and self.schedule is not None:
                    product = self.schedule.product(node, product, wide)
                result = a.round_product(product, wide)
            self._collapsed[node] = result
            return b.word_resize(result, ty)
        if isinstance(node.op, ast.Pow):
            if node not in self._collapsed:
                self._collapsed[node] = self._power(node)
            return b.word_resize(self._collapsed[node], ty)
        if isinstance(node.op, ast.FloorDiv):
            divisor = node.right
            if (
                not isinstance(divisor, ast.Constant)
                or type(divisor.value) is not int
                or divisor.value <= 0
                or divisor.value & (divisor.value - 1)
                or divisor.value > q.max_value
            ):
                raise ValueError(
                    "floor divisor must be a representable positive integer power-of-two literal"
                )
            left = self.value(node.left, dw)
            result = b.word_binary(
                "ashr", left, a.constant(q.fraction + divisor.value.bit_length() - 1, dw), f"i{dw}"
            )
            result = b.word_binary("shl", result, a.constant(q.fraction, dw), f"i{dw}")
            return b.word_resize(result, ty)
        if isinstance(node.op, ast.Mod):
            divisor = node.right
            if (
                not isinstance(divisor, ast.Constant)
                or isinstance(divisor.value, bool)
                or not isinstance(divisor.value, (int, float))
                or not math.isfinite(divisor.value)
                or not 0 < divisor.value <= q.max_value
            ):
                raise ValueError("modulo divisor must be a representable finite positive literal")
            period = round(float(divisor.value) * (1 << q.fraction))
            if period <= 0:
                raise ValueError("modulo divisor underflows the configured Q format")
            remainder = b.word_binary(
                "rem", self.value(node.left, dw), a.constant(period, dw), f"i{dw}"
            )
            negative = b.word_compare("lt", remainder, a.constant(0, dw))
            positive = b.word_binary("add", remainder, a.constant(period, dw), f"i{dw}")
            return b.word_resize(b.word_select(negative, positive, remainder, f"i{dw}"), ty)
        raise ValueError(f"unsupported native equation operator: {type(node.op).__name__}")

    def _inline_call(self, node: ast.Call, width: int) -> int:
        """Retain context-sized branches for non-collapsing absolute/clamp/min/max calls."""
        assert isinstance(node.func, ast.Name)
        name, a, b = node.func.id, self.a, self.a.builder
        tables.check_call_arguments(name, node)
        wide, ty = 2 * a.q.data_width, f"i{width}"
        if name == "abs":
            raw = self.value(node.args[0], width)
            condition = b.word_compare("lt", self.value(node.args[0], wide), a.constant(0, wide))
            return b.word_select(condition, b.word_unary("neg", raw, ty), raw, ty)
        first = node.args[0]
        others = node.args[1:]
        if name == "clip":
            first = ast.Call(func=ast.Name(id="max"), args=[first, others[0]], keywords=[])
            others = others[1:]
        for other in others:
            comparison = ast.Gt() if name == "max" else ast.Lt()
            test = ast.Compare(left=first, ops=[comparison], comparators=[other])
            first = ast.IfExp(test=test, body=first, orelse=other)
        return self.value(first, width)

    def _power(self, node: ast.BinOp) -> int:
        """Collapse every power multiplication and reuse the existing fractional-power units."""
        a, b, q = self.a, self.a.builder, self.a.q
        exponent = tables.const_float(node.right)
        if (
            isinstance(node.right, ast.Constant)
            and isinstance(node.right.value, (int, float))
            and (
                node.right.value in (2, 3)
                or type(node.right.value) is int
                and 4 <= node.right.value <= 8
            )
        ):
            wide, result = 2 * q.data_width, self.value(node.left, 2 * q.data_width)
            base = result
            for _ in range(int(node.right.value) - 1):
                product = b.word_binary("mul", result, base, f"i{wide}")
                result = b.word_resize(a.round_product(product, wide), f"i{wide}")
            return b.word_resize(result, f"i{q.data_width}")
        if exponent is not None and (
            abs(exponent - 0.5) < 1e-6 or abs(exponent - 1.0 / 3.0) < 1e-6
        ):
            name = "sqrt" if abs(exponent - 0.5) < 1e-6 else "cbrt"
            argument = self.value(node.left, q.data_width)
            if q.function_lowering == "lut-v1":
                return lower_lookup(name, argument, a)
            call = ast.Call(func=ast.Name(id=name), args=[node.left], keywords=[])
            if name == "sqrt":
                return lower_call(call, a, self.value, self.externs)
            from ..function_unit_programs import program
            from ..native_program import lower_program

            return b.word_resize(
                lower_program(
                    program("cbrt", q.data_width, q.fraction), b, [b.word_resize(argument, "i64")]
                ),
                f"i{q.data_width}",
            )
        raise ValueError("native equation powers require integer 2-8 or fractional 1/2, 1/3")
