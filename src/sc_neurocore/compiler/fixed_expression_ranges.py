# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Conservative argument ranges in fixed-point expression arithmetic
"""Bound function operands after the datapath's encoding, rounding and wrapping.

These intervals describe possible signed codes, including holes and combinations
which a model may never reach. State and input reads start at the complete word
range, independently of the initial state. The analysis does not prove trajectory
invariants or operand independence. A wrap across the signed discontinuity, an
opaque expression, or a possibly zero divisor widens the result conservatively.
Bounds on a division apply to defined evaluations only; they do not certify that
the model avoids a zero divisor.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction

from . import expr_lut_tables, function_units
from .lut_accuracy import GEOMETRY, lut_read_int
from .power_units import power_code

Bounds = tuple[int, int]


@dataclass(frozen=True, slots=True)
class CodeRange:
    """Inclusive signed-code bounds and reasons for conservative widening.

    Parameters
    ----------
    minimum, maximum : int
        Smallest and largest possible codes, including any unproved interior.
    notes : tuple of str, optional
        Opaque expressions or potentially undefined arithmetic encountered.
    """

    minimum: int
    maximum: int
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Refuse noninteger or reversed code intervals."""
        if (
            type(self.minimum) is not int
            or type(self.maximum) is not int
            or self.minimum > self.maximum
        ):
            raise ValueError("code ranges need ordered integer endpoints")

    def to_public_dict(self, fraction: int) -> dict[str, object]:
        """Return inclusive code and real bounds at the supplied binary fraction."""
        scale = 1 << fraction
        return {
            "codes": [self.minimum, self.maximum],
            "values": [self.minimum / scale, self.maximum / scale],
            "notes": list(self.notes),
        }


def expression_code_range(
    expression: str,
    data_width: int,
    fraction: int,
    *,
    values: Mapping[str, float] | None = None,
    variables: Iterable[str] = (),
    rounding: str = "truncate",
    function_lowering: str = "v2",
) -> CodeRange:
    """Infer a function operand's conservative range after signed word narrowing.

    Parameters
    ----------
    expression : str
        Equation expression, evaluated with the fixed C/Rust/Verilog rules.
    data_width, fraction : int
        Signed word width and binary fraction bits.
    values : mapping of str to float, optional
        Fixed parameters and constants, encoded before expression arithmetic.
    variables : iterable of str, optional
        State names. They take precedence over fixed values and named constants.
        Other unknown names are also unconstrained signed-word inputs.
    rounding : str, optional
        Product policy: ``truncate``, ``nearest`` or ``bankers``.
    function_lowering : str, optional
        ``v2`` or ``lut-v1``; constant function calls follow the selected units.

    Returns
    -------
    CodeRange
        Inclusive interval containing every defined signed-word result.

    Raises
    ------
    ValueError
        For an invalid format, policy, or literal division/modulo operand.
    SyntaxError
        If the expression cannot be parsed.
    """
    if not 2 <= data_width <= 64 or not 0 <= fraction < data_width:
        raise ValueError("argument ranges need 2 <= data_width <= 64 and 0 <= fraction < width")
    if rounding not in {"truncate", "nearest", "bankers"}:
        raise ValueError(f"unknown product rounding {rounding!r}")
    if function_lowering not in {"lut-v1", "v2"}:
        raise ValueError(f"unknown function lowering {function_lowering!r}")
    if function_lowering == "v2":
        function_units.check_format(data_width, fraction)
    analysis = _Ranges(
        data_width, fraction, values or {}, frozenset(variables), rounding, function_lowering
    )
    bounds = analysis.infer(ast.parse(expression, mode="eval").body)
    result = analysis.word if bounds is None else _wrap(bounds, data_width)
    return CodeRange(*result, tuple(sorted(analysis.notes)))


def _wrap(bounds: Bounds, width: int) -> Bounds:
    """Keep the exact shifted interval unless it crosses a signed discontinuity."""
    low, high = bounds
    half, period = 1 << (width - 1), 1 << width
    first, last = (low + half) // period, (high + half) // period
    return (low - first * period, high - first * period) if first == last else (-half, half - 1)


class _Ranges:
    """Carry wide sums until a real datapath narrowing point."""

    def __init__(
        self,
        width: int,
        fraction: int,
        values: Mapping[str, float],
        variables: frozenset[str],
        rounding: str,
        lowering: str,
    ) -> None:
        self.width, self.fraction = width, fraction
        self.scale = 1 << fraction
        self.word = (-(1 << (width - 1)), (1 << (width - 1)) - 1)
        self.values, self.variables = values, variables
        self.rounding, self.lowering = rounding, lowering
        self.notes: set[str] = set()

    def encode(self, value: int | float) -> Bounds:
        code = round(Fraction(float(value)) * self.scale)
        return _wrap((code, code), self.width)

    def multiply(self, left: Bounds, right: Bounds) -> Bounds:
        products = [a * b for a in left for b in right]
        low, high = _wrap((min(products), max(products)), 2 * self.width)
        if self.fraction and self.rounding != "truncate":
            half = 1 << (self.fraction - 1)
            if self.rounding == "nearest":
                low += half - int(low < 0)
                high += half - int(high < 0)
            else:
                # The tie clear is monotone before the wide addition wraps.
                low += half
                high += half
            rounded = _wrap((low, high), 2 * self.width)
            if self.rounding == "bankers":
                lo, hi = rounded
                low = (lo >> self.fraction) - int(
                    lo % self.scale == 0 and (lo >> self.fraction) & 1
                )
                high = (hi >> self.fraction) - int(
                    hi % self.scale == 0 and (hi >> self.fraction) & 1
                )
                return _wrap((low, high), self.width)
            low, high = rounded
        return _wrap((low >> self.fraction, high >> self.fraction), self.width)

    def infer(self, node: ast.AST) -> Bounds | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return self.encode(node.value)
        if isinstance(node, ast.Name):
            if node.id in self.variables:
                return self.word
            if node.id in self.values:
                return self.encode(self.values[node.id])
            if node.id in expr_lut_tables.NAMED_CONSTANTS:
                return self.encode(expr_lut_tables.NAMED_CONSTANTS[node.id])
            return self.word
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self.infer(node.operand)
            return (
                None
                if value is None
                else (-value[1], -value[0])
                if isinstance(node.op, ast.USub)
                else value
            )
        if isinstance(node, (ast.Compare, ast.BoolOp)):
            return (0, self.scale)
        if isinstance(node, ast.IfExp):
            left, right = self.infer(node.body), self.infer(node.orelse)
            return (
                None
                if left is None or right is None
                else (min(left[0], right[0]), max(left[1], right[1]))
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return self.call(node)
        if isinstance(node, ast.BinOp):
            return self.binary(node)
        self.notes.add(f"opaque {type(node).__name__}; no narrower bound")
        return None

    def binary(self, node: ast.BinOp) -> Bounds | None:
        left = self.infer(node.left)
        if left is None:
            return None
        if isinstance(node.op, ast.Pow):
            kind, exponent = expr_lut_tables.classify_power(
                node.right, self.lowering, set(self.values) | self.variables
            )
            if kind == "multiply":
                assert exponent is not None
                result = left
                for _ in range(int(exponent) - 1):
                    result = self.multiply(result, left)
                return result
            if kind in {"sqrt", "cbrt"}:
                return self.unit(kind, [left])
            base = _wrap(left, self.width)
            dynamic = self.infer(node.right) if exponent is None else None
            if base[0] == base[1] and (
                exponent is not None or dynamic is not None and dynamic[0] == dynamic[1]
            ):
                code = power_code(
                    base[0],
                    self.width,
                    self.fraction,
                    exponent=exponent,
                    exponent_code=None if dynamic is None else _wrap(dynamic, self.width)[0],
                )
                return (code, code)
            return self.word
        if (
            isinstance(node.op, ast.Div)
            and isinstance(node.right, ast.Constant)
            and isinstance(node.right.value, (int, float))
        ):
            if node.right.value == 0:
                raise ValueError("an expression divides by the literal 0")
            return self.multiply(left, self.encode(1.0 / node.right.value))
        if isinstance(node.op, (ast.Mod, ast.FloorDiv)):
            return self.period(node, left)
        right = self.infer(node.right)
        if right is None:
            return None
        if isinstance(node.op, ast.Add):
            return (left[0] + right[0], left[1] + right[1])
        if isinstance(node.op, ast.Sub):
            return (left[0] - right[1], left[1] - right[0])
        if isinstance(node.op, ast.Mult):
            return self.multiply(left, right)
        if isinstance(node.op, ast.Div):
            if right[0] <= 0 <= right[1]:
                self.notes.add("divisor may be zero; bound applies only to defined evaluations")
                return None
            shifted = _wrap((left[0] * self.scale, left[1] * self.scale), 2 * self.width)
            quotients = [
                (-1 if (a < 0) != (b < 0) else 1) * (abs(a) // abs(b))
                for a in shifted
                for b in right
            ]
            return _wrap((min(quotients), max(quotients)), self.width)
        self.notes.add(f"opaque {type(node.op).__name__}; no narrower bound")
        return None

    def period(self, node: ast.BinOp, left: Bounds) -> Bounds:
        if (
            not isinstance(node.right, ast.Constant)
            or not isinstance(node.right.value, (int, float))
            or isinstance(node.right.value, bool)
            or node.right.value <= 0
        ):
            raise ValueError("modulo and floor division need an admitted positive literal")
        value = node.right.value
        if value > self.word[1] / self.scale:
            raise ValueError("period exceeds the signed fixed-point maximum")
        low, high = _wrap(left, self.width)
        if isinstance(node.op, ast.FloorDiv):
            if type(value) is not int or value & (value - 1):
                raise ValueError("floor divisor must be an integer power of two")
            return _wrap(
                (
                    (low // (value * self.scale)) * self.scale,
                    (high // (value * self.scale)) * self.scale,
                ),
                self.width,
            )
        code = round(value * self.scale)
        if code <= 0:
            raise ValueError("modulo period underflows to zero")
        return (low % code, high % code) if low // code == high // code else (0, code - 1)

    def call(self, node: ast.Call) -> Bounds | None:
        assert isinstance(node.func, ast.Name)
        name = node.func.id
        arguments = [self.infer(arg) for arg in node.args]
        if not arguments or any(arg is None for arg in arguments):
            self.notes.add(f"opaque {name} result; no narrower bound")
            return None
        args = [arg for arg in arguments if arg is not None]
        if name == "abs":
            low, high = args[0]
            return (0 if low <= 0 <= high else min(abs(low), abs(high)), max(abs(low), abs(high)))
        if name in {"min", "max"}:
            operation = min if name == "min" else max
            return (operation(arg[0] for arg in args), operation(arg[1] for arg in args))
        if name == "clip" and len(args) == 3:
            value, lower, upper = args
            return (min(max(value[0], lower[0]), upper[0]), min(max(value[1], lower[1]), upper[1]))
        unit = expr_lut_tables.V2_UNIT_FOR.get(name)
        if unit is not None:
            return self.unit(unit, args)
        self.notes.add(f"opaque {name} result; no narrower bound")
        return None

    def unit(self, name: str, args: list[Bounds]) -> Bounds:
        words = [_wrap(arg, self.width) for arg in args]
        if all(low == high for low, high in words):
            if self.lowering == "v2" or name in expr_lut_tables.EXACT_FUNCTIONS:
                result = function_units.evaluate(
                    name,
                    words[0][0],
                    self.width,
                    self.fraction,
                    additional_inputs=tuple(arg[0] for arg in words[1:]),
                )
                return (result, result)
            if name in GEOMETRY:
                result = lut_read_int(name, words[0][0], self.width, self.fraction)
                return (result, result)
            self.notes.add(f"{name} has no legacy table; no narrower bound")
        return self.word
