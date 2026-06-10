# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Formal overflow proof

"""Formal overflow proof utilities.

Uses interval arithmetic on the ODE AST to statically prove that
no overflow occurs at a given precision.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class Interval:
    """A closed interval [lo, hi] for interval arithmetic."""

    lo: float
    hi: float

    def __add__(self, other: Interval) -> Interval:
        """Add two intervals: [a,b] + [c,d] = [a+c, b+d]."""
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def __sub__(self, other: Interval) -> Interval:
        """Subtract intervals: [a,b] - [c,d] = [a-d, b-c]."""
        return Interval(self.lo - other.hi, self.hi - other.lo)

    def __mul__(self, other: Interval) -> Interval:
        """Multiply intervals: all four products, take min/max."""
        products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ]
        return Interval(min(products), max(products))

    def __truediv__(self, other: Interval) -> Interval:
        """Divide intervals. Raises if divisor contains zero."""
        if other.lo <= 0 <= other.hi:
            return Interval(float("-inf"), float("inf"))
        quotients = [
            self.lo / other.lo,
            self.lo / other.hi,
            self.hi / other.lo,
            self.hi / other.hi,
        ]
        return Interval(min(quotients), max(quotients))

    def __neg__(self) -> Interval:
        """Negate: -[a,b] = [-b, -a]."""
        return Interval(-self.hi, -self.lo)

    def contains(self, lo: float, hi: float) -> bool:
        """Check if this interval is contained within [lo, hi]."""
        return self.lo >= lo and self.hi <= hi


class _IntervalEvaluator(ast.NodeVisitor):
    """Evaluate an AST expression using interval arithmetic."""

    def __init__(self, bounds: dict[str, tuple[float, float]]):
        """Initialise with variable bounds.

        Parameters
        ----------
        bounds : dict
            Mapping from variable name to (min, max) bounds.
        """
        self.bounds = bounds

    def visit_BinOp(self, node: ast.BinOp) -> Interval:
        """Evaluate a binary operation on intervals."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            return left / right
        raise ValueError(f"Unsupported op: {type(node.op).__name__}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Interval:
        """Evaluate a unary operation on an interval."""
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        raise ValueError(f"Unsupported unary: {type(node.op).__name__}")

    def visit_Name(self, node: ast.Name) -> Interval:
        """Look up the interval for a variable name."""
        if node.id in self.bounds:
            lo, hi = self.bounds[node.id]
            return Interval(lo, hi)
        raise KeyError(f"No bounds for variable '{node.id}'")

    def visit_Constant(self, node: ast.Constant) -> Interval:
        """A constant is a point interval [c, c]."""
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsupported constant: {node.value!r}")
        v = float(node.value)
        return Interval(v, v)

    def generic_visit(self, node: ast.AST) -> Interval:
        """Raise for unsupported nodes."""
        raise ValueError(f"Unsupported node: {type(node).__name__}")


@dataclass
class OverflowProofResult:
    """Result of a formal overflow proof.

    Attributes
    ----------
    proven_safe : bool
        True if the expression provably cannot overflow at the given precision.
    expr_interval : Interval
        The computed output interval of the expression.
    q_min : float
        Minimum representable value in the Q-format.
    q_max : float
        Maximum representable value in the Q-format.
    margin_lo : float
        How far the expression minimum is from the Q-format minimum (positive = safe).
    margin_hi : float
        How far the expression maximum is from the Q-format maximum (positive = safe).
    """

    proven_safe: bool
    expr_interval: Interval
    q_min: float
    q_max: float
    margin_lo: float
    margin_hi: float


@dataclass(frozen=True)
class FixedPointEnvelopeProof:
    """Static fixed-point width proof over conservative Q-code bounds.

    The proof is intended for deployment artefacts that already compute
    conservative absolute output bounds, such as mixed Q8.8/Q16.16 and
    block-floating dense paths.  It does not use realised output cancellation:
    a small output value remains unsafe when the absolute product envelope
    exceeds the signed Q-format capacity.
    """

    proof_kind: str
    output_format: str
    signed: bool
    total_bits: int
    fractional_bits: int
    conservative_safe_bound_code: int
    max_abs_bound_code: int
    min_headroom_code: int
    required_total_bits: int
    required_integer_bits: int
    width_headroom_bits: int
    saturation_required: bool
    static_overflow_proven_safe: bool

    def manifest(self) -> dict[str, int | bool | str]:
        """Return a stable JSON-serialisable proof manifest."""
        return {
            "proof_kind": self.proof_kind,
            "output_format": self.output_format,
            "signed": self.signed,
            "total_bits": self.total_bits,
            "fractional_bits": self.fractional_bits,
            "conservative_safe_bound_code": self.conservative_safe_bound_code,
            "max_abs_bound_code": self.max_abs_bound_code,
            "min_headroom_code": self.min_headroom_code,
            "required_total_bits": self.required_total_bits,
            "required_integer_bits": self.required_integer_bits,
            "width_headroom_bits": self.width_headroom_bits,
            "saturation_required": self.saturation_required,
            "static_overflow_proven_safe": self.static_overflow_proven_safe,
        }


def _required_total_bits_for_bound(abs_bound_code: int, *, signed: bool) -> int:
    """Return the minimum total bit width needed by a non-negative code bound."""
    if abs_bound_code <= 0:
        return 1
    if signed:
        return abs_bound_code.bit_length() + 1
    return abs_bound_code.bit_length()


def prove_fixed_point_envelope(
    bound_codes: Sequence[int],
    *,
    total_bits: int = 32,
    fractional_bits: int = 16,
    signed: bool = True,
) -> FixedPointEnvelopeProof:
    """Prove whether conservative Q-code bounds fit a fixed-point format.

    Parameters
    ----------
    bound_codes : Sequence[int]
        Conservative absolute output bounds in integer Q-code units.  For
        signed formats the function accepts signed codes and proves against
        their absolute magnitudes.  Unsigned formats reject negative codes.
    total_bits : int
        Total output width, including the sign bit for signed formats.
    fractional_bits : int
        Fractional precision bits in the Q-format.
    signed : bool
        Whether the target fixed-point format is signed.

    Returns
    -------
    FixedPointEnvelopeProof
        Fail-closed width proof and saturation requirement.
    """
    if total_bits <= 0:
        raise ValueError("total_bits must be positive")
    if fractional_bits < 0:
        raise ValueError("fractional_bits must be non-negative")
    if fractional_bits >= total_bits:
        raise ValueError("fractional_bits must be smaller than total_bits")
    if not bound_codes:
        raise ValueError("bound_codes must contain at least one conservative bound")

    normalised_bounds: list[int] = []
    for code in bound_codes:
        if isinstance(code, bool) or not isinstance(code, int):
            raise TypeError("bound_codes must be integer Q-code values")
        if not signed and code < 0:
            raise ValueError("unsigned fixed-point envelope cannot contain negative codes")
        normalised_bounds.append(abs(code) if signed else code)

    max_abs_bound = max(normalised_bounds)
    if signed:
        safe_bound = (1 << (total_bits - 1)) - 1
        integer_floor = 1
        proof_kind = "signed_symmetric_fixed_point_width"
    else:
        safe_bound = (1 << total_bits) - 1
        integer_floor = 0
        proof_kind = "unsigned_fixed_point_width"

    required_total = _required_total_bits_for_bound(max_abs_bound, signed=signed)
    required_integer = max(required_total - fractional_bits, integer_floor)
    width_headroom = total_bits - required_total
    saturation_required = max_abs_bound > safe_bound
    output_format = f"Q{total_bits - fractional_bits}.{fractional_bits}"

    return FixedPointEnvelopeProof(
        proof_kind=proof_kind,
        output_format=output_format,
        signed=signed,
        total_bits=total_bits,
        fractional_bits=fractional_bits,
        conservative_safe_bound_code=safe_bound,
        max_abs_bound_code=max_abs_bound,
        min_headroom_code=safe_bound - max_abs_bound,
        required_total_bits=required_total,
        required_integer_bits=required_integer,
        width_headroom_bits=width_headroom,
        saturation_required=saturation_required,
        static_overflow_proven_safe=not saturation_required,
    )


def prove_no_overflow(
    expr_str: str,
    bounds: dict[str, tuple[float, float]],
    data_width: int = 16,
    fraction: int = 8,
    signed: bool = True,
) -> OverflowProofResult:
    """Statically prove that an expression cannot overflow at a given precision.

    Uses interval arithmetic to compute the range of possible output values,
    then checks whether the output interval fits within the Q-format range.

    Parameters
    ----------
    expr_str : str
        Python-syntax arithmetic expression.
    bounds : dict
        Mapping from variable name to (min, max) bounds.
    data_width : int
        Total bit width.
    fraction : int
        Fractional bits.
    signed : bool
        Whether the format is signed.

    Returns
    -------
    OverflowProofResult
        Contains ``proven_safe=True`` if the proof succeeds.
    """
    tree = ast.parse(expr_str, mode="eval")
    evaluator = _IntervalEvaluator(bounds)
    result_interval = evaluator.visit(tree.body)

    if signed:
        q_max = ((1 << (data_width - 1)) - 1) / (1 << fraction)
        q_min = -(1 << (data_width - 1)) / (1 << fraction)
    else:
        q_max = ((1 << data_width) - 1) / (1 << fraction)
        q_min = 0.0

    margin_lo = result_interval.lo - q_min
    margin_hi = q_max - result_interval.hi

    return OverflowProofResult(
        proven_safe=(margin_lo >= 0 and margin_hi >= 0),
        expr_interval=result_interval,
        q_min=q_min,
        q_max=q_max,
        margin_lo=margin_lo,
        margin_hi=margin_hi,
    )
