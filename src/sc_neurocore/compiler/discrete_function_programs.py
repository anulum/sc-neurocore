# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rounding and Heaviside integer programs
"""Construct exact discontinuous-function programs for C, Rust and hardware."""

from __future__ import annotations

from .function_unit_ir import Program


def build_round(p: Program, code: str, data_width: int, fraction: int) -> str:
    """Append integer rounding with ties to even to an existing program.

    Parameters
    ----------
    p : Program
        Destination signed-64-bit instruction program.
    code : str
        Input instruction identifier holding the signed fixed-point code.
    data_width : int
        Signed result width admitted by the function-unit factory.
    fraction : int
        Fractional bits in the input and output encodings.

    Returns
    -------
    str
        Result instruction identifier, clamped to the declared signed word.
    """
    quotient = p.shr(code, fraction)
    doubled = p.shl(p.mask(code, fraction), 1)
    scale = p.const(1 << fraction)
    increment = p.select(
        p.lt(scale, doubled),
        p.const(1),
        p.select(p.eq(doubled, scale), p.mask(quotient, 1), p.const(0)),
    )
    rounded = p.shl(p.add(quotient, increment), fraction)
    high = p.const((1 << (data_width - 1)) - 1)
    low = p.const(-(1 << (data_width - 1)))
    return p.select(p.lt(rounded, low), low, p.select(p.lt(high, rounded), high, rounded))


def build_heaviside(p: Program, code: str, at_zero: str, data_width: int, fraction: int) -> str:
    """Append a Heaviside step with a dynamic value at zero.

    Parameters
    ----------
    p : Program
        Destination signed-64-bit instruction program.
    code : str
        First operand identifier, holding a signed fixed-point word.
    at_zero : str
        Second operand identifier, returned unchanged when the first is zero.
    data_width : int
        Signed result width admitted by the function-unit factory.
    fraction : int
        Fractional bits in both operands and the result.

    Returns
    -------
    str
        Result identifier: zero, the supplied second operand, or saturated one.
    """
    zero = p.const(0)
    one = p.const(min(1 << fraction, (1 << (data_width - 1)) - 1))
    return p.select(p.lt(code, zero), zero, p.select(p.lt(zero, code), one, at_zero))
