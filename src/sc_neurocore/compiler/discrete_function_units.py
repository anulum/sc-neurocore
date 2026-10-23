# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Integer rounding and Heaviside function references
"""Evaluate discontinuous functions on the compiler's signed fixed-point words."""

from __future__ import annotations


def round_code(code: int, data_width: int, fraction: int) -> int:
    """Round a fixed-point value to an integer, resolving ties to even.

    Parameters
    ----------
    code : int
        Signed input code within the declared word.
    data_width : int
        Signed word width, validated by the public function-unit dispatcher.
    fraction : int
        Fractional bits in both the input and output words.

    Returns
    -------
    int
        The nearest integer in the same fixed-point encoding, saturated when
        rounding the largest positive values would leave the signed word.
    """
    scale = 1 << fraction
    quotient, remainder = divmod(code, scale)
    doubled = 2 * remainder
    increment = doubled > scale or doubled == scale and quotient % 2 != 0
    rounded = (quotient + int(increment)) * scale
    return max(-(1 << (data_width - 1)), min((1 << (data_width - 1)) - 1, rounded))


def heaviside_code(code: int, at_zero: int, data_width: int, fraction: int) -> int:
    """Evaluate the Heaviside step with an explicitly encoded value at zero.

    Parameters
    ----------
    code : int
        Signed first operand, within the declared word.
    at_zero : int
        Signed second operand in the same format, returned when ``code == 0``.
    data_width : int
        Signed word width, validated by the public function-unit dispatcher.
    fraction : int
        Fractional bits in the operands and result.

    Returns
    -------
    int
        Zero for a negative first operand, ``at_zero`` for zero, or the encoded
        value one for a positive operand. One saturates if the word cannot
        represent it, including signed fractional-only formats.
    """
    if code < 0:
        return 0
    if code == 0:
        return at_zero
    return min(1 << fraction, (1 << (data_width - 1)) - 1)
