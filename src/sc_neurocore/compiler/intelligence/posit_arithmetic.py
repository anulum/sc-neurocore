# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Posit arithmetic

"""Posit arithmetic utilities for compact neuron representations.

Provides encode/decode functions for Posit formats (8-bit and 16-bit).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PositConfig:
    """Posit number format configuration.

    Attributes
    ----------
    nbits : int
        Total bit width (8 or 16).
    es : int
        Exponent field size (0, 1, or 2).
    """

    nbits: int
    es: int

    @property
    def useed(self) -> int:
        """The useed value: 2^(2^es)."""
        return 1 << (1 << self.es)

    @property
    def max_value(self) -> float:
        """Maximum finite value."""
        return float(self.useed ** (self.nbits - 2))

    @property
    def min_positive(self) -> float:
        """Smallest positive value."""
        return 1.0 / self.max_value


def posit_encode(value: float, config: PositConfig) -> int:
    """Encode a float to posit integer representation.

    Parameters
    ----------
    value : float
        Value to encode.
    config : PositConfig
        Posit format.

    Returns
    -------
    int
        Posit-encoded integer (nbits wide).
    """
    nbits = config.nbits
    es = config.es

    if value == 0:
        return 0
    if math.isinf(value) or math.isnan(value):
        return 1 << (nbits - 1)  # NaR

    sign = value < 0.0
    x = abs(value)

    # Decompose x = 2**scale * (1 + frac) with frac in [0, 1).
    mantissa, exp = math.frexp(x)  # x = mantissa * 2**exp, mantissa in [0.5, 1)
    scale = exp - 1
    frac = mantissa * 2.0 - 1.0

    # Split the binary scale into a regime (base useed = 2**2**es) and exponent.
    two_es = 1 << es
    k = scale // two_es
    e = scale - k * two_es  # 0 <= e < 2**es

    # Regime bits: k >= 0 -> (k+1) ones then a 0; k < 0 -> (-k) zeros then a 1.
    if k >= 0:
        regime_len = k + 2
        regime_bits = ((1 << (k + 1)) - 1) << 1
    else:
        regime_len = -k + 1
        regime_bits = 1

    avail = nbits - 1
    frac_width = avail + 3
    frac_int = min((1 << frac_width) - 1, int(round(frac * (1 << frac_width))))

    pattern = (((regime_bits << es) | e) << frac_width) | frac_int
    pattern_len = regime_len + es + frac_width

    # Round the assembled pattern to the available payload width (nearest, ties to even).
    # frac_width = avail + 3 guarantees pattern_len > avail, so the shift is always positive.
    shift = pattern_len - avail
    kept = pattern >> shift
    dropped = pattern & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    if dropped > half or (dropped == half and kept & 1):
        kept += 1
    encoded = kept

    # Saturate between minpos (1) and maxpos ((1<<avail)-1); never the 0 or NaR codes.
    encoded = min((1 << avail) - 1, max(1, encoded))

    if sign:
        encoded = (1 << nbits) - encoded

    return encoded & ((1 << nbits) - 1)


def posit_decode(bits: int, config: PositConfig) -> float:
    """Decode a posit integer to float.

    Parameters
    ----------
    bits : int
        Posit-encoded integer.
    config : PositConfig
        Posit format.

    Returns
    -------
    float
        Decoded value.
    """
    nbits = config.nbits
    es = config.es
    mask = (1 << nbits) - 1
    bits = bits & mask

    if bits == 0:
        return 0.0
    if bits == (1 << (nbits - 1)):
        return float("inf")  # NaR

    sign = bits >> (nbits - 1)
    if sign:
        bits = (1 << nbits) - bits

    # Payload is the nbits-1 bits below the sign bit, parsed MSB-first.
    payload = bits & ((1 << (nbits - 1)) - 1)
    pos = nbits - 2

    regime_sign = (payload >> pos) & 1
    run = 0
    while pos >= 0 and ((payload >> pos) & 1) == regime_sign:
        run += 1
        pos -= 1
    pos -= 1  # consume the terminating bit
    k = (run - 1) if regime_sign == 1 else -run

    # Exponent: next es bits (bits past the payload read as 0).
    e = 0
    for _ in range(es):
        e = (e << 1) | ((payload >> pos) & 1 if pos >= 0 else 0)
        pos -= 1

    # Fraction: whatever payload bits remain.
    frac_bits = pos + 1
    frac = (payload & ((1 << frac_bits) - 1)) / (1 << frac_bits) if frac_bits > 0 else 0.0

    value = (2.0 ** (k * (1 << es) + e)) * (1.0 + frac)
    return -value if sign else value


POSIT8_0 = PositConfig(8, 0)
POSIT8_1 = PositConfig(8, 1)
POSIT16_1 = PositConfig(16, 1)
POSIT16_2 = PositConfig(16, 2)
