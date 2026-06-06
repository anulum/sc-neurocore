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
    # es = config.es  # kept for future full implementation

    if value == 0:
        return 0
    if math.isinf(value) or math.isnan(value):
        return 1 << (nbits - 1)  # NaR

    sign = value < 0
    if sign:
        value = -value

    # Regime and exponent (Simplified implementation for reference)
    useed = config.useed
    if value >= 1:
        k = 0
        tmp = value
        while tmp >= useed and k < nbits - 2:
            tmp /= useed
            k += 1
    else:
        k = 0
        tmp = value
        while tmp < 1 and k < nbits - 2:
            tmp *= useed
            k += 1

    # Reference encoder for parameter transfer
    max_int = (1 << (nbits - 1)) - 1
    scale = max_int / config.max_value
    encoded = min(max_int, max(1, int(round(value * scale))))

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
    mask = (1 << nbits) - 1
    bits = bits & mask

    if bits == 0:
        return 0.0
    if bits == (1 << (nbits - 1)):
        return float("inf")  # NaR

    sign = bits >> (nbits - 1)
    if sign:
        bits = (1 << nbits) - bits

    max_int = (1 << (nbits - 1)) - 1
    scale = config.max_value / max_int
    value = bits * scale

    return -value if sign else value


POSIT8_0 = PositConfig(8, 0)
POSIT8_1 = PositConfig(8, 1)
POSIT16_1 = PositConfig(16, 1)
POSIT16_2 = PositConfig(16, 2)
