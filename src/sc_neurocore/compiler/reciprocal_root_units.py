# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exactly rounded reciprocal square root reference
"""Define the fixed-point reciprocal root independently of its bounded datapath."""

from __future__ import annotations

from math import isqrt


def rsqrt_code(code: int, data_width: int, fraction: int) -> int:
    """Return the saturated nearest code of a positive reciprocal square root.

    Parameters
    ----------
    code : int
        Signed input code. Nonpositive codes produce the explicit zero sentinel,
        outside the mathematical domain; they never approximate a real result.
    data_width, fraction : int
        Common input and output format, validated by the public unit dispatcher.

    Returns
    -------
    int
        Nearest code of ``sqrt(2**(3*fraction) / code)``, with ties rounded up
        and saturation at the signed-word maximum. Arbitrary-precision arithmetic
        makes this reference independent of the emitted two-limb comparisons.
    """
    if code <= 0:
        return 0
    numerator = 1 << (3 * fraction)
    root = isqrt(numerator // code)
    rounded = root + int(4 * numerator >= code * (2 * root + 1) ** 2)
    return min(rounded, (1 << (data_width - 1)) - 1)
