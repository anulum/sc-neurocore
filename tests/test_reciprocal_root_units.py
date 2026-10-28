# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reciprocal square root reference contracts
"""Check reciprocal roots against independent high-precision arithmetic."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP, localcontext

import pytest

from sc_neurocore.compiler.function_units import evaluate
from sc_neurocore.compiler.lut_accuracy import measure


def decimal_rsqrt(code: int, width: int, fraction: int) -> int:
    """Round a separate decimal reciprocal-root calculation to the admitted word."""
    if code <= 0:
        return 0
    with localcontext() as context:
        context.prec = 100
        scale = Decimal(2) ** fraction
        value = scale / (Decimal(code) / scale).sqrt()
        return min(int(value.to_integral_value(rounding=ROUND_HALF_UP)), (1 << (width - 1)) - 1)


@pytest.mark.parametrize("width,fraction", [(8, 0), (8, 4), (8, 7), (16, 0), (16, 8), (16, 15)])
def test_every_positive_word_matches_decimal(width: int, fraction: int) -> None:
    """Every quantized positive input includes all midpoint and saturation transitions."""
    for code in range(1, 1 << (width - 1)):
        assert evaluate("rsqrt", code, width, fraction) == decimal_rsqrt(code, width, fraction)


@pytest.mark.parametrize("fraction", [0, 1, 8, 16, 24, 29])
def test_wide_edges_match_decimal(fraction: int) -> None:
    """Powers of two and adjacent codes exercise limb carries and the smallest input."""
    high = (1 << 31) - 1
    codes = {1, 2, 3, 4, 5, high, high - 1}
    codes.update((1 << bit) + delta for bit in range(31) for delta in (-1, 0, 1))
    for code in sorted(c for c in codes if 0 < c <= high):
        assert evaluate("rsqrt", code, 32, fraction) == decimal_rsqrt(code, 32, fraction)


@pytest.mark.parametrize("code", [-128, -1, 0])
def test_nonpositive_codes_have_explicit_zero_sentinel(code: int) -> None:
    """Outside-domain datapaths return zero without claiming a real reciprocal root."""
    assert evaluate("rsqrt", code, 8, 4) == 0


@pytest.mark.parametrize("code", [-129, 128])
def test_outside_word_inputs_refuse(code: int) -> None:
    """The dispatcher refuses inputs that no signed input port can carry."""
    with pytest.raises(ValueError, match="outside signed"):
        evaluate("rsqrt", code, 8, 4)


@pytest.mark.parametrize("width,fraction", [(8, 0), (16, 8), (32, 16), (32, 29), (31, 30)])
def test_accuracy_reports_positive_domain_and_half_lsb_bound(width: int, fraction: int) -> None:
    """Measured evidence excludes nonpositive sentinels and states sampled versus exhaustive scope."""
    report = measure("rsqrt", width, fraction, lowering="v2")
    scale = 1 << fraction
    assert report.declared_error_bound == 0.5 / scale
    assert report.max_error <= report.declared_error_bound
    assert report.operand_ranges == ((1 / scale, ((1 << (width - 1)) - 1) / scale),)
    assert len(report.worst_inputs) == 1
    assert report.exhaustive is (width <= 16)
    if report.exhaustive:
        assert report.codes_evaluated == (1 << (width - 1)) - 1
