# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Euclidean magnitude reference contracts
"""Check public norm dispatch against decimal square roots and measured bounds."""

from __future__ import annotations

from decimal import Decimal, localcontext

import pytest

from sc_neurocore.compiler.function_units import evaluate
from sc_neurocore.compiler import function_units
from sc_neurocore.compiler.lut_accuracy import measure


def decimal_norm(first: int, second: int, width: int) -> int:
    """Round an independent high-precision square root, then cap its signed word."""
    with localcontext() as context:
        context.prec = 80
        root = Decimal(first * first + second * second).sqrt()
        return min(int(root.to_integral_value()), (1 << (width - 1)) - 1)


def test_every_eight_bit_coordinate_pair_matches_decimal_norm() -> None:
    """Every sign, zero, rounding transition and saturation agrees with decimal arithmetic."""
    for first in range(-128, 128):
        for second in range(-128, 128):
            assert evaluate("hypot", first, 8, 4, additional_inputs=(second,)) == decimal_norm(
                first, second, 8
            )


@pytest.mark.parametrize("width,fraction", [(2, 0), (16, 8), (18, 9), (32, 0), (32, 16), (31, 30)])
def test_wide_norm_boundaries_match_independent_decimal(width: int, fraction: int) -> None:
    """Most-negative pairs never overflow, and the function preserves Pythagorean triples."""
    low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
    axis = {low, low + 1, high, high - 1, -1, 0, 1, 3, 4, 5, 6, 8, 12, 13}
    axis = {code for code in axis if low <= code <= high}
    for first in axis:
        for second in axis:
            assert evaluate(
                "hypot", first, width, fraction, additional_inputs=(second,)
            ) == decimal_norm(first, second, width)


@pytest.mark.parametrize("operands", [(), (1, 2)])
def test_norm_refuses_wrong_operand_count(operands: tuple[int, ...]) -> None:
    """A binary norm never drops extra coordinates or invents a missing coordinate."""
    with pytest.raises(ValueError, match="exactly 2"):
        evaluate("hypot", 0, 8, 4, additional_inputs=operands)


@pytest.mark.parametrize("first,second", [(-129, 0), (128, 0), (0, -129), (0, 128)])
def test_norm_refuses_outside_word_coordinates(first: int, second: int) -> None:
    """Both operands must belong to the declared input word before norm evaluation."""
    with pytest.raises(ValueError, match="outside signed 8-bit domain"):
        evaluate("hypot", first, 8, 4, additional_inputs=(second,))


@pytest.mark.parametrize("width,fraction", [(8, 4), (16, 8), (32, 16), (31, 30)])
def test_accuracy_receipt_states_both_domains_and_half_lsb_bound(width: int, fraction: int) -> None:
    """The public report includes both operands and distinguishes exhaustive from sampled proof."""
    report = measure("hypot", width, fraction, lowering="v2")
    scale = 1 << fraction
    assert report.declared_error_bound == 0.5 / scale
    assert report.max_error <= report.declared_error_bound
    assert len(report.operand_ranges) == len(report.worst_inputs) == 2
    assert report.exhaustive is (width <= 8)
    if report.exhaustive:
        assert report.codes_evaluated == (1 << (2 * width))


def test_accuracy_measurement_refuses_a_kernel_that_breaks_its_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corrupted public kernel cannot publish an accuracy receipt beyond its declared bound."""

    def corrupted(*args: object, **kwargs: object) -> int:
        """Model a broken kernel returning zero for every coordinate pair."""
        return 0

    monkeypatch.setattr(function_units, "evaluate", corrupted)
    with pytest.raises(ValueError, match="exceeds its declared arithmetic error bound"):
        measure("hypot", 3, 1, lowering="v2")
