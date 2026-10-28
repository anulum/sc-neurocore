# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Power reference domains and independent real-function accuracy
"""Compare the integer power reference with independent high-precision Decimal powers."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from sc_neurocore.compiler.power_units import exponent_format, power_code


@pytest.mark.parametrize("width,fraction", [(8, 4), (16, 8), (32, 0), (32, 16), (32, 29), (31, 30)])
@pytest.mark.parametrize("exponent", [-64.0, -9.0, -1.0, -0.5, 0.0, 0.125, 1.5, 9.0, 64.0])
def test_reference_matches_real_powers_on_representable_inputs(
    width: int, fraction: int, exponent: float
) -> None:
    """The ordinary literal range remains within 0.501 output LSB of real powers."""
    scale, high = 1 << fraction, (1 << (width - 1)) - 1
    bases = {1, 2, 3, max(1, scale // 2), max(1, scale - 1), scale, scale + 1, high}
    with localcontext() as context:
        context.prec = 96
        for base in sorted(code for code in bases if code <= high):
            real = context.power(Decimal(base) / scale, Decimal.from_float(exponent)) * scale
            expected = min(Decimal(high), real)
            actual = power_code(base, width, fraction, exponent=exponent)
            assert abs(Decimal(actual) - expected) <= Decimal("0.501"), (
                width,
                fraction,
                exponent,
                base,
                actual,
                expected,
            )


def test_literal_precision_is_independent_of_a_narrow_base_format() -> None:
    """Q1.15 retains exponent 1.5 rather than wrapping it into the base word."""
    encoded = exponent_format(1.5)
    assert Fraction(encoded.code, 1 << encoded.fraction) == Fraction(3, 2)
    assert power_code(16384, 16, 15, exponent=1.5) == round(math.sqrt(0.125) * 32768)


def test_original_sign_and_parity_survive_literal_coding() -> None:
    """Tiny signs and huge odd integers retain zero/negative-base domain semantics."""
    assert exponent_format(math.ulp(0.0)).code == 0
    assert power_code(0, 16, 8, exponent=math.ulp(0.0)) == 0
    assert power_code(256, 16, 8, exponent=-math.ulp(0.0)) == 256
    assert power_code(0, 16, 8, exponent=-math.ulp(0.0)) == 0
    odd = (1 << 100) + 1
    assert exponent_format(odd).odd
    assert power_code(-256, 16, 8, exponent=odd) == -256
    assert power_code(-256, 16, 8, exponent=-odd) == -256
    assert power_code(-256, 16, 8, exponent=odd + 1) == 256


def test_domains_signed_saturation_and_zero_power() -> None:
    """Signed saturation precedes sign loss; invalid mathematical inputs use the sentinel."""
    assert power_code(-512, 16, 8, exponent=9) == -32768
    assert power_code(512, 16, 8, exponent=9) == 32767
    assert power_code(-512, 16, 8, exponent=-3) == -32
    assert power_code(-256, 16, 8, exponent=0.125) == 0
    assert power_code(0, 16, 8, exponent=-1) == 0
    assert power_code(0, 16, 8, exponent=0) == 256
    assert power_code(-32768, 16, 15, exponent=0) == 32767
    assert power_code(0, 16, 8, exponent_code=0) == 0
    assert power_code(-256, 16, 8, exponent_code=512) == 0
    assert power_code(256, 16, 8, exponent_code=0) == 256


@pytest.mark.parametrize("value", [True, False, float("inf"), -float("inf"), float("nan")])
def test_nonreal_or_boolean_literal_is_refused(value: float) -> None:
    """A public literal code cannot silently admit Boolean or nonfinite exponents."""
    with pytest.raises(ValueError, match="exponent"):
        exponent_format(value)


def test_invalid_operand_pairs_and_words_are_refused() -> None:
    """Operand arity and word typing are validated before arithmetic begins."""
    for base in [True, 32768, -32769]:
        with pytest.raises(ValueError, match="signed word"):
            power_code(base, 16, 8, exponent=1.5)
    with pytest.raises(ValueError, match="needs an exponent code"):
        power_code(256, 16, 8)
    with pytest.raises(ValueError, match="excludes an exponent code"):
        power_code(256, 16, 8, exponent=1.5, exponent_code=0)
    with pytest.raises(ValueError, match="signed word"):
        power_code(256, 16, 8, exponent_code=True)
    with pytest.raises(ValueError, match="format widths"):
        power_code(1, True, 0, exponent=1.5)
