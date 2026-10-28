# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Signed64 power program arithmetic and operand boundaries
"""Exercise public power programs against unbounded arithmetic at each checked step."""

from __future__ import annotations

import pytest

from sc_neurocore.compiler.power_unit_programs import power_program
from sc_neurocore.compiler.power_units import power_code


@pytest.mark.parametrize("fraction", [0, 2, 4])
@pytest.mark.parametrize("exponent", [-9.0, -3.0, -1.0, -0.5, 0.0, 0.125, 1.5, 9.0, 64.0])
def test_constant_program_exhausts_small_signed_words(fraction: int, exponent: float) -> None:
    """Every signed five-bit input agrees, including eagerly computed invalid-domain branches."""
    unit = power_program(5, fraction, exponent)
    for base in range(-16, 16):
        assert unit.evaluate(base) == power_code(base, 5, fraction, exponent=exponent)


@pytest.mark.parametrize("fraction", [0, 2, 4])
def test_dynamic_program_exhausts_both_small_operands(fraction: int) -> None:
    """Every ordered five-bit input tuple remains within signed64 and matches the reference."""
    unit = power_program(5, fraction)
    for base in range(-16, 16):
        for exponent in range(-16, 16):
            actual = unit.evaluate(base, additional_inputs=(exponent,))
            assert actual == power_code(base, 5, fraction, exponent_code=exponent)


def test_word_and_exponent_extremes_cover_each_admitted_geometry() -> None:
    """Leading-bit, limb-carry, clipping and shift boundaries span every admitted word width."""
    for width in range(2, 33):
        for fraction in sorted({0, width // 2, min(30, width - 1, 61 - width)}):
            low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
            scale = 1 << fraction
            bases = {low, -1, 0, 1, high, scale - 1, scale, scale + 1}
            unit = power_program(width, fraction)
            for base in sorted(value for value in bases if low <= value <= high):
                for exponent in (low, -1, 0, 1, high):
                    actual = unit.evaluate(base, additional_inputs=(exponent,))
                    assert actual == power_code(base, width, fraction, exponent_code=exponent)


def test_literal_classification_does_not_alias_a_cached_numeric_key() -> None:
    """Validation precedes caching and preserves original huge-integer odd parity."""
    assert power_program(16, 8, 1).evaluate(256) == 256
    with pytest.raises(ValueError, match="exponent"):
        power_program(16, 8, True)
    odd = (1 << 100) + 1
    assert power_program(16, 8, odd).evaluate(-256) == -256
    assert power_program(16, 8, odd + 1).evaluate(-256) == 256
    with pytest.raises(ValueError, match="format widths"):
        power_program(16, True)
    with pytest.raises(ValueError, match="needs exactly 2"):
        power_program(16, 8).evaluate(256)
    with pytest.raises(ValueError, match="needs exactly 1"):
        power_program(16, 8, 1.5).evaluate(256, additional_inputs=(0,))
