# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation safety numeric-limit tests

"""Nested exponentiation, exponent ceiling, and literal-magnitude contracts."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.equation_safety import ExpressionSafetyValidator


def test_nested_exponentiation_is_rejected() -> None:
    """Chained exponents (``9**9**9**9``) blow up under eval and are refused."""
    with pytest.raises(ValueError, match="Nested exponentiation blocked"):
        ExpressionSafetyValidator().validate("9**9**9**9")


def test_oversized_literal_exponent_is_rejected() -> None:
    """A single ``**`` with a huge literal exponent is refused before eval."""
    with pytest.raises(ValueError, match="Exponent .* exceeds limit"):
        ExpressionSafetyValidator().validate("9 ** 999999")


def test_oversized_numeric_constant_is_rejected() -> None:
    """A literal whose magnitude overflows the sandbox ceiling is refused."""
    with pytest.raises(ValueError, match="exceeds magnitude limit"):
        ExpressionSafetyValidator().validate("1e400")
