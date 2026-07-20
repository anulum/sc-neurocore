# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Q-format validation contracts

"""Contracts for scalar and mixed Q-format validation."""

from __future__ import annotations

import pytest

from sc_neurocore.compiler.q_format import QFormat, QFormatMixed


@pytest.mark.parametrize(
    "kwargs",
    [
        {"integer_bits": 1.0, "fraction_bits": 8},
        {"integer_bits": 8, "fraction_bits": 1.0},
    ],
)
def test_qformat_rejects_non_integer_fields(kwargs: dict[str, object]) -> None:
    """QFormat rejects non-integer bit widths."""
    with pytest.raises(TypeError):
        QFormat(**kwargs)  # type: ignore[arg-type]


def test_qformat_from_string_rejects_non_string() -> None:
    """QFormat.from_string rejects a non-string format."""
    with pytest.raises(TypeError):
        QFormat.from_string(123)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "override",
    [
        {"weight_fmt": "not a qformat"},
        {"accum_fmt": "not a qformat"},
        {"scale_per_tensor": "yes"},
        {"rounding": "bogus"},
        {"accum_fmt": QFormat(4, 4)},
        {"accum_fmt": QFormat(16, 2)},
    ],
)
def test_qformat_mixed_rejects_invalid_fields(override: dict[str, object]) -> None:
    """QFormatMixed rejects malformed formats, flags and rounding modes."""
    valid = {
        "weight_fmt": QFormat(8, 8),
        "accum_fmt": QFormat(16, 16),
        "scale_per_tensor": True,
        "rounding": "nearest",
    }
    valid.update(override)

    with pytest.raises((TypeError, ValueError)):
        QFormatMixed(**valid)  # type: ignore[arg-type]
