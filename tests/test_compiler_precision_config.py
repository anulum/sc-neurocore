# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for fixed-point and block-floating precision configs

"""Contracts for the fixed-point and block-floating precision configurations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sc_neurocore.compiler.precision_config import (
    BlockFloatingPrecisionConfig,
    PrecisionConfig,
)


def _bfp() -> BlockFloatingPrecisionConfig:
    return BlockFloatingPrecisionConfig(mantissa_bits=16, exponent_bits=3, block_size=32)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mantissa_bits": 1.0, "exponent_bits": 3, "block_size": 32},
        {"mantissa_bits": 16, "exponent_bits": 1.0, "block_size": 32},
        {"mantissa_bits": 16, "exponent_bits": 3, "block_size": 1.0},
    ],
)
def test_block_floating_config_rejects_non_integer_fields(kwargs: dict[str, Any]) -> None:
    """BlockFloatingPrecisionConfig rejects non-integer mantissa/exponent/block widths."""
    with pytest.raises(TypeError):
        BlockFloatingPrecisionConfig(**kwargs)


def test_block_floating_config_properties_and_delegation() -> None:
    """The block-floating config exposes int_bits, the block-floating flag and exponent layout."""
    bfp = _bfp()

    assert bfp.int_bits == 15
    assert bfp.is_block_floating is True

    count = bfp.block_exponent_count(100)
    assert count >= 1
    validated = bfp.validate_exponents(np.zeros(count, dtype=np.int64), parameter_count=100)
    assert validated.shape == (count,)


def test_block_floating_config_encode_is_unsupported() -> None:
    """Block-floating encoding without per-block exponents is unsupported."""
    with pytest.raises(NotImplementedError):
        _bfp().encode(1.0)


def test_precision_config_is_not_block_floating_and_encodes() -> None:
    """A fixed-point config reports not-block-floating and encodes signed/unsigned values."""
    signed = PrecisionConfig(8, 7)
    unsigned = PrecisionConfig(8, 7, signed=False)

    assert signed.is_block_floating is False
    assert isinstance(signed.encode(0.5), int)
    assert isinstance(unsigned.encode(0.5), int)
