# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for block-floating mode parsing and exponent layouts

"""Contracts for block-floating format parsing and shared-exponent layout validation."""

from __future__ import annotations

from typing import Any

import pytest

from sc_neurocore.compiler.block_floating import BlockExponentLayout, BlockFloatingMode


def test_block_exponent_count_is_zero_for_empty_payload() -> None:
    """A zero-parameter payload needs no shared exponents."""
    mode = BlockFloatingMode(mantissa_bits=16, exponent_bits=3, block_size=32)
    assert mode.block_exponent_count(0) == 0


def test_from_string_parses_canonical_and_blocked_formats() -> None:
    """from_string parses the canonical BFP form with and without an explicit block."""
    assert BlockFloatingMode.from_string("BFP16E3").block_size == 32
    assert BlockFloatingMode.from_string("BFP16E3X64").block_size == 64


@pytest.mark.parametrize(
    "fmt",
    ["XYZ16E3", "BFP", "BFPxyz", "BFP1E3", "BFP16E0", "BFP16E3X0"],
)
def test_from_string_rejects_malformed_formats(fmt: str) -> None:
    """from_string rejects non-BFP, empty, unparsable and out-of-range formats."""
    with pytest.raises(ValueError):
        BlockFloatingMode.from_string(fmt)


def test_from_aliases_accepts_tolerant_separators() -> None:
    """from_aliases accepts underscore separators and explicit block sizes."""
    assert BlockFloatingMode.from_aliases("BFP16_E3").exponent_bits == 3
    assert BlockFloatingMode.from_aliases("BFP16E3X64").block_size == 64


def test_from_aliases_rejects_non_string() -> None:
    """from_aliases rejects a non-string format argument."""
    with pytest.raises(TypeError):
        BlockFloatingMode.from_aliases(123)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "fmt",
    ["XYZ", "BFP16E3X32X4", "BFP16", "BFP16E3Xabc"],
)
def test_from_aliases_rejects_malformed_formats(fmt: str) -> None:
    """from_aliases rejects non-BFP, over-split, exponent-less and bad-block aliases."""
    with pytest.raises(ValueError):
        BlockFloatingMode.from_aliases(fmt)


def _valid_layout_kwargs() -> dict[str, Any]:
    return {"parameter_count": 10, "block_size": 4, "exponent_count": 3}


@pytest.mark.parametrize(
    ("override", "exc"),
    [
        ({"parameter_count": 1.0}, TypeError),
        ({"block_size": 1.0}, TypeError),
        ({"exponent_count": 1.0}, TypeError),
        ({"parameter_count": -1, "exponent_count": 0}, ValueError),
        ({"block_size": 0}, ValueError),
        ({"exponent_count": 99}, ValueError),
        ({"alignment": "weird"}, ValueError),
        ({"flattened_order": "col_major"}, ValueError),
    ],
)
def test_block_exponent_layout_rejects_invalid_fields(
    override: dict[str, Any], exc: type[Exception]
) -> None:
    """Each BlockExponentLayout invariant rejects its malformed field."""
    kwargs = _valid_layout_kwargs()
    kwargs.update(override)
    with pytest.raises(exc):
        BlockExponentLayout(**kwargs)


def test_block_exponent_layout_last_block_size_for_empty_payload() -> None:
    """A zero-parameter layout reports a zero final block size."""
    layout = BlockExponentLayout(parameter_count=0, block_size=4, exponent_count=0)
    assert layout.last_block_size == 0
