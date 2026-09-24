# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Aggregate evidence input configuration

"""Explicit byte budget defaults, exact overrides and fail-closed validation."""

from typing import cast

import pytest

from sc_neurocore.studio.platform.evidence_limits import (
    DEFAULT_EVIDENCE_MAX_INPUT_BYTES,
    parse_evidence_input_limit,
    validate_evidence_input_limit,
)


@pytest.mark.parametrize("value", [1, 256 * 1024 * 1024, 2**40])
def test_limit_preserves_explicit_positive_integer_bytes(value: int) -> None:
    """There is no hidden clamp, unit conversion or truncation of byte counts."""
    assert validate_evidence_input_limit(value) == value
    assert parse_evidence_input_limit(str(value)) == value


def test_absent_override_uses_documented_default() -> None:
    """Only absence selects the256MiB operational default."""
    assert parse_evidence_input_limit(None) == DEFAULT_EVIDENCE_MAX_INPUT_BYTES == 268435456


@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, "1", None])
def test_invalid_native_limit_is_not_coerced(value: object) -> None:
    """Reject invalid untyped callers, including bool-as-int and floats."""
    with pytest.raises(ValueError, match="positive integer"):
        validate_evidence_input_limit(cast(int, value))


@pytest.mark.parametrize("value", ["", " ", "0", "-1", "1.5", "false"])
def test_invalid_explicit_override_never_falls_back(value: str) -> None:
    """An invalid environment override refuses instead of hiding bad config."""
    with pytest.raises(ValueError):
        parse_evidence_input_limit(value)
