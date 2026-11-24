# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (validate_flags) from former test_scnir_compatibility.py

"""The findings the SC-NIR compatibility-matrix validator flags."""

from __future__ import annotations

import pytest
from tests.scnir_compatibility_support import (
    _patch_matrix,
    _row,
    validate_scnir_compatibility_matrix,
)


def test_validate_matrix_flags_missing_parser_primitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate matrix flags missing parser primitive."""
    _patch_matrix(monkeypatch, ())
    with pytest.raises(ValueError, match="misses parser primitives"):
        validate_scnir_compatibility_matrix()


def test_validate_matrix_flags_stale_primitive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate matrix flags stale primitive."""
    _patch_matrix(monkeypatch, (_row("_Foo"), _row("Ghost")))
    with pytest.raises(ValueError, match="stale primitives"):
        validate_scnir_compatibility_matrix()


def test_validate_matrix_flags_duplicate_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate matrix flags duplicate row."""
    _patch_matrix(monkeypatch, (_row("_Foo"), _row("_Foo")))
    with pytest.raises(ValueError, match="duplicate SC-NIR compatibility row"):
        validate_scnir_compatibility_matrix()


def test_validate_matrix_flags_hdl_support_without_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate matrix flags HDL support without metadata."""
    _patch_matrix(
        monkeypatch, (_row("_Foo", support_level="metadata_and_hdl", stream_metadata=()),)
    )
    with pytest.raises(ValueError, match="claims HDL support without stream metadata"):
        validate_scnir_compatibility_matrix()


def test_validate_matrix_flags_missing_audit_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate matrix flags missing audit evidence."""
    _patch_matrix(monkeypatch, (_row("_Foo", audit_evidence=()),))
    with pytest.raises(ValueError, match="no audit evidence pointer"):
        validate_scnir_compatibility_matrix()
