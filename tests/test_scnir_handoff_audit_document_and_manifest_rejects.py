# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCNIR document and manifest rejection contracts

"""Reject absent, unparsable, structurally invalid, or inconsistent manifests."""

from pathlib import Path

import pytest

from tests.scnir_handoff_audit_support import (
    SCNIRHDLHandoffAuditError,
    _read_manifest,
    _write_manifest,
    _write_valid_handoff,
    audit_scnir_hdl_handoff,
)


def test_audit_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(SCNIRHDLHandoffAuditError, match="does not exist"):
        audit_scnir_hdl_handoff(tmp_path / "absent")


def test_audit_rejects_unparsable_document(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    (handoff / "scnir_document.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SCNIRHDLHandoffAuditError, match="invalid scnir_document.json"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_unparsable_manifest(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    (handoff / "scnir_source_manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SCNIRHDLHandoffAuditError, match="invalid scnir_source_manifest.json"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_non_object_manifest(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    (handoff / "scnir_source_manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(SCNIRHDLHandoffAuditError, match="must be a JSON object"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_manifest_key_mismatch(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest = _read_manifest(handoff)
    manifest["unexpected"] = 1
    _write_manifest(handoff, manifest)
    with pytest.raises(SCNIRHDLHandoffAuditError, match="source manifest keys mismatch"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_manifest_schema_version(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest = _read_manifest(handoff)
    manifest["schema_version"] = "sc-neurocore.scnir.hdl-sources.v0.1"
    _write_manifest(handoff, manifest)
    with pytest.raises(SCNIRHDLHandoffAuditError, match="schema_version must be"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_stream_count_mismatch(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest = _read_manifest(handoff)
    manifest["scnir_stream_count"] = 99
    _write_manifest(handoff, manifest)
    with pytest.raises(SCNIRHDLHandoffAuditError, match="does not match document stream count"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_sources_length_mismatch(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest = _read_manifest(handoff)
    manifest["sources"] = manifest["sources"][:-1]
    _write_manifest(handoff, manifest)
    with pytest.raises(SCNIRHDLHandoffAuditError, match="sources length"):
        audit_scnir_hdl_handoff(handoff)
