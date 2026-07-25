# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCNIR source and external-input layout rejection contracts

"""Reject invalid external-input rows and source-to-stream mappings."""

from pathlib import Path

import pytest

from tests.scnir_handoff_audit_support import (
    SCNIRHDLHandoffAuditError,
    _read_manifest,
    _write_manifest,
    _write_valid_handoff,
    audit_scnir_hdl_handoff,
)


def test_audit_rejects_external_input_key_mismatch(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest = _read_manifest(handoff)
    manifest["scnir_external_inputs"][0]["extra"] = 1
    _write_manifest(handoff, manifest)
    with pytest.raises(
        SCNIRHDLHandoffAuditError, match=r"scnir_external_inputs\[0\] keys mismatch"
    ):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_external_input_duplicate_source(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest = _read_manifest(handoff)
    manifest["scnir_external_inputs"][1]["source"] = manifest["scnir_external_inputs"][0]["source"]
    _write_manifest(handoff, manifest)
    with pytest.raises(SCNIRHDLHandoffAuditError, match="duplicate source"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_external_input_non_positive_width(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest = _read_manifest(handoff)
    manifest["scnir_external_inputs"] = [{"source": "only", "offset": 0, "width": 0}]
    _write_manifest(handoff, manifest)
    with pytest.raises(SCNIRHDLHandoffAuditError, match="width must be positive"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_duplicate_source_row(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest = _read_manifest(handoff)
    manifest["sources"][1]["stream_id"] = manifest["sources"][0]["stream_id"]
    _write_manifest(handoff, manifest)
    with pytest.raises(SCNIRHDLHandoffAuditError, match="duplicate source row"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_source_row_for_unknown_stream(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest = _read_manifest(handoff)
    manifest["sources"][0]["stream_id"] = "pop.ghost.spike"
    _write_manifest(handoff, manifest)
    with pytest.raises(SCNIRHDLHandoffAuditError, match="unknown stream_id"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_source_row_key_mismatch(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest = _read_manifest(handoff)
    manifest["sources"][0]["unexpected"] = 1
    _write_manifest(handoff, manifest)
    with pytest.raises(SCNIRHDLHandoffAuditError, match=r"sources\[0\] keys mismatch"):
        audit_scnir_hdl_handoff(handoff)
