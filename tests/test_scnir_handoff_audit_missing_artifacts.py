# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCNIR handoff artifact and metadata rejection contracts

"""Reject missing generated artifacts and mismatched route/source metadata."""

import json
from pathlib import Path

import pytest

from tests.scnir_handoff_audit_support import (
    SCNIRHDLHandoffAuditError,
    _write_valid_handoff,
    audit_scnir_hdl_handoff,
)


def test_audit_scnir_hdl_handoff_rejects_missing_source_module(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    (handoff / "scnir_src_002_conn_li_to_lif_weight.v").unlink()
    with pytest.raises(SCNIRHDLHandoffAuditError, match="source module file"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_scnir_hdl_handoff_rejects_missing_hierarchy_module(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    (handoff / "mixed_audit_net_core.v").unlink()
    with pytest.raises(SCNIRHDLHandoffAuditError, match="hierarchy module"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_scnir_hdl_handoff_rejects_missing_hierarchy_top_instance(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    (handoff / "mixed_audit_net.v").write_text(
        "\n".join(
            [
                "module mixed_audit_net;",
                "localparam integer SCNIR_BITSTREAM_LENGTH = 512;",
                "localparam integer SCNIR_STREAM_COUNT = 3;",
                "localparam integer SCNIR_SOURCE_MODULE_COUNT = 3;",
                "endmodule",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SCNIRHDLHandoffAuditError, match="hierarchy instance"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_scnir_hdl_handoff_rejects_route_mismatch(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest_path = handoff / "scnir_source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scnir_signal_routes"]["analogue_state"] = "weighted_event_aer"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SCNIRHDLHandoffAuditError, match="scnir_signal_routes"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_scnir_hdl_handoff_rejects_external_input_layout_gap(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest_path = handoff / "scnir_source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scnir_external_inputs"][1]["offset"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SCNIRHDLHandoffAuditError, match="scnir_external_inputs"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_scnir_hdl_handoff_rejects_source_metadata_mismatch(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest_path = handoff / "scnir_source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"][0]["sobol_dimension"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SCNIRHDLHandoffAuditError, match="sobol_dimension"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_scnir_hdl_handoff_rejects_transform_metadata_mismatch(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest_path = handoff / "scnir_source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"][2]["transforms"] = [
        {
            "kind": "threshold",
            "position": "source",
            "comparison": "greater_than",
            "values": [0.25],
        }
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SCNIRHDLHandoffAuditError, match="transforms"):
        audit_scnir_hdl_handoff(handoff)
