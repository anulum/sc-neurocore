# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (handoff_reject) from former test_scnir_handoff_audit.py

from __future__ import annotations

from tests.scnir_handoff_audit_support import *  # noqa: F403


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


def test_audit_scnir_hdl_handoff_rejects_missing_hierarchy_top_instance(
    tmp_path: Path,
) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    top_path = handoff / "mixed_audit_net.v"
    top_path.write_text(
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


def test_audit_scnir_hdl_handoff_rejects_transform_metadata_mismatch(
    tmp_path: Path,
) -> None:
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


def test_audit_rejects_missing_directory(tmp_path: Path) -> None:
    """A non-existent handoff directory is rejected up front."""
    with pytest.raises(SCNIRHDLHandoffAuditError, match="does not exist"):
        audit_scnir_hdl_handoff(tmp_path / "absent")


def test_audit_rejects_unparsable_document(tmp_path: Path) -> None:
    """A corrupt scnir_document.json is reported as invalid."""
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    (handoff / "scnir_document.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(SCNIRHDLHandoffAuditError, match="invalid scnir_document.json"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_unparsable_manifest(tmp_path: Path) -> None:
    """A corrupt scnir_source_manifest.json is reported as invalid."""
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    (handoff / "scnir_source_manifest.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(SCNIRHDLHandoffAuditError, match="invalid scnir_source_manifest.json"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_non_object_manifest(tmp_path: Path) -> None:
    """A manifest that is a JSON array rather than an object is rejected."""
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    (handoff / "scnir_source_manifest.json").write_text("[]", encoding="utf-8")

    with pytest.raises(SCNIRHDLHandoffAuditError, match="must be a JSON object"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_manifest_key_mismatch(tmp_path: Path) -> None:
    """An unexpected manifest key is reported as a key mismatch."""
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest = _read_manifest(handoff)
    manifest["unexpected"] = 1
    _write_manifest(handoff, manifest)

    with pytest.raises(SCNIRHDLHandoffAuditError, match="source manifest keys mismatch"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_manifest_schema_version(tmp_path: Path) -> None:
    """A manifest with the wrong schema version is rejected."""
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest = _read_manifest(handoff)
    manifest["schema_version"] = "sc-neurocore.scnir.hdl-sources.v0.1"
    _write_manifest(handoff, manifest)

    with pytest.raises(SCNIRHDLHandoffAuditError, match="schema_version must be"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_stream_count_mismatch(tmp_path: Path) -> None:
    """A scnir_stream_count that disagrees with the document is rejected."""
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest = _read_manifest(handoff)
    manifest["scnir_stream_count"] = 99
    _write_manifest(handoff, manifest)

    with pytest.raises(SCNIRHDLHandoffAuditError, match="does not match document stream count"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_sources_length_mismatch(tmp_path: Path) -> None:
    """A sources array shorter than the document stream set is rejected."""
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest = _read_manifest(handoff)
    manifest["sources"] = manifest["sources"][:-1]
    _write_manifest(handoff, manifest)

    with pytest.raises(SCNIRHDLHandoffAuditError, match="sources length"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_external_input_key_mismatch(tmp_path: Path) -> None:
    """An external-input row with an unexpected key is rejected."""
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
    """Two external-input rows sharing a source name are rejected."""
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest = _read_manifest(handoff)
    manifest["scnir_external_inputs"][1]["source"] = manifest["scnir_external_inputs"][0]["source"]
    _write_manifest(handoff, manifest)

    with pytest.raises(SCNIRHDLHandoffAuditError, match="duplicate source"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_external_input_non_positive_width(tmp_path: Path) -> None:
    """An external-input row with a non-positive width is rejected."""
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest = _read_manifest(handoff)
    manifest["scnir_external_inputs"] = [{"source": "only", "offset": 0, "width": 0}]
    _write_manifest(handoff, manifest)

    with pytest.raises(SCNIRHDLHandoffAuditError, match="width must be positive"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_duplicate_source_row(tmp_path: Path) -> None:
    """Two source rows for the same stream id are rejected."""
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest = _read_manifest(handoff)
    manifest["sources"][1]["stream_id"] = manifest["sources"][0]["stream_id"]
    _write_manifest(handoff, manifest)

    with pytest.raises(SCNIRHDLHandoffAuditError, match="duplicate source row"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_source_row_for_unknown_stream(tmp_path: Path) -> None:
    """A source row referencing a stream absent from the document is rejected."""
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest = _read_manifest(handoff)
    manifest["sources"][0]["stream_id"] = "pop.ghost.spike"
    _write_manifest(handoff, manifest)

    with pytest.raises(SCNIRHDLHandoffAuditError, match="unknown stream_id"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_hierarchy_module_without_declaration(tmp_path: Path) -> None:
    """A hierarchy module file that does not declare its module is rejected."""
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    (handoff / "mixed_audit_net_core.v").write_text("// empty\n", encoding="utf-8")

    with pytest.raises(SCNIRHDLHandoffAuditError, match="does not declare module"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_hierarchy_module_missing_port(tmp_path: Path) -> None:
    """A hierarchy module file that omits a declared port is rejected."""
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    (handoff / "mixed_audit_net_core.v").write_text(
        "module mixed_audit_net_core;\nendmodule\n", encoding="utf-8"
    )

    with pytest.raises(SCNIRHDLHandoffAuditError, match="is missing port"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_top_instance_missing_port(tmp_path: Path) -> None:
    """A top module hierarchy instance that omits a port connection is rejected."""
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    top_path = handoff / "mixed_audit_net.v"
    top = top_path.read_text(encoding="utf-8").replace(
        "    .weight_i(mixed_audit_net_core__weight_i)\n", ""
    )
    top_path.write_text(top, encoding="utf-8")

    with pytest.raises(SCNIRHDLHandoffAuditError, match="missing port"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_missing_top_localparam(tmp_path: Path) -> None:
    """A top module missing a required localparam declaration is rejected."""
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    top_path = handoff / "mixed_audit_net.v"
    top = top_path.read_text(encoding="utf-8").replace(
        "localparam integer SCNIR_BITSTREAM_LENGTH = 512;\n", ""
    )
    top_path.write_text(top, encoding="utf-8")

    with pytest.raises(SCNIRHDLHandoffAuditError, match="top module missing"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_source_row_key_mismatch(tmp_path: Path) -> None:
    """A source row carrying an unexpected key is rejected as a keys mismatch."""
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest = _read_manifest(handoff)
    manifest["sources"][0]["unexpected"] = 1
    _write_manifest(handoff, manifest)

    with pytest.raises(SCNIRHDLHandoffAuditError, match=r"sources\[0\] keys mismatch"):
        audit_scnir_hdl_handoff(handoff)
