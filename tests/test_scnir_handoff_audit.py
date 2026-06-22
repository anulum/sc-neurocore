# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SC-NIR HDL handoff audit

"""Contract tests for executable SC-NIR HDL handoff audits."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from sc_neurocore.cli import main
from sc_neurocore.ir import (
    SCNIRDocument,
    SCNIRHierarchyInstance,
    SCNIRHierarchyPort,
    SCNIRPrecision,
    SCNIRSource,
    SCNIRStream,
    write_scnir,
)
from sc_neurocore.ir.scnir_handoff_audit import (
    SCNIRHDLHandoffAuditError,
    _delay_steps_for_row,
    _expect_int,
    _expect_mapping_sequence,
    _expect_non_empty_string,
    _expect_non_negative_int,
    _expect_positive_int,
    _verify_source_row_matches_stream,
    audit_scnir_hdl_handoff,
    write_scnir_hdl_handoff_audit,
)


def _document() -> SCNIRDocument:
    precision = SCNIRPrecision(
        signed=True,
        total_bits=16,
        fractional_bits=8,
        accumulator_bits=34,
        rounding="nearest_even",
        overflow="saturate",
    )
    return SCNIRDocument(
        producer="sc-neurocore-test",
        streams=(
            SCNIRStream(
                stream_id="pop.li.state",
                layer="li",
                bitstream_length=512,
                encoding="bipolar",
                signal_kind="analogue_state",
                precision=precision,
                source=SCNIRSource(kind="sobol", seed=91, sobol_dimension=1),
            ),
            SCNIRStream(
                stream_id="pop.lif.spike",
                layer="lif",
                bitstream_length=512,
                encoding="unipolar",
                signal_kind="spike",
                precision=precision,
                source=SCNIRSource(kind="sobol", seed=92, sobol_dimension=2),
            ),
            SCNIRStream(
                stream_id="conn.li_to_lif.weight",
                layer="lif",
                bitstream_length=512,
                encoding="bipolar",
                signal_kind="weight",
                precision=precision,
                source=SCNIRSource(kind="sobol", seed=93, sobol_dimension=3),
            ),
        ),
        hierarchy=(
            SCNIRHierarchyInstance(
                instance_id="top.mixed",
                module_name="mixed_audit_net_core",
                ports=(
                    SCNIRHierarchyPort(
                        port_name="li_state_i",
                        direction="input",
                        stream_id="pop.li.state",
                        signal_kind="analogue_state",
                        bit_width=16,
                    ),
                    SCNIRHierarchyPort(
                        port_name="lif_spike_o",
                        direction="output",
                        stream_id="pop.lif.spike",
                        signal_kind="spike",
                        bit_width=1,
                    ),
                    SCNIRHierarchyPort(
                        port_name="weight_i",
                        direction="input",
                        stream_id="conn.li_to_lif.weight",
                        signal_kind="weight",
                        bit_width=16,
                    ),
                ),
            ),
        ),
    )


def _write_valid_handoff(root: Path) -> None:
    root.mkdir()
    write_scnir(root / "scnir_document.json", _document())
    manifest = {
        "schema_version": "sc-neurocore.scnir.hdl-sources.v0.2",
        "module_name": "mixed_audit_net",
        "bitstream_length": 512,
        "source_kind": "sobol",
        "interconnect": "direct",
        "q_format": "Q8.8",
        "total_neurons": 3,
        "total_synapses": 6,
        "scnir_stream_count": 3,
        "scnir_signal_kinds": {"analogue_state": 1, "spike": 1, "weight": 1},
        "scnir_signal_routes": {
            "analogue_state": "direct_mac",
            "spike": "direct_wire",
            "weight": "stochastic_source_module",
        },
        "scnir_external_inputs": [
            {"source": "sensor_a", "offset": 0, "width": 2},
            {"source": "sensor_b", "offset": 2, "width": 1},
        ],
        "scnir_hierarchy_instance_count": 1,
        "scnir_hierarchy_port_count": 3,
        "sources": [
            {
                "stream_id": "pop.li.state",
                "layer": "li",
                "module_name": "scnir_src_000_pop_li_state",
                "source_kind": "sobol16",
                "seed": 91,
                "bitstream_length": 512,
                "encoding": "bipolar",
                "signal_kind": "analogue_state",
                "delay_steps": 0,
                "total_bits": 16,
                "fractional_bits": 8,
                "transforms": [],
                "online_learning": None,
                "lfsr_polynomial": None,
                "tap_mask": None,
                "sobol_dimension": 1,
            },
            {
                "stream_id": "pop.lif.spike",
                "layer": "lif",
                "module_name": "scnir_src_001_pop_lif_spike",
                "source_kind": "sobol16",
                "seed": 92,
                "bitstream_length": 512,
                "encoding": "unipolar",
                "signal_kind": "spike",
                "delay_steps": 0,
                "total_bits": 16,
                "fractional_bits": 8,
                "transforms": [],
                "online_learning": None,
                "lfsr_polynomial": None,
                "tap_mask": None,
                "sobol_dimension": 2,
            },
            {
                "stream_id": "conn.li_to_lif.weight",
                "layer": "lif",
                "module_name": "scnir_src_002_conn_li_to_lif_weight",
                "source_kind": "sobol16",
                "seed": 93,
                "bitstream_length": 512,
                "encoding": "bipolar",
                "signal_kind": "weight",
                "delay_steps": 0,
                "total_bits": 16,
                "fractional_bits": 8,
                "transforms": [],
                "online_learning": None,
                "lfsr_polynomial": None,
                "tap_mask": None,
                "sobol_dimension": 3,
            },
        ],
    }
    (root / "scnir_source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "mixed_audit_net.v").write_text(
        "\n".join(
            [
                "module mixed_audit_net;",
                "localparam integer SCNIR_BITSTREAM_LENGTH = 512;",
                "localparam integer SCNIR_STREAM_COUNT = 3;",
                "localparam integer SCNIR_SOURCE_MODULE_COUNT = 3;",
                "wire signed [15:0] mixed_audit_net_core__li_state_i;",
                "wire mixed_audit_net_core__lif_spike_o;",
                "wire signed [15:0] mixed_audit_net_core__weight_i;",
                "mixed_audit_net_core mixed_audit_net_core_hierarchy_inst (",
                "    .li_state_i(mixed_audit_net_core__li_state_i),",
                "    .lif_spike_o(mixed_audit_net_core__lif_spike_o),",
                "    .weight_i(mixed_audit_net_core__weight_i)",
                ");",
                "endmodule",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "sc_nir_weight_rom.v").write_text("module sc_nir_weight_rom; endmodule\n")
    for row in manifest["sources"]:
        (root / f"{row['module_name']}.v").write_text(
            f"module {row['module_name']}; endmodule\n",
            encoding="utf-8",
        )
    (root / "mixed_audit_net_core.v").write_text(
        "\n".join(
            [
                "module mixed_audit_net_core(",
                "    input wire signed [15:0] li_state_i,",
                "    output wire lif_spike_o,",
                "    input wire signed [15:0] weight_i",
                ");",
                "assign lif_spike_o = 1'b0;",
                "endmodule",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_audit_scnir_hdl_handoff_accepts_complete_compile_output(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)

    report = audit_scnir_hdl_handoff(handoff)

    assert report.module_name == "mixed_audit_net"
    assert report.stream_count == 3
    assert report.source_module_count == 3
    assert report.hierarchy_instance_count == 1
    assert report.hierarchy_port_count == 3
    assert report.hierarchy_instances == {
        "top.mixed": {
            "module_name": "mixed_audit_net_core",
            "ports": [
                {
                    "port_name": "li_state_i",
                    "direction": "input",
                    "stream_id": "pop.li.state",
                    "signal_kind": "analogue_state",
                    "bit_width": 16,
                },
                {
                    "port_name": "lif_spike_o",
                    "direction": "output",
                    "stream_id": "pop.lif.spike",
                    "signal_kind": "spike",
                    "bit_width": 1,
                },
                {
                    "port_name": "weight_i",
                    "direction": "input",
                    "stream_id": "conn.li_to_lif.weight",
                    "signal_kind": "weight",
                    "bit_width": 16,
                },
            ],
        }
    }
    assert report.signal_routes["analogue_state"] == "direct_mac"
    assert report.external_input_count == 2
    assert report.external_inputs == (
        {"source": "sensor_a", "offset": 0, "width": 2},
        {"source": "sensor_b", "offset": 2, "width": 1},
    )
    assert "scnir_document.json" in report.artefacts
    assert report.as_dict()["external_inputs"] == [
        {"source": "sensor_a", "offset": 0, "width": 2},
        {"source": "sensor_b", "offset": 2, "width": 1},
    ]
    assert report.as_dict()["hierarchy_port_count"] == 3
    assert report.as_dict()["status"] == "valid"
    assert "mixed_audit_net_core.v" in report.artefacts


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


def test_scnir_audit_hdl_cli_writes_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    handoff = tmp_path / "handoff"
    report_path = tmp_path / "audit.json"
    _write_valid_handoff(handoff)

    with mock.patch(
        "sys.argv",
        [
            "sc-neurocore",
            "scnir",
            "audit-hdl",
            str(handoff),
            "--output",
            str(report_path),
        ],
    ):
        rc = main()

    assert rc == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "valid"
    assert "SC-NIR HDL handoff valid" in capsys.readouterr().out


def test_scnir_audit_hdl_cli_reports_invalid_handoff(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    (handoff / "scnir_source_manifest.json").unlink()

    with mock.patch("sys.argv", ["sc-neurocore", "scnir", "audit-hdl", str(handoff)]):
        rc = main()

    assert rc == 1
    assert "SC-NIR HDL handoff invalid" in capsys.readouterr().out


def test_audit_scnir_hdl_handoff_accepts_real_compile_nir_output(tmp_path: Path) -> None:
    nir = pytest.importorskip("nir")
    model_path = tmp_path / "model.nir"
    out_dir = tmp_path / "compiled"
    graph = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.array([[0.25, -0.5], [0.75, 0.125]], dtype=np.float32),
                bias=np.zeros(2, dtype=np.float32),
            ),
            "lif": nir.LIF(
                tau=np.full(2, 20.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[("input", "aff"), ("aff", "lif"), ("lif", "output")],
    )
    nir.write(str(model_path), graph)

    with mock.patch(
        "sys.argv",
        [
            "sc-neurocore",
            "compile-nir",
            str(model_path),
            "--module-name",
            "real_handoff_net",
            "--T",
            "512",
            "--source-kind",
            "sobol",
            "--base-seed",
            "101",
            "--output",
            str(out_dir),
        ],
    ):
        rc = main()

    assert rc == 0
    report = audit_scnir_hdl_handoff(out_dir)
    assert report.module_name == "real_handoff_net"
    assert report.stream_count == 2
    assert report.source_module_count == 2
    assert report.hierarchy_instance_count == 0
    assert report.hierarchy_port_count == 0


def _read_manifest(root: Path) -> dict:
    return json.loads((root / "scnir_source_manifest.json").read_text(encoding="utf-8"))


def _write_manifest(root: Path, manifest: dict) -> None:
    (root / "scnir_source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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


def _lfsr_stream() -> SCNIRStream:
    return SCNIRStream(
        stream_id="s",
        layer="L",
        bitstream_length=512,
        encoding="bipolar",
        signal_kind="spike",
        precision=SCNIRPrecision(
            signed=True,
            total_bits=16,
            fractional_bits=8,
            accumulator_bits=32,
            rounding="nearest_even",
            overflow="saturate",
        ),
        source=SCNIRSource(kind="lfsr", seed=5, lfsr_polynomial="x^16", tap_mask=0xB400),
    )


def _lfsr_row() -> dict:
    return {
        "layer": "L",
        "bitstream_length": 512,
        "encoding": "bipolar",
        "signal_kind": "spike",
        "delay_steps": 0,
        "total_bits": 16,
        "fractional_bits": 8,
        "source_kind": "lfsr16",
        "transforms": [],
        "online_learning": None,
        "seed": 5,
        "lfsr_polynomial": "x^16",
        "tap_mask": 0xB400,
    }


def test_source_row_match_accepts_lfsr_fields() -> None:
    """A matching LFSR source row carries seed, polynomial and tap-mask expectations."""
    _verify_source_row_matches_stream(_lfsr_row(), _lfsr_stream(), 0)


def test_source_row_match_rejects_lfsr_tap_mask_mismatch() -> None:
    """A divergent LFSR tap mask is reported against the stream value."""
    row = _lfsr_row()
    row["tap_mask"] = 0x1234

    with pytest.raises(SCNIRHDLHandoffAuditError, match="tap_mask"):
        _verify_source_row_matches_stream(row, _lfsr_stream(), 0)


def test_delay_steps_for_row_expands_vector() -> None:
    """A per-source-column delay vector is materialised as a list of ints."""
    assert _delay_steps_for_row((1, 2, 3)) == [1, 2, 3]


def test_low_level_manifest_validators_reject_bad_values() -> None:
    """The structural manifest validators reject malformed sequences and scalars."""
    with pytest.raises(SCNIRHDLHandoffAuditError, match="must be a sequence"):
        _expect_mapping_sequence({"rows": "nope"}, "rows")
    with pytest.raises(SCNIRHDLHandoffAuditError, match="must be a JSON object"):
        _expect_mapping_sequence({"rows": [123]}, "rows")
    with pytest.raises(SCNIRHDLHandoffAuditError, match="must be a non-empty string"):
        _expect_non_empty_string({"k": ""}, "k")
    with pytest.raises(SCNIRHDLHandoffAuditError, match="must be an integer"):
        _expect_int({"k": "x"}, "k")
    with pytest.raises(SCNIRHDLHandoffAuditError, match="must be positive"):
        _expect_positive_int({"k": 0}, "k")
    with pytest.raises(SCNIRHDLHandoffAuditError, match="must be non-negative"):
        _expect_non_negative_int({"k": -1}, "k")


def test_audit_rejects_source_row_key_mismatch(tmp_path: Path) -> None:
    """A source row carrying an unexpected key is rejected as a keys mismatch."""
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest = _read_manifest(handoff)
    manifest["sources"][0]["unexpected"] = 1
    _write_manifest(handoff, manifest)

    with pytest.raises(SCNIRHDLHandoffAuditError, match=r"sources\[0\] keys mismatch"):
        audit_scnir_hdl_handoff(handoff)


def test_write_audit_report_emits_valid_json(tmp_path: Path) -> None:
    """Writing the audit report serialises the valid summary to JSON on disk."""
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    output = tmp_path / "audit.json"

    report = write_scnir_hdl_handoff_audit(handoff, output)

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["status"] == "valid"
    assert written["module_name"] == report.module_name
    assert written == report.as_dict()
