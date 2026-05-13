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
    SCNIRPrecision,
    SCNIRSource,
    SCNIRStream,
    write_scnir,
)
from sc_neurocore.ir.scnir_handoff_audit import (
    SCNIRHDLHandoffAuditError,
    audit_scnir_hdl_handoff,
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
    )


def _write_valid_handoff(root: Path) -> None:
    root.mkdir()
    write_scnir(root / "scnir_document.json", _document())
    manifest = {
        "schema_version": "sc-neurocore.scnir.hdl-sources.v0.1",
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


def test_audit_scnir_hdl_handoff_accepts_complete_compile_output(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)

    report = audit_scnir_hdl_handoff(handoff)

    assert report.module_name == "mixed_audit_net"
    assert report.stream_count == 3
    assert report.source_module_count == 3
    assert report.signal_routes["analogue_state"] == "direct_mac"
    assert "scnir_document.json" in report.artefacts
    assert report.as_dict()["status"] == "valid"


def test_audit_scnir_hdl_handoff_rejects_missing_source_module(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    (handoff / "scnir_src_002_conn_li_to_lif_weight.v").unlink()

    with pytest.raises(SCNIRHDLHandoffAuditError, match="source module file"):
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


def test_audit_scnir_hdl_handoff_rejects_source_metadata_mismatch(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    manifest_path = handoff / "scnir_source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"][0]["sobol_dimension"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SCNIRHDLHandoffAuditError, match="sobol_dimension"):
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
