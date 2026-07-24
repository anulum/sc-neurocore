# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_scnir_handoff_audit.py

from __future__ import annotations


"""Contract tests for executable SC-NIR HDL handoff audits."""


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


def _read_manifest(root: Path) -> dict:
    return json.loads((root / "scnir_source_manifest.json").read_text(encoding="utf-8"))


def _write_manifest(root: Path, manifest: dict) -> None:
    (root / "scnir_source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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


__all__ = ['json', 'Path', 'mock', 'np', 'pytest', 'main', 'SCNIRDocument', 'SCNIRHierarchyInstance', 'SCNIRHierarchyPort', 'SCNIRPrecision', 'SCNIRSource', 'SCNIRStream', 'write_scnir', 'SCNIRHDLHandoffAuditError', '_delay_steps_for_row', '_expect_int', '_expect_mapping_sequence', '_expect_non_empty_string', '_expect_non_negative_int', '_expect_positive_int', '_verify_source_row_matches_stream', 'audit_scnir_hdl_handoff', 'write_scnir_hdl_handoff_audit', '_document', '_write_valid_handoff', '_read_manifest', '_write_manifest', '_lfsr_stream', '_lfsr_row']

