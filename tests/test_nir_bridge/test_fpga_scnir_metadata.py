# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SC-NIR FPGA integration

"""SC-NIR metadata integration tests for FPGA compilation artefacts."""

from __future__ import annotations

import pytest

nir = pytest.importorskip("nir")

from sc_neurocore.ir import scnir_to_dict, validate_scnir_dict
from sc_neurocore.nir_bridge import compile_network_to_fpga, from_nir, from_scnetwork

from tests.test_nir_bridge.fpga_metadata_graphs import (
    _flattened_input_graph,
    _integrator_graph,
    _post_weight_scale_graph,
    _post_weight_threshold_graph,
    _source_scale_graph,
    _source_threshold_graph,
)


def test_fpga_compile_emits_integrator_module_and_analogue_state_routes() -> None:
    network = from_nir(_integrator_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    result = compile_network_to_fpga(
        neuron_graph,
        module_name="scnir_integrator",
        bitstream_length=704,
    )

    payload = scnir_to_dict(result.scnir_document)
    validate_scnir_dict(payload)
    streams = {stream["stream_id"]: stream for stream in payload["streams"]}
    assert streams["pop.i.state"]["signal_kind"] == "analogue_state"
    assert streams["conn.i_to_lif.weight"]["signal_kind"] == "weight"
    assert "integrator" in result.neuron_modules
    assert "module sc_nir_integrator" in result.neuron_modules["integrator"]
    assert "sc_nir_integrator p0_n0_inst" in result.top_module
    assert "p0_n0_v * 16'sh0080" in result.top_module
    assert "p0_n1_v * 16'shffc0" in result.top_module
    assert not any("Unsupported neuron type 'integrator'" in item for item in result.warnings)


def test_fpga_compile_preserves_source_side_scale_in_weight_terms() -> None:
    network = from_nir(_source_scale_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    result = compile_network_to_fpga(
        neuron_graph,
        module_name="scnir_source_scale",
        bitstream_length=768,
    )

    payload = scnir_to_dict(result.scnir_document)
    validate_scnir_dict(payload)
    assert "conn.li_to_lif.weight" in {stream["stream_id"] for stream in payload["streams"]}
    assert "p0_n0_v * 16'sh0080" in result.top_module
    assert "p0_n1_v * 16'shffe0" in result.top_module


def test_fpga_compile_preserves_post_weight_scale_in_rows_and_bias() -> None:
    network = from_nir(_post_weight_scale_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    result = compile_network_to_fpga(
        neuron_graph,
        module_name="scnir_post_weight_scale",
        bitstream_length=768,
    )

    payload = scnir_to_dict(result.scnir_document)
    validate_scnir_dict(payload)
    assert "conn.input_to_lif.weight" in {stream["stream_id"] for stream in payload["streams"]}
    assert "ext_input_0 * 16'sh0080" in result.top_module
    assert "ext_input_1 * 16'shff00" in result.top_module
    assert "34'sh000000033" in result.top_module
    assert "34'sh3ffffffe6" in result.top_module


def test_fpga_compile_preserves_flattened_input_weight_terms() -> None:
    network = from_nir(_flattened_input_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    result = compile_network_to_fpga(
        neuron_graph,
        module_name="scnir_flattened_input",
        bitstream_length=768,
    )

    payload = scnir_to_dict(result.scnir_document)
    validate_scnir_dict(payload)
    assert "conn.input_to_lif.weight" in {stream["stream_id"] for stream in payload["streams"]}
    assert "ext_input_0 * 16'sh0040" in result.top_module
    assert "ext_input_1 * 16'shff80" in result.top_module
    assert "ext_input_2 * 16'sh0020" in result.top_module
    assert "ext_input_3 * 16'sh00c0" in result.top_module
    assert "localparam integer SCNIR_STREAM_COUNT = 2;" in result.top_module


def test_fpga_compile_preserves_source_side_threshold_comparators() -> None:
    network = from_nir(_source_threshold_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    result = compile_network_to_fpga(
        neuron_graph,
        module_name="scnir_source_threshold",
        bitstream_length=768,
    )

    payload = scnir_to_dict(result.scnir_document)
    validate_scnir_dict(payload)
    stream = next(
        stream for stream in payload["streams"] if stream["stream_id"] == "conn.li_to_lif.weight"
    )
    assert stream["transforms"][0]["position"] == "source"
    manifest = {entry.stream_id: entry.as_dict() for entry in result.scnir_source_manifest}
    assert manifest["conn.li_to_lif.weight"]["transforms"] == stream["transforms"]
    assert "(p0_n0_v > 16'sh0040 ? 34'sh000000080 : 34'sd0)" in result.top_module
    assert "(p0_n1_v > 16'sh0080 ? 34'sh3ffffffc0 : 34'sd0)" in result.top_module


def test_fpga_compile_preserves_post_weight_threshold_comparators() -> None:
    network = from_nir(_post_weight_threshold_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    result = compile_network_to_fpga(
        neuron_graph,
        module_name="scnir_post_weight_threshold",
        bitstream_length=768,
    )

    payload = scnir_to_dict(result.scnir_document)
    validate_scnir_dict(payload)
    stream = next(
        stream for stream in payload["streams"] if stream["stream_id"] == "conn.input_to_lif.weight"
    )
    assert stream["transforms"][0]["position"] == "destination"
    manifest = {entry.stream_id: entry.as_dict() for entry in result.scnir_source_manifest}
    assert manifest["conn.input_to_lif.weight"]["transforms"] == stream["transforms"]
    assert "p0_n0_c0_threshold_out = (p0_n0_c0_raw > 34'sh000000033)" in result.top_module
    assert "p0_n1_c0_threshold_out = (p0_n1_c0_raw > 34'sh3ffffffe6)" in result.top_module
    assert "p0_n0_c0_threshold_out ? 34'sh000000100 : 34'sd0" in result.top_module
