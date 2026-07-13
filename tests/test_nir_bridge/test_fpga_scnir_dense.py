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

from tests.test_nir_bridge.fpga_dense_graphs import (
    _conv1d_graph,
    _conv2d_graph,
    _sum_pool2d_graph,
)


def test_fpga_compile_lowers_conv1d_to_dense_weight_stream() -> None:
    network = from_nir(_conv1d_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    result = compile_network_to_fpga(
        neuron_graph,
        module_name="scnir_conv1d_lif",
        bitstream_length=640,
    )

    payload = scnir_to_dict(result.scnir_document)
    validate_scnir_dict(payload)
    streams = {stream["stream_id"]: stream for stream in payload["streams"]}
    assert streams["conn.input_to_lif.weight"]["signal_kind"] == "weight"
    assert result.total_synapses == 24
    assert "ext_input_0 * 16'sh0100" in result.top_module
    assert "ext_input_1 * 16'sh0200" in result.top_module
    assert "ext_input_2 * 16'shff00" in result.top_module
    assert "ext_input_3 * 16'sh0080" in result.top_module
    assert "localparam integer SCNIR_STREAM_COUNT = 2;" in result.top_module


def test_fpga_compile_lowers_conv2d_to_dense_weight_stream() -> None:
    network = from_nir(_conv2d_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    result = compile_network_to_fpga(
        neuron_graph,
        module_name="scnir_conv2d_lif",
        bitstream_length=640,
    )

    payload = scnir_to_dict(result.scnir_document)
    validate_scnir_dict(payload)
    streams = {stream["stream_id"]: stream for stream in payload["streams"]}
    assert streams["conn.input_to_lif.weight"]["signal_kind"] == "weight"
    assert result.total_synapses == 36
    assert "ext_input_0 * 16'sh0100" in result.top_module
    assert "ext_input_1 * 16'sh0200" in result.top_module
    assert "ext_input_4 * 16'sh0400" in result.top_module
    assert "36'sh000000080" in result.top_module
    assert "localparam integer SCNIR_STREAM_COUNT = 2;" in result.top_module


def test_fpga_compile_lowers_sum_pool2d_to_dense_weight_stream() -> None:
    network = from_nir(_sum_pool2d_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    result = compile_network_to_fpga(
        neuron_graph,
        module_name="scnir_sum_pool2d_lif",
        bitstream_length=640,
    )

    payload = scnir_to_dict(result.scnir_document)
    validate_scnir_dict(payload)
    streams = {stream["stream_id"]: stream for stream in payload["streams"]}
    assert streams["conn.input_to_lif.weight"]["signal_kind"] == "weight"
    assert result.total_synapses == 36
    assert "ext_input_0 * 16'sh0100" in result.top_module
    assert "ext_input_4 * 16'sh0100" in result.top_module
    assert "ext_input_8 * 16'sh0100" in result.top_module
    assert "localparam integer SCNIR_STREAM_COUNT = 2;" in result.top_module
