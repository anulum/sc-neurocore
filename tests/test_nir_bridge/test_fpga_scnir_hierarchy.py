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

from tests.test_nir_bridge.fpga_hierarchy_graphs import (
    _multiport_multioutput_nested_graph,
    _multiport_nested_graph,
    _single_port_nested_graph,
)


def test_fpga_compile_inlines_single_port_nested_graph_weight_terms() -> None:
    network = from_nir(_single_port_nested_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    result = compile_network_to_fpga(
        neuron_graph,
        module_name="scnir_nested_inline",
        bitstream_length=896,
    )

    payload = scnir_to_dict(result.scnir_document)
    validate_scnir_dict(payload)
    weight_stream = next(
        stream
        for stream in payload["streams"]
        if stream["stream_id"] == "conn.subgraph__input_to_lif.weight"
    )
    assert weight_stream["signal_kind"] == "weight"
    assert "output wire signed [63:0] weight_0" in result.scnir_hierarchy_modules["scnir_subgraph"]
    assert (
        "assign weight_0[0 +: 16] = 16'sh0040;" in result.scnir_hierarchy_modules["scnir_subgraph"]
    )
    assert (
        "assign weight_0[16 +: 16] = 16'shff80;" in result.scnir_hierarchy_modules["scnir_subgraph"]
    )
    assert (
        "assign weight_0[32 +: 16] = 16'sh00c0;" in result.scnir_hierarchy_modules["scnir_subgraph"]
    )
    assert (
        "assign weight_0[48 +: 16] = 16'sh0020;" in result.scnir_hierarchy_modules["scnir_subgraph"]
    )
    assert "ext_input_0 * scnir_subgraph__weight_0[0 +: DATA_WIDTH]" in result.top_module
    assert "ext_input_1 * scnir_subgraph__weight_0[16 +: DATA_WIDTH]" in result.top_module
    assert "ext_input_0 * scnir_subgraph__weight_0[32 +: DATA_WIDTH]" in result.top_module
    assert "ext_input_1 * scnir_subgraph__weight_0[48 +: DATA_WIDTH]" in result.top_module


def test_fpga_compile_reports_inlined_single_port_hierarchy_metadata() -> None:
    network = from_nir(_single_port_nested_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    result = compile_network_to_fpga(
        neuron_graph,
        module_name="scnir_nested_hierarchy",
        bitstream_length=896,
    )

    payload = scnir_to_dict(result.scnir_document)
    validate_scnir_dict(payload)
    assert payload["hierarchy"] == [
        {
            "instance_id": "subgraph",
            "module_name": "scnir_subgraph",
            "ports": [
                {
                    "port_name": "weight_0",
                    "direction": "output",
                    "stream_id": "conn.subgraph__input_to_lif.weight",
                    "signal_kind": "weight",
                    "bit_width": 64,
                }
            ],
        }
    ]


def test_fpga_compile_inlines_exact_multiport_nested_graph_weight_terms() -> None:
    network = from_nir(_multiport_nested_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    result = compile_network_to_fpga(
        neuron_graph,
        module_name="scnir_multiport_nested_inline",
        bitstream_length=896,
    )

    payload = scnir_to_dict(result.scnir_document)
    validate_scnir_dict(payload)
    assert payload["hierarchy"] == [
        {
            "instance_id": "subgraph",
            "module_name": "scnir_subgraph",
            "ports": [
                {
                    "port_name": "weight_0",
                    "direction": "output",
                    "stream_id": "conn.subgraph__a_to_lif.weight",
                    "signal_kind": "weight",
                    "bit_width": 32,
                }
            ],
        }
    ]
    assert "output wire signed [31:0] weight_0" in result.scnir_hierarchy_modules["scnir_subgraph"]
    assert (
        "assign weight_0[0 +: 16] = 16'sh0080;" in result.scnir_hierarchy_modules["scnir_subgraph"]
    )
    assert (
        "assign weight_0[16 +: 16] = 16'shffc0;" in result.scnir_hierarchy_modules["scnir_subgraph"]
    )
    assert "ext_input_0 * scnir_subgraph__weight_0[0 +: DATA_WIDTH]" in result.top_module
    assert "ext_input_1 * scnir_subgraph__weight_0[16 +: DATA_WIDTH]" in result.top_module


def test_fpga_compile_inlines_exact_multiport_multioutput_nested_graph() -> None:
    network = from_nir(_multiport_multioutput_nested_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    result = compile_network_to_fpga(
        neuron_graph,
        module_name="scnir_multiport_multioutput_inline",
        bitstream_length=896,
    )

    payload = scnir_to_dict(result.scnir_document)
    validate_scnir_dict(payload)
    assert payload["hierarchy"][0]["ports"] == [
        {
            "port_name": "weight_0",
            "direction": "output",
            "stream_id": "conn.subgraph__a_to_lif_a.weight",
            "signal_kind": "weight",
            "bit_width": 16,
        },
        {
            "port_name": "weight_1",
            "direction": "output",
            "stream_id": "conn.subgraph__b_to_lif_b.weight",
            "signal_kind": "weight",
            "bit_width": 16,
        },
    ]
    assert "ext_input_0 * scnir_subgraph__weight_0" in result.top_module
    assert "ext_input_1 * scnir_subgraph__weight_1" in result.top_module
    assert "lif_a" in result.top_module
    assert "lif_b" in result.top_module


def test_fpga_compile_emits_standalone_hierarchy_boundary_module() -> None:
    network = from_nir(_multiport_multioutput_nested_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    result = compile_network_to_fpga(
        neuron_graph,
        module_name="scnir_hierarchy_boundary",
        bitstream_length=896,
    )

    assert set(result.scnir_hierarchy_modules) == {"scnir_subgraph"}
    hierarchy_module = result.scnir_hierarchy_modules["scnir_subgraph"]
    assert "module scnir_subgraph (" in hierarchy_module
    assert "output wire signed [15:0] weight_0" in hierarchy_module
    assert "output wire signed [15:0] weight_1" in hierarchy_module
    assert "assign weight_0 = 16'sh0080;" in hierarchy_module
    assert "assign weight_1 = 16'shffc0;" in hierarchy_module
    assert "// stream_id: conn.subgraph__a_to_lif_a.weight" in hierarchy_module
    assert "// stream_id: conn.subgraph__b_to_lif_b.weight" in hierarchy_module
    assert "wire signed [DATA_WIDTH - 1:0] scnir_subgraph__weight_0;" in result.top_module
    assert "wire signed [DATA_WIDTH - 1:0] scnir_subgraph__weight_1;" in result.top_module
    assert "scnir_subgraph scnir_subgraph_hierarchy_inst (" in result.top_module
    assert ".weight_0(scnir_subgraph__weight_0)" in result.top_module
    assert ".weight_1(scnir_subgraph__weight_1)" in result.top_module
    assert "ext_input_0 * scnir_subgraph__weight_0" in result.top_module
    assert "ext_input_1 * scnir_subgraph__weight_1" in result.top_module
