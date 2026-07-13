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

from tests.test_nir_bridge.fpga_delay_graphs import (
    _explicit_analogue_delay_graph,
    _explicit_delay_graph,
    _graph,
    _heterogeneous_delay_graph,
    _recurrent_graph,
)


def test_fpga_compile_result_carries_valid_scnir_metadata() -> None:
    network = from_nir(_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    result = compile_network_to_fpga(
        neuron_graph,
        module_name="scnir_integrated",
        data_width=18,
        fraction=10,
        bitstream_length=2048,
    )

    payload = scnir_to_dict(result.scnir_document)
    validate_scnir_dict(payload)
    assert {stream["bitstream_length"] for stream in payload["streams"]} == {2048}
    assert {stream["precision"]["total_bits"] for stream in payload["streams"]} == {18}
    assert {stream["precision"]["fractional_bits"] for stream in payload["streams"]} == {10}
    assert "localparam integer SCNIR_BITSTREAM_LENGTH = 2048;" in result.top_module
    assert "localparam integer SCNIR_STREAM_COUNT = 2;" in result.top_module


def test_fpga_compile_carries_recurrent_scnir_stream_and_delay_register() -> None:
    network = from_nir(_recurrent_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    result = compile_network_to_fpga(
        neuron_graph,
        module_name="scnir_recurrent",
        bitstream_length=768,
    )

    payload = scnir_to_dict(result.scnir_document)
    validate_scnir_dict(payload)
    assert {stream["stream_id"] for stream in payload["streams"]} == {
        "pop.lif.spike",
        "conn.input_to_lif.weight",
        "conn.lif_to_lif.weight",
    }
    recurrent_stream = next(
        stream for stream in payload["streams"] if stream["stream_id"] == "conn.lif_to_lif.weight"
    )
    assert recurrent_stream["delay_steps"] == 1
    assert "localparam integer SCNIR_STREAM_COUNT = 3;" in result.top_module
    assert "reg p0_n0_spike_d1;" in result.top_module
    assert "(p0_n0_spike_d1 ? 34'sh000000040 : 34'sd0)" in result.top_module
    recurrent_manifest = next(
        entry
        for entry in result.scnir_source_manifest
        if entry.stream_id == "conn.lif_to_lif.weight"
    )
    assert recurrent_manifest.delay_steps == 1


def test_fpga_compile_carries_explicit_nir_delay_stream_and_register_chain() -> None:
    network = from_nir(_explicit_delay_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    result = compile_network_to_fpga(
        neuron_graph,
        module_name="scnir_explicit_delay",
        bitstream_length=896,
    )

    payload = scnir_to_dict(result.scnir_document)
    validate_scnir_dict(payload)
    delayed_stream = next(
        stream for stream in payload["streams"] if stream["stream_id"] == "conn.lif0_to_lif1.weight"
    )
    assert delayed_stream["delay_steps"] == 2
    delayed_manifest = next(
        entry
        for entry in result.scnir_source_manifest
        if entry.stream_id == "conn.lif0_to_lif1.weight"
    )
    assert delayed_manifest.delay_steps == 2
    assert "reg p0_n0_spike_d1;" in result.top_module
    assert "reg p0_n0_spike_d2;" in result.top_module
    assert "(p0_n0_spike_d2 ? 34'sh000000040 : 34'sd0)" in result.top_module


def test_fpga_compile_uses_delayed_voltage_for_explicit_analogue_delay() -> None:
    network = from_nir(_explicit_analogue_delay_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    result = compile_network_to_fpga(
        neuron_graph,
        module_name="scnir_explicit_analogue_delay",
        bitstream_length=896,
    )

    payload = scnir_to_dict(result.scnir_document)
    validate_scnir_dict(payload)
    delayed_stream = next(
        stream for stream in payload["streams"] if stream["stream_id"] == "conn.li_to_lif.weight"
    )
    assert delayed_stream["delay_steps"] == 2
    assert "reg signed [DATA_WIDTH - 1:0] p0_n0_v_d2;" in result.top_module
    assert "p0_n0_v_d2 * 16'sh0080" in result.top_module


def test_fpga_compile_carries_heterogeneous_delay_vector_register_taps() -> None:
    network = from_nir(_heterogeneous_delay_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    result = compile_network_to_fpga(
        neuron_graph,
        module_name="scnir_heterogeneous_delay",
        bitstream_length=896,
    )

    payload = scnir_to_dict(result.scnir_document)
    validate_scnir_dict(payload)
    delayed_stream = next(
        stream for stream in payload["streams"] if stream["stream_id"] == "conn.lif0_to_lif1.weight"
    )
    assert delayed_stream["delay_steps"] == [1, 2]
    delayed_manifest = next(
        entry
        for entry in result.scnir_source_manifest
        if entry.stream_id == "conn.lif0_to_lif1.weight"
    )
    assert delayed_manifest.delay_steps == (1, 2)
    assert "reg p0_n0_spike_d1;" in result.top_module
    assert "reg p0_n1_spike_d2;" in result.top_module
    assert "(p0_n0_spike_d1 ? 34'sh000000040 : 34'sd0)" in result.top_module
    assert "(p0_n1_spike_d2 ? 34'sh3ffffffe0 : 34'sd0)" in result.top_module
