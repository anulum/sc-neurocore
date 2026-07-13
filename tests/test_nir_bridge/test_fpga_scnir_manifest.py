# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SC-NIR FPGA integration

"""SC-NIR metadata integration tests for FPGA compilation artefacts."""

from __future__ import annotations

from typing import Any, cast

import pytest

nir = pytest.importorskip("nir")

from sc_neurocore.ir import scnir_to_dict, validate_scnir_dict
from sc_neurocore.learning.online_o1 import OnlineO1Config
from sc_neurocore.nir_bridge import compile_network_to_fpga, from_nir, from_scnetwork

from tests.test_nir_bridge.fpga_delay_graphs import (
    _graph,
)
from tests.test_nir_bridge.fpga_dense_graphs import (
    _mixed_analogue_spiking_graph,
)


def test_fpga_compile_carries_mixed_signal_kinds_into_manifest() -> None:
    network = from_nir(_mixed_analogue_spiking_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    result = compile_network_to_fpga(
        neuron_graph,
        module_name="scnir_mixed_signal",
        bitstream_length=640,
    )

    payload = scnir_to_dict(result.scnir_document)
    validate_scnir_dict(payload)
    streams = {stream["stream_id"]: stream for stream in payload["streams"]}
    assert streams["pop.li.state"]["signal_kind"] == "analogue_state"
    assert streams["pop.lif.spike"]["signal_kind"] == "spike"
    assert streams["conn.li_to_lif.weight"]["signal_kind"] == "weight"
    assert "p0_n0_v * 16'sh0080" in result.top_module
    assert "p0_n1_v * 16'shffc0" in result.top_module

    manifest = {entry.stream_id: entry for entry in result.scnir_source_manifest}
    assert manifest["pop.li.state"].signal_kind == "analogue_state"
    assert manifest["pop.lif.spike"].signal_kind == "spike"
    assert manifest["conn.li_to_lif.weight"].signal_kind == "weight"


def test_fpga_compile_carries_online_learning_metadata_into_manifest() -> None:
    network = from_nir(_mixed_analogue_spiking_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)
    annotation = OnlineO1Config(
        weight_bits=9,
        trace_bits=5,
        reward_bits=4,
        learning_shift=3,
        trace_decay_shift=2,
    ).to_scnir_annotation(rule_id="li_to_lif_online")

    result = compile_network_to_fpga(
        neuron_graph,
        module_name="scnir_online_learning_manifest",
        bitstream_length=640,
        online_learning={"conn.li_to_lif.weight": annotation},
    )

    payload = scnir_to_dict(result.scnir_document)
    validate_scnir_dict(payload)
    streams = {stream["stream_id"]: stream for stream in payload["streams"]}
    assert streams["conn.li_to_lif.weight"]["online_learning"] == annotation
    assert streams["pop.li.state"]["online_learning"] is None

    manifest = {entry.stream_id: entry.as_dict() for entry in result.scnir_source_manifest}
    assert manifest["conn.li_to_lif.weight"]["online_learning"] == annotation
    assert manifest["pop.li.state"]["online_learning"] is None


def test_fpga_compile_materialises_scnir_lfsr_source_modules() -> None:
    network = from_nir(_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    result = compile_network_to_fpga(
        neuron_graph,
        module_name="scnir_sources",
        bitstream_length=1024,
        base_seed=0x1234,
    )

    assert "localparam integer SCNIR_SOURCE_MODULE_COUNT = 2;" in result.top_module
    assert len(result.scnir_source_manifest) == len(result.scnir_document.streams) == 2
    assert set(result.scnir_source_modules) == {
        entry.module_name for entry in result.scnir_source_manifest
    }

    first = result.scnir_source_manifest[0]
    assert first.stream_id == "pop.lif.spike"
    assert first.source_kind == "lfsr16"
    assert first.seed == 0x1234
    assert first.bitstream_length == 1024
    assert first.lfsr_polynomial == "x^16 + x^14 + x^13 + x^11 + 1"
    assert first.tap_mask == 0xB400

    first_module = result.scnir_source_modules[first.module_name]
    assert f"module {first.module_name}" in first_module
    assert "localparam [15:0] SEED = 16'h1234;" in first_module
    assert "output wire bit_out" in first_module


def test_fpga_compile_materialises_selected_sobol_sources() -> None:
    network = from_nir(_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    result = compile_network_to_fpga(
        neuron_graph,
        module_name="scnir_sobol_sources",
        bitstream_length=512,
        source_kind="sobol",
        base_seed=0x0042,
    )

    assert len(result.scnir_source_manifest) == 2
    first = result.scnir_source_manifest[0]
    assert first.source_kind == "sobol16"
    assert first.seed == 0x0042
    assert first.sobol_dimension == 1

    first_module = result.scnir_source_modules[first.module_name]
    assert f"module {first.module_name}" in first_module
    assert "localparam [15:0] SEED = 16'h0042;" in first_module
    assert "output reg [15:0] value" in first_module


def test_fpga_compile_rejects_invalid_scnir_bitstream_length() -> None:
    network = from_nir(_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    with pytest.raises(ValueError, match="bitstream_length"):
        compile_network_to_fpga(neuron_graph, bitstream_length=0)


def test_fpga_compile_rejects_unmaterialisable_scnir_source_kind() -> None:
    network = from_nir(_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    with pytest.raises(ValueError, match="source_kind"):
        compile_network_to_fpga(neuron_graph, source_kind=cast(Any, "halton"))
