# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SC-NIR FPGA integration

"""SC-NIR metadata integration tests for FPGA compilation artefacts."""

from __future__ import annotations

import numpy as np
import pytest

nir = pytest.importorskip("nir")

from sc_neurocore.ir import scnir_to_dict, validate_scnir_dict
from sc_neurocore.nir_bridge import compile_network_to_fpga, from_nir, from_scnetwork


def _graph() -> object:
    return nir.NIRGraph(
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


def _recurrent_graph() -> object:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.eye(2, dtype=np.float32),
                bias=np.zeros(2, dtype=np.float32),
            ),
            "lif": nir.LIF(
                tau=np.full(2, 20.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "rec": nir.Linear(weight=np.array([[0.25, 0.0], [0.0, 0.125]], dtype=np.float32)),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[
            ("input", "aff"),
            ("aff", "lif"),
            ("lif", "rec"),
            ("rec", "lif"),
            ("lif", "output"),
        ],
    )


def _explicit_delay_graph() -> object:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([1])}),
            "aff": nir.Affine(
                weight=np.ones((1, 1), dtype=np.float32),
                bias=np.zeros(1, dtype=np.float32),
            ),
            "lif0": nir.LIF(
                tau=np.full(1, 20.0),
                r=np.ones(1),
                v_leak=np.zeros(1),
                v_threshold=np.ones(1),
            ),
            "delay": nir.Delay(delay=np.array([2.0])),
            "readout": nir.Linear(weight=np.array([[0.25]], dtype=np.float32)),
            "lif1": nir.LIF(
                tau=np.full(1, 20.0),
                r=np.ones(1),
                v_leak=np.zeros(1),
                v_threshold=np.ones(1),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[
            ("input", "aff"),
            ("aff", "lif0"),
            ("lif0", "delay"),
            ("delay", "readout"),
            ("readout", "lif1"),
            ("lif1", "output"),
        ],
    )


def _explicit_analogue_delay_graph() -> object:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([1])}),
            "aff": nir.Affine(
                weight=np.ones((1, 1), dtype=np.float32),
                bias=np.zeros(1, dtype=np.float32),
            ),
            "li": nir.LI(
                tau=np.full(1, 15.0),
                r=np.ones(1),
                v_leak=np.zeros(1),
            ),
            "delay": nir.Delay(delay=np.array([2.0])),
            "readout": nir.Linear(weight=np.array([[0.5]], dtype=np.float32)),
            "lif": nir.LIF(
                tau=np.full(1, 20.0),
                r=np.ones(1),
                v_leak=np.zeros(1),
                v_threshold=np.ones(1),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[
            ("input", "aff"),
            ("aff", "li"),
            ("li", "delay"),
            ("delay", "readout"),
            ("readout", "lif"),
            ("lif", "output"),
        ],
    )


def _integrator_graph() -> object:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.eye(2, dtype=np.float32),
                bias=np.zeros(2, dtype=np.float32),
            ),
            "i": nir.I(r=np.ones(2)),
            "readout": nir.Linear(weight=np.array([[0.5, -0.25]], dtype=np.float32)),
            "lif": nir.LIF(
                tau=np.full(1, 20.0),
                r=np.ones(1),
                v_leak=np.zeros(1),
                v_threshold=np.ones(1),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[
            ("input", "aff"),
            ("aff", "i"),
            ("i", "readout"),
            ("readout", "lif"),
            ("lif", "output"),
        ],
    )


def _source_scale_graph() -> object:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.eye(2, dtype=np.float32),
                bias=np.zeros(2, dtype=np.float32),
            ),
            "li": nir.LI(
                tau=np.full(2, 15.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
            ),
            "scale": nir.Scale(scale=np.array([2.0, 0.5], dtype=np.float32)),
            "readout": nir.Linear(weight=np.array([[0.25, -0.25]], dtype=np.float32)),
            "lif": nir.LIF(
                tau=np.full(1, 20.0),
                r=np.ones(1),
                v_leak=np.zeros(1),
                v_threshold=np.ones(1),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[
            ("input", "aff"),
            ("aff", "li"),
            ("li", "scale"),
            ("scale", "readout"),
            ("readout", "lif"),
            ("lif", "output"),
        ],
    )


def _post_weight_scale_graph() -> object:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.array([[0.25, -0.5], [0.125, 0.25]], dtype=np.float32),
                bias=np.array([0.1, -0.2], dtype=np.float32),
            ),
            "scale": nir.Scale(scale=np.array([2.0, 0.5], dtype=np.float32)),
            "lif": nir.LIF(
                tau=np.full(2, 20.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[("input", "aff"), ("aff", "scale"), ("scale", "lif"), ("lif", "output")],
    )


def _flattened_input_graph() -> object:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2, 2])}),
            "flatten": nir.Flatten(input_type={"input": np.array([2, 2])}, start_dim=0),
            "aff": nir.Affine(
                weight=np.array(
                    [[0.25, -0.5, 0.125, 0.75], [-0.25, 0.5, -0.125, 0.25]],
                    dtype=np.float32,
                ),
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
        edges=[("input", "flatten"), ("flatten", "aff"), ("aff", "lif"), ("lif", "output")],
    )


def _source_threshold_graph() -> object:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.eye(2, dtype=np.float32),
                bias=np.zeros(2, dtype=np.float32),
            ),
            "li": nir.LI(
                tau=np.full(2, 15.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
            ),
            "threshold": nir.Threshold(threshold=np.array([0.25, 0.5], dtype=np.float32)),
            "readout": nir.Linear(weight=np.array([[0.5, -0.25]], dtype=np.float32)),
            "lif": nir.LIF(
                tau=np.full(1, 20.0),
                r=np.ones(1),
                v_leak=np.zeros(1),
                v_threshold=np.ones(1),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[
            ("input", "aff"),
            ("aff", "li"),
            ("li", "threshold"),
            ("threshold", "readout"),
            ("readout", "lif"),
            ("lif", "output"),
        ],
    )


def _post_weight_threshold_graph() -> object:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.array([[0.25, -0.5], [0.125, 0.25]], dtype=np.float32),
                bias=np.array([0.1, -0.2], dtype=np.float32),
            ),
            "threshold": nir.Threshold(threshold=np.array([0.2, -0.1], dtype=np.float32)),
            "lif": nir.LIF(
                tau=np.full(2, 20.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[("input", "aff"), ("aff", "threshold"), ("threshold", "lif"), ("lif", "output")],
    )


def _mixed_analogue_spiking_graph() -> object:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.eye(2, dtype=np.float32),
                bias=np.zeros(2, dtype=np.float32),
            ),
            "li": nir.LI(
                tau=np.full(2, 15.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
            ),
            "readout": nir.Linear(weight=np.array([[0.5, -0.25]], dtype=np.float32)),
            "lif": nir.LIF(
                tau=np.full(1, 20.0),
                r=np.ones(1),
                v_leak=np.zeros(1),
                v_threshold=np.ones(1),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[
            ("input", "aff"),
            ("aff", "li"),
            ("li", "readout"),
            ("readout", "lif"),
            ("lif", "output"),
        ],
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
        stream
        for stream in payload["streams"]
        if stream["stream_id"] == "conn.input_to_lif.weight"
    )
    assert stream["transforms"][0]["position"] == "destination"
    manifest = {entry.stream_id: entry.as_dict() for entry in result.scnir_source_manifest}
    assert manifest["conn.input_to_lif.weight"]["transforms"] == stream["transforms"]
    assert "p0_n0_c0_threshold_out = (p0_n0_c0_raw > 34'sh000000033)" in result.top_module
    assert "p0_n1_c0_threshold_out = (p0_n1_c0_raw > 34'sh3ffffffe6)" in result.top_module
    assert "p0_n0_c0_threshold_out ? 34'sh000000100 : 34'sd0" in result.top_module


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
        compile_network_to_fpga(neuron_graph, source_kind="halton")
