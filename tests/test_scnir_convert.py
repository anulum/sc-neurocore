# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SC-NIR conversion wiring

"""SC-NIR conversion tests for the NIR/NeuronGraph pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

nir = pytest.importorskip("nir")

from sc_neurocore.cli import main
from sc_neurocore.ir import scnir_to_dict, validate_scnir_dict
from sc_neurocore.ir.scnir_convert import (
    SCNIRConversionConfig,
    build_scnir_from_neuron_graph,
    export_scnir_from_nir,
)
from sc_neurocore.nir_bridge import from_nir, from_scnetwork


def _build_small_lif_graph() -> object:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.array([[0.5, -0.25], [0.125, 0.75]], dtype=np.float32),
                bias=np.array([0.0, 0.1], dtype=np.float32),
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


def _build_recurrent_lif_graph() -> object:
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
            "rec": nir.Linear(weight=np.array([[0.125, 0.0], [0.0, -0.25]], dtype=np.float32)),
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


def _build_explicit_delay_lif_graph() -> object:
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


def _build_heterogeneous_delay_lif_graph() -> object:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.eye(2, dtype=np.float32),
                bias=np.zeros(2, dtype=np.float32),
            ),
            "lif0": nir.LIF(
                tau=np.full(2, 20.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "delay": nir.Delay(delay=np.array([1.0, 2.0])),
            "readout": nir.Linear(weight=np.array([[0.25, -0.125]], dtype=np.float32)),
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


def _build_mixed_li_lif_graph() -> object:
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


def _build_source_scale_li_lif_graph() -> object:
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


def _build_post_weight_scale_lif_graph() -> object:
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


def _build_flattened_input_lif_graph() -> object:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2, 2])}),
            "flatten": nir.Flatten(input_type={"input": np.array([2, 2])}, start_dim=0),
            "aff": nir.Affine(
                weight=np.array(
                    [[0.25, -0.5, 0.125, 0.75], [-0.25, 0.5, -0.125, 0.25]],
                    dtype=np.float32,
                ),
                bias=np.array([0.1, -0.2], dtype=np.float32),
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


def _build_incompatible_flattened_input_lif_graph() -> object:
    graph = _build_flattened_input_lif_graph()
    graph.nodes["aff"] = nir.Affine(
        weight=np.array([[0.25, -0.5, 0.125]], dtype=np.float32),
        bias=np.array([0.1], dtype=np.float32),
    )
    graph.nodes["lif"] = nir.LIF(
        tau=np.full(1, 20.0),
        r=np.ones(1),
        v_leak=np.zeros(1),
        v_threshold=np.ones(1),
    )
    graph.nodes["output"] = nir.Output(output_type={"output": np.array([1])})
    return graph


def _build_post_weight_flatten_lif_graph() -> object:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.array(
                    [[0.25, -0.5], [0.125, 0.75], [-0.25, 0.5], [0.375, -0.125]],
                    dtype=np.float32,
                ),
                bias=np.array([0.1, -0.2, 0.0, 0.05], dtype=np.float32),
            ),
            "flatten": nir.Flatten(input_type={"input": np.array([4])}, start_dim=0),
            "lif": nir.LIF(
                tau=np.full(4, 20.0),
                r=np.ones(4),
                v_leak=np.zeros(4),
                v_threshold=np.ones(4),
            ),
            "output": nir.Output(output_type={"output": np.array([4])}),
        },
        edges=[("input", "aff"), ("aff", "flatten"), ("flatten", "lif"), ("lif", "output")],
    )


def _build_incompatible_post_weight_flatten_lif_graph() -> object:
    graph = _build_post_weight_flatten_lif_graph()
    graph.nodes["lif"] = nir.LIF(
        tau=np.full(3, 20.0),
        r=np.ones(3),
        v_leak=np.zeros(3),
        v_threshold=np.ones(3),
    )
    graph.nodes["output"] = nir.Output(output_type={"output": np.array([3])})
    return graph


def _build_source_threshold_li_lif_graph() -> object:
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


def _build_post_weight_threshold_lif_graph() -> object:
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
        edges=[
            ("input", "aff"),
            ("aff", "threshold"),
            ("threshold", "lif"),
            ("lif", "output"),
        ],
    )


def _build_conv1d_lif_graph() -> object:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([1, 4])}),
            "conv": nir.Conv1d(
                input_shape=4,
                weight=np.array(
                    [[[1.0, 2.0]], [[-1.0, 0.5]]],
                    dtype=np.float32,
                ),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=np.array([0.1, -0.2], dtype=np.float32),
            ),
            "flatten": nir.Flatten(input_type={"input": np.array([2, 3])}, start_dim=0),
            "lif": nir.LIF(
                tau=np.full(6, 20.0),
                r=np.ones(6),
                v_leak=np.zeros(6),
                v_threshold=np.ones(6),
            ),
            "output": nir.Output(output_type={"output": np.array([6])}),
        },
        edges=[("input", "conv"), ("conv", "flatten"), ("flatten", "lif"), ("lif", "output")],
    )


def _build_conv1d_without_shape_lif_graph() -> object:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([1, 4])}),
            "conv": nir.Conv1d(
                input_shape=None,
                weight=np.array([[[1.0, 2.0]]], dtype=np.float32),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=np.zeros(1, dtype=np.float32),
            ),
            "lif": nir.LIF(
                tau=np.full(3, 20.0),
                r=np.ones(3),
                v_leak=np.zeros(3),
                v_threshold=np.ones(3),
            ),
            "output": nir.Output(output_type={"output": np.array([3])}),
        },
        edges=[("input", "conv"), ("conv", "lif"), ("lif", "output")],
        type_check=False,
    )


def _neuron_graph() -> object:
    network = from_nir(_build_small_lif_graph(), dt=1.0)
    return from_scnetwork(network, dt=1.0)


def test_scnir_export_from_neuron_graph_records_population_and_weight_streams() -> None:
    config = SCNIRConversionConfig(
        bitstream_length=2048,
        data_width=18,
        fraction=10,
        accumulator_bits=40,
        base_seed=101,
        max_abs_correlation=0.02,
    )

    document = build_scnir_from_neuron_graph(_neuron_graph(), config=config)
    payload = scnir_to_dict(document)
    validate_scnir_dict(payload)

    stream_ids = {stream["stream_id"] for stream in payload["streams"]}
    assert stream_ids == {"pop.lif.spike", "conn.input_to_lif.weight"}
    assert {stream["bitstream_length"] for stream in payload["streams"]} == {2048}

    pop_stream = next(
        stream for stream in payload["streams"] if stream["stream_id"] == "pop.lif.spike"
    )
    weight_stream = next(
        stream for stream in payload["streams"] if stream["stream_id"] == "conn.input_to_lif.weight"
    )
    assert pop_stream["encoding"] == "unipolar"
    assert weight_stream["encoding"] == "bipolar"
    assert weight_stream["precision"] == {
        "signed": True,
        "total_bits": 18,
        "fractional_bits": 10,
        "accumulator_bits": 40,
        "rounding": "nearest_even",
        "overflow": "saturate",
    }
    assert pop_stream["source"]["seed"] != weight_stream["source"]["seed"]
    assert weight_stream["correlation_constraints"] == [
        {
            "peer_stream_id": "pop.lif.spike",
            "policy": "max_correlation",
            "max_abs_correlation": 0.02,
            "seed_domain": "scnir-default",
        }
    ]


def test_scnir_conversion_rejects_invalid_precision_config() -> None:
    with pytest.raises(ValueError, match="fraction"):
        SCNIRConversionConfig(bitstream_length=1024, data_width=8, fraction=8)


def test_scnir_conversion_is_deterministic() -> None:
    graph = _neuron_graph()
    config = SCNIRConversionConfig(bitstream_length=512, base_seed=7)

    left = scnir_to_dict(build_scnir_from_neuron_graph(graph, config=config))
    right = scnir_to_dict(build_scnir_from_neuron_graph(graph, config=config))

    assert left == right


def test_scnir_export_preserves_delayed_recurrent_weight_stream() -> None:
    network = from_nir(_build_recurrent_lif_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    recurrent_connections = [
        conn for conn in neuron_graph.connections if conn.src == "lif" and conn.dst == "lif"
    ]
    assert len(recurrent_connections) == 1
    assert recurrent_connections[0].delay_steps == 1

    document = build_scnir_from_neuron_graph(
        neuron_graph,
        config=SCNIRConversionConfig(bitstream_length=768, base_seed=41),
    )
    payload = scnir_to_dict(document)
    validate_scnir_dict(payload)

    stream_ids = {stream["stream_id"] for stream in payload["streams"]}
    assert "conn.lif_to_lif.weight" in stream_ids
    assert {stream["bitstream_length"] for stream in payload["streams"]} == {768}
    recurrent_stream = next(
        stream for stream in payload["streams"] if stream["stream_id"] == "conn.lif_to_lif.weight"
    )
    assert recurrent_stream["delay_steps"] == 1


def test_scnir_export_preserves_explicit_nir_delay_weight_stream() -> None:
    network = from_nir(_build_explicit_delay_lif_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    delayed_connections = [
        conn for conn in neuron_graph.connections if conn.src == "lif0" and conn.dst == "lif1"
    ]
    assert len(delayed_connections) == 1
    assert delayed_connections[0].delay_steps == 2

    document = build_scnir_from_neuron_graph(
        neuron_graph,
        config=SCNIRConversionConfig(bitstream_length=896, base_seed=71),
    )
    payload = scnir_to_dict(document)
    validate_scnir_dict(payload)

    delayed_stream = next(
        stream for stream in payload["streams"] if stream["stream_id"] == "conn.lif0_to_lif1.weight"
    )
    assert delayed_stream["delay_steps"] == 2
    assert delayed_stream["correlation_constraints"][0]["peer_stream_id"] == "pop.lif1.spike"


def test_scnir_export_rejects_heterogeneous_explicit_nir_delay() -> None:
    network = from_nir(_build_heterogeneous_delay_lif_graph(), dt=1.0)

    with pytest.raises(ValueError, match="heterogeneous delay_steps"):
        from_scnetwork(network, dt=1.0)


def test_scnir_export_marks_mixed_analogue_state_and_spike_streams() -> None:
    network = from_nir(_build_mixed_li_lif_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    document = build_scnir_from_neuron_graph(
        neuron_graph,
        config=SCNIRConversionConfig(bitstream_length=640, base_seed=61),
    )
    payload = scnir_to_dict(document)
    validate_scnir_dict(payload)

    streams = {stream["stream_id"]: stream for stream in payload["streams"]}
    assert set(streams) == {
        "pop.li.state",
        "pop.lif.spike",
        "conn.input_to_li.weight",
        "conn.li_to_lif.weight",
    }
    assert streams["pop.li.state"]["signal_kind"] == "analogue_state"
    assert streams["pop.li.state"]["encoding"] == "bipolar"
    assert streams["pop.lif.spike"]["signal_kind"] == "spike"
    assert streams["conn.li_to_lif.weight"]["signal_kind"] == "weight"
    assert (
        streams["conn.input_to_li.weight"]["correlation_constraints"][0]["peer_stream_id"]
        == "pop.li.state"
    )


def test_neuron_graph_folds_source_side_scale_into_downstream_weights() -> None:
    network = from_nir(_build_source_scale_li_lif_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    scaled_connection = next(
        conn for conn in neuron_graph.connections if conn.src == "li" and conn.dst == "lif"
    )

    np.testing.assert_allclose(scaled_connection.weights, [[0.5, -0.125]])


def test_neuron_graph_folds_post_weight_scale_into_rows_and_bias() -> None:
    network = from_nir(_build_post_weight_scale_lif_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    scaled_connection = next(
        conn for conn in neuron_graph.connections if conn.src == "input" and conn.dst == "lif"
    )

    np.testing.assert_allclose(scaled_connection.weights, [[0.5, -1.0], [0.0625, 0.125]])
    np.testing.assert_allclose(scaled_connection.bias, [0.2, -0.1])


def test_neuron_graph_rejects_incompatible_source_side_scale_length() -> None:
    graph = _build_source_scale_li_lif_graph()
    graph.nodes["scale"] = nir.Scale(scale=np.array([1.0, 2.0, 3.0], dtype=np.float32))
    network = from_nir(graph, dt=1.0)

    with pytest.raises(ValueError, match="source-side Scale"):
        from_scnetwork(network, dt=1.0)


def test_neuron_graph_rejects_incompatible_post_weight_scale_length() -> None:
    graph = _build_post_weight_scale_lif_graph()
    graph.nodes["scale"] = nir.Scale(scale=np.array([1.0, 2.0, 3.0], dtype=np.float32))
    network = from_nir(graph, dt=1.0)

    with pytest.raises(ValueError, match="post-weight Scale"):
        from_scnetwork(network, dt=1.0)


def test_neuron_graph_preserves_source_side_flatten_into_weight_width() -> None:
    network = from_nir(_build_flattened_input_lif_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    flattened_connection = next(
        conn for conn in neuron_graph.connections if conn.src == "input" and conn.dst == "lif"
    )

    assert flattened_connection.weights.shape == (2, 4)
    np.testing.assert_allclose(
        flattened_connection.weights,
        [[0.25, -0.5, 0.125, 0.75], [-0.25, 0.5, -0.125, 0.25]],
    )


def test_neuron_graph_rejects_flatten_width_mismatched_to_weight_columns() -> None:
    network = from_nir(_build_incompatible_flattened_input_lif_graph(), dt=1.0)

    with pytest.raises(ValueError, match="Flatten.*weight input width"):
        from_scnetwork(network, dt=1.0)


def test_neuron_graph_preserves_post_weight_flatten_into_destination_width() -> None:
    network = from_nir(_build_post_weight_flatten_lif_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    flattened_connection = next(
        conn for conn in neuron_graph.connections if conn.src == "input" and conn.dst == "lif"
    )

    assert flattened_connection.weights.shape == (4, 2)
    np.testing.assert_allclose(flattened_connection.bias, [0.1, -0.2, 0.0, 0.05])


def test_neuron_graph_rejects_flatten_width_mismatched_to_destination_neurons() -> None:
    network = from_nir(_build_incompatible_post_weight_flatten_lif_graph(), dt=1.0)

    with pytest.raises(ValueError, match="Flatten.*destination"):
        from_scnetwork(network, dt=1.0)


def test_neuron_graph_preserves_source_side_threshold_metadata() -> None:
    network = from_nir(_build_source_threshold_li_lif_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    thresholded_connection = next(
        conn for conn in neuron_graph.connections if conn.src == "li" and conn.dst == "lif"
    )

    np.testing.assert_allclose(thresholded_connection.source_threshold, [0.25, 0.5])
    assert thresholded_connection.destination_threshold is None

    payload = scnir_to_dict(
        build_scnir_from_neuron_graph(
            neuron_graph,
            config=SCNIRConversionConfig(bitstream_length=512, base_seed=91),
        )
    )
    thresholded_stream = next(
        stream for stream in payload["streams"] if stream["stream_id"] == "conn.li_to_lif.weight"
    )
    assert thresholded_stream["transforms"] == [
        {
            "kind": "threshold",
            "position": "source",
            "comparison": "greater_than",
            "values": [0.25, 0.5],
        }
    ]


def test_neuron_graph_preserves_post_weight_threshold_metadata() -> None:
    network = from_nir(_build_post_weight_threshold_lif_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    thresholded_connection = next(
        conn for conn in neuron_graph.connections if conn.src == "input" and conn.dst == "lif"
    )

    assert thresholded_connection.source_threshold is None
    np.testing.assert_allclose(thresholded_connection.destination_threshold, [0.2, -0.1])

    payload = scnir_to_dict(
        build_scnir_from_neuron_graph(
            neuron_graph,
            config=SCNIRConversionConfig(bitstream_length=512, base_seed=97),
        )
    )
    thresholded_stream = next(
        stream
        for stream in payload["streams"]
        if stream["stream_id"] == "conn.input_to_lif.weight"
    )
    assert thresholded_stream["transforms"][0]["kind"] == "threshold"
    assert thresholded_stream["transforms"][0]["position"] == "destination"
    assert thresholded_stream["transforms"][0]["comparison"] == "greater_than"
    assert thresholded_stream["transforms"][0]["values"] == pytest.approx([0.2, -0.1])


def test_neuron_graph_rejects_incompatible_source_threshold_length() -> None:
    graph = _build_source_threshold_li_lif_graph()
    graph.nodes["threshold"] = nir.Threshold(threshold=np.array([0.25, 0.5, 0.75]))
    network = from_nir(graph, dt=1.0)

    with pytest.raises(ValueError, match="source-side Threshold"):
        from_scnetwork(network, dt=1.0)


def test_neuron_graph_rejects_incompatible_post_weight_threshold_length() -> None:
    graph = _build_post_weight_threshold_lif_graph()
    graph.nodes["threshold"] = nir.Threshold(threshold=np.array([0.1, 0.2, 0.3]))
    network = from_nir(graph, dt=1.0)

    with pytest.raises(ValueError, match="post-weight Threshold"):
        from_scnetwork(network, dt=1.0)


def test_neuron_graph_lowers_shape_known_conv1d_to_weight_matrix() -> None:
    network = from_nir(_build_conv1d_lif_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    conv_connection = next(
        conn for conn in neuron_graph.connections if conn.src == "input" and conn.dst == "lif"
    )

    assert conv_connection.weights.shape == (6, 4)
    np.testing.assert_allclose(
        conv_connection.weights,
        [
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 0.0, 1.0, 2.0],
            [-1.0, 0.5, 0.0, 0.0],
            [0.0, -1.0, 0.5, 0.0],
            [0.0, 0.0, -1.0, 0.5],
        ],
    )
    np.testing.assert_allclose(conv_connection.bias, [0.1, 0.1, 0.1, -0.2, -0.2, -0.2])

    payload = scnir_to_dict(
        build_scnir_from_neuron_graph(
            neuron_graph,
            config=SCNIRConversionConfig(bitstream_length=512, base_seed=103),
        )
    )
    validate_scnir_dict(payload)
    assert "conn.input_to_lif.weight" in {stream["stream_id"] for stream in payload["streams"]}


def test_neuron_graph_rejects_conv1d_without_input_shape() -> None:
    network = from_nir(_build_conv1d_without_shape_lif_graph(), dt=1.0)

    with pytest.raises(ValueError, match="Conv1d.*input_shape"):
        from_scnetwork(network, dt=1.0)


def test_scnir_export_from_nir_file_round_trips(tmp_path: Path) -> None:
    model_path = tmp_path / "model.nir"
    output_path = tmp_path / "model.scnir.json"
    nir.write(str(model_path), _build_small_lif_graph())

    document = export_scnir_from_nir(
        model_path,
        output_path=output_path,
        config=SCNIRConversionConfig(bitstream_length=1024, base_seed=19),
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == scnir_to_dict(document)


def test_scnir_export_cli_writes_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_path = tmp_path / "model.nir"
    output_path = tmp_path / "export.scnir.json"
    nir.write(str(model_path), _build_small_lif_graph())

    with mock.patch(
        "sys.argv",
        [
            "sc-neurocore",
            "scnir",
            "export",
            str(model_path),
            "--output",
            str(output_path),
            "--T",
            "1024",
        ],
    ):
        rc = main()

    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    validate_scnir_dict(payload)
    assert "SC-NIR exported" in capsys.readouterr().out
