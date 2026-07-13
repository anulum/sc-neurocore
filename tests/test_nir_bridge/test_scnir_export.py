# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SC-NIR conversion wiring

"""SC-NIR conversion tests for the NIR/NeuronGraph pipeline."""

from __future__ import annotations


import pytest

nir = pytest.importorskip("nir")

from sc_neurocore.ir import scnir_to_dict, validate_scnir_dict
from sc_neurocore.ir.scnir_convert import (
    SCNIRConversionConfig,
    build_scnir_from_neuron_graph,
)
from sc_neurocore.learning.online_o1 import OnlineO1Config
from sc_neurocore.nir_bridge import from_nir, from_scnetwork

from tests.test_nir_bridge.scnir_delay_graphs import (
    _build_explicit_delay_lif_graph,
    _build_heterogeneous_delay_lif_graph,
    _build_mixed_li_lif_graph,
    _build_recurrent_lif_graph,
    _neuron_graph,
)


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


def test_scnir_export_preserves_heterogeneous_explicit_nir_delay_vector() -> None:
    network = from_nir(_build_heterogeneous_delay_lif_graph(), dt=1.0)

    neuron_graph = from_scnetwork(network, dt=1.0)
    delayed_connections = [
        conn for conn in neuron_graph.connections if conn.src == "lif0" and conn.dst == "lif1"
    ]
    assert len(delayed_connections) == 1
    assert delayed_connections[0].delay_steps == (1, 2)

    document = build_scnir_from_neuron_graph(
        neuron_graph,
        config=SCNIRConversionConfig(bitstream_length=896, base_seed=73),
    )
    payload = scnir_to_dict(document)
    validate_scnir_dict(payload)

    delayed_stream = next(
        stream for stream in payload["streams"] if stream["stream_id"] == "conn.lif0_to_lif1.weight"
    )
    assert delayed_stream["delay_steps"] == [1, 2]
    assert delayed_stream["correlation_constraints"][0]["peer_stream_id"] == "pop.lif1.spike"


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


def test_scnir_export_preserves_online_learning_annotation_on_weight_stream() -> None:
    network = from_nir(_build_mixed_li_lif_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)
    annotation = OnlineO1Config(weight_bits=9, trace_bits=5).to_scnir_annotation(
        rule_id="li_to_lif_online"
    )

    document = build_scnir_from_neuron_graph(
        neuron_graph,
        config=SCNIRConversionConfig(
            bitstream_length=640,
            base_seed=61,
            online_learning={"conn.li_to_lif.weight": annotation},
        ),
    )
    payload = scnir_to_dict(document)
    validate_scnir_dict(payload)

    streams = {stream["stream_id"]: stream for stream in payload["streams"]}
    assert streams["conn.li_to_lif.weight"]["online_learning"] == annotation
    assert streams["conn.input_to_li.weight"]["online_learning"] is None
    assert streams["pop.li.state"]["online_learning"] is None
