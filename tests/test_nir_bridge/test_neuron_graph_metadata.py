# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SC-NIR conversion wiring

"""SC-NIR conversion tests for the NIR/NeuronGraph pipeline."""

from __future__ import annotations


import numpy as np
import pytest

nir = pytest.importorskip("nir")

from sc_neurocore.ir import scnir_to_dict
from sc_neurocore.ir.scnir_convert import (
    SCNIRConversionConfig,
    build_scnir_from_neuron_graph,
)
from sc_neurocore.nir_bridge import from_nir, from_scnetwork

from tests.test_nir_bridge.scnir_metadata_graphs import (
    _build_flattened_input_lif_graph,
    _build_incompatible_flattened_input_lif_graph,
    _build_incompatible_post_weight_flatten_lif_graph,
    _build_post_weight_flatten_lif_graph,
    _build_post_weight_scale_lif_graph,
    _build_post_weight_threshold_lif_graph,
    _build_source_scale_li_lif_graph,
    _build_source_threshold_li_lif_graph,
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
    assert scaled_connection.bias is not None
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
    assert flattened_connection.bias is not None
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

    assert thresholded_connection.source_threshold is not None
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
    assert thresholded_connection.destination_threshold is not None
    np.testing.assert_allclose(thresholded_connection.destination_threshold, [0.2, -0.1])

    payload = scnir_to_dict(
        build_scnir_from_neuron_graph(
            neuron_graph,
            config=SCNIRConversionConfig(bitstream_length=512, base_seed=97),
        )
    )
    thresholded_stream = next(
        stream for stream in payload["streams"] if stream["stream_id"] == "conn.input_to_lif.weight"
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
