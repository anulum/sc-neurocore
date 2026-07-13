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

from sc_neurocore.ir import scnir_to_dict, validate_scnir_dict
from sc_neurocore.ir.scnir_convert import (
    SCNIRConversionConfig,
    build_scnir_from_neuron_graph,
)
from sc_neurocore.nir_bridge import from_nir, from_scnetwork

from tests.test_nir_bridge.scnir_dense_graphs import (
    _build_avg_pool2d_lif_graph,
    _build_conv1d_lif_graph,
    _build_conv1d_without_shape_lif_graph,
    _build_conv2d_lif_graph,
    _build_conv2d_without_shape_lif_graph,
    _build_sum_pool2d_lif_graph,
    _build_sum_pool2d_without_shape_lif_graph,
)


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
    assert conv_connection.bias is not None
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


def test_neuron_graph_lowers_shape_known_conv2d_to_weight_matrix() -> None:
    network = from_nir(_build_conv2d_lif_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    conv_connection = next(
        conn for conn in neuron_graph.connections if conn.src == "input" and conn.dst == "lif"
    )

    assert conv_connection.weights.shape == (4, 9)
    np.testing.assert_allclose(
        conv_connection.weights,
        [
            [1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0],
        ],
    )
    assert conv_connection.bias is not None
    np.testing.assert_allclose(conv_connection.bias, np.full(4, 0.5))


def test_neuron_graph_rejects_conv2d_without_input_shape() -> None:
    network = from_nir(_build_conv2d_without_shape_lif_graph(), dt=1.0)

    with pytest.raises(ValueError, match="Conv2d.*input_shape"):
        from_scnetwork(network, dt=1.0)


def test_neuron_graph_lowers_sum_pool2d_to_weight_matrix() -> None:
    network = from_nir(_build_sum_pool2d_lif_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    pool_connection = next(
        conn for conn in neuron_graph.connections if conn.src == "input" and conn.dst == "lif"
    )

    assert pool_connection.weights.shape == (4, 9)
    np.testing.assert_allclose(
        pool_connection.weights,
        [
            [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        ],
    )
    assert pool_connection.bias is None


def test_neuron_graph_lowers_avg_pool2d_to_weight_matrix() -> None:
    network = from_nir(_build_avg_pool2d_lif_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    pool_connection = next(
        conn for conn in neuron_graph.connections if conn.src == "input" and conn.dst == "lif"
    )

    assert pool_connection.weights.shape == (4, 9)
    np.testing.assert_allclose(pool_connection.weights.sum(axis=1), np.ones(4))
    np.testing.assert_allclose(pool_connection.weights[0, [0, 1, 3, 4]], np.full(4, 0.25))


def test_neuron_graph_rejects_sum_pool2d_without_shape_metadata() -> None:
    network = from_nir(_build_sum_pool2d_without_shape_lif_graph(), dt=1.0)

    with pytest.raises(ValueError, match="SumPool2d.*shape metadata"):
        from_scnetwork(network, dt=1.0)
