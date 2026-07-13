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

from tests.test_nir_bridge.scnir_hierarchy_graphs import (
    _build_exact_multiport_multioutput_nested_subgraph_graph,
    _build_exact_multiport_nested_subgraph_graph,
    _build_multiport_nested_subgraph_graph,
    _build_nested_subgraph_lif_graph,
)


def test_neuron_graph_inlines_single_port_nested_nir_graph_for_hardware_lowering() -> None:
    network = from_nir(_build_nested_subgraph_lif_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    assert [pop.name for pop in neuron_graph.populations] == ["lif"]
    assert len(neuron_graph.connections) == 1
    nested_connection = neuron_graph.connections[0]
    assert nested_connection.src == "subgraph__input"
    assert nested_connection.dst == "lif"
    np.testing.assert_allclose(nested_connection.weights, np.eye(2))


def test_scnir_export_records_inlined_single_port_hierarchy_metadata() -> None:
    network = from_nir(_build_nested_subgraph_lif_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    document = build_scnir_from_neuron_graph(
        neuron_graph,
        config=SCNIRConversionConfig(
            bitstream_length=1024,
            data_width=18,
            fraction=10,
            base_seed=83,
        ),
    )
    payload = scnir_to_dict(document)
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
                    "bit_width": 72,
                }
            ],
        }
    ]


def test_neuron_graph_inlines_exact_multiport_nested_nir_graph_for_hardware_lowering() -> None:
    network = from_nir(_build_exact_multiport_nested_subgraph_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    assert [pop.name for pop in neuron_graph.populations] == ["lif"]
    assert len(neuron_graph.connections) == 1
    nested_connection = neuron_graph.connections[0]
    assert nested_connection.src == "subgraph__a"
    assert nested_connection.dst == "lif"
    np.testing.assert_allclose(nested_connection.weights, [[1.0, -1.0]])

    payload = scnir_to_dict(
        build_scnir_from_neuron_graph(
            neuron_graph,
            config=SCNIRConversionConfig(bitstream_length=1024, base_seed=89),
        )
    )
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


def test_neuron_graph_inlines_exact_multiport_multioutput_nested_graph() -> None:
    network = from_nir(_build_exact_multiport_multioutput_nested_subgraph_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    assert [pop.name for pop in neuron_graph.populations] == ["lif_a", "lif_b"]
    connections = {(conn.src, conn.dst): conn for conn in neuron_graph.connections}
    assert set(connections) == {
        ("subgraph__a", "lif_a"),
        ("subgraph__b", "lif_b"),
    }
    np.testing.assert_allclose(connections[("subgraph__a", "lif_a")].weights, [[0.5]])
    np.testing.assert_allclose(connections[("subgraph__b", "lif_b")].weights, [[-0.25]])

    payload = scnir_to_dict(
        build_scnir_from_neuron_graph(
            neuron_graph,
            config=SCNIRConversionConfig(bitstream_length=1024, base_seed=93),
        )
    )
    validate_scnir_dict(payload)
    assert payload["hierarchy"] == [
        {
            "instance_id": "subgraph",
            "module_name": "scnir_subgraph",
            "ports": [
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
            ],
        }
    ]


def test_neuron_graph_rejects_unmapped_multiport_nested_nir_graph_hardware_lowering() -> None:
    network = from_nir(_build_multiport_nested_subgraph_graph(), dt=1.0)

    with pytest.raises(ValueError, match="Multi-port nested NIRGraph.*boundary mapping"):
        from_scnetwork(network, dt=1.0)
