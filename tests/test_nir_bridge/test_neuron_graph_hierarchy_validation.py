# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Nested hardware graph validation contracts

"""Exercise nested graph failures and identifier normalisation through conversion."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.nir_bridge import from_scnetwork
from sc_neurocore.nir_bridge.node_map import (
    SCAffineNode,
    SCInputNode,
    SCLIFNode,
    SCOutputNode,
)
from sc_neurocore.nir_bridge.parser import (
    SCMultiPortSubgraphNode,
    SCNetwork,
    SCSubgraphNode,
)


def _lif(name: str = "lif") -> SCLIFNode:
    """Build a deterministic one-neuron parsed node."""
    return SCLIFNode(
        name,
        1,
        tau=np.array([20.0]),
        r=np.ones(1),
        v_leak=np.zeros(1),
        v_threshold=np.ones(1),
        v_reset=np.zeros(1),
    )


def _inner_network() -> SCNetwork:
    """Build a valid single-port parsed subnetwork."""
    nodes = {
        "input": SCInputNode("input", (1,)),
        "affine": SCAffineNode("affine", np.ones((1, 1)), np.zeros(1)),
        "output": SCOutputNode("output", (1,)),
    }
    return SCNetwork(
        nodes=nodes,
        edges=[("input", "affine"), ("affine", "output")],
        input_nodes=["input"],
        output_nodes=["output"],
    )


def _parent_network(name: str, subgraph: Any) -> SCNetwork:
    """Place one parsed subgraph between a real input and LIF destination."""
    nodes = {
        "outer_input": SCInputNode("outer_input", (1,)),
        name: subgraph,
        "lif": _lif(),
        "output": SCOutputNode("output", (1,)),
    }
    return SCNetwork(
        nodes=nodes,
        edges=[("outer_input", name), (name, "lif"), ("lif", "output")],
        input_nodes=["outer_input"],
        output_nodes=["output"],
    )


def test_flattening_rejects_unknown_edge_endpoints() -> None:
    """Fail before lowering when an edge names a missing parsed node."""
    network = SCNetwork(
        nodes={"lif": _lif()},
        edges=[("missing", "lif")],
        _topo_order=["lif"],
    )
    with pytest.raises(ValueError, match="edge references unknown node"):
        from_scnetwork(network)


def test_flattening_rejects_unbroken_cycles() -> None:
    """Fail when a caller bypasses parser cycle breaking with a cyclic graph."""
    network = SCNetwork(
        nodes={"left": _lif("left"), "right": _lif("right")},
        edges=[("left", "right"), ("right", "left")],
        _topo_order=["left", "right"],
    )
    with pytest.raises(ValueError, match="contains a cycle"):
        from_scnetwork(network)


@pytest.mark.parametrize(
    ("name", "module_name"),
    [("!!!", "scnir_instance"), ("123", "scnir_n_123")],
)
def test_nested_instance_names_are_normalised_for_hdl(name: str, module_name: str) -> None:
    """Normalise empty and leading-numeric identifiers while retaining instance IDs."""
    graph = from_scnetwork(_parent_network(name, SCSubgraphNode(name, _inner_network())))
    assert graph.hierarchy[0].instance_id == name
    assert graph.hierarchy[0].module_name == module_name


def test_nested_node_requires_a_parsed_network() -> None:
    """Reject a corrupted subgraph wrapper that has lost its parsed network."""
    subgraph = SCSubgraphNode("block", _inner_network())
    cast(Any, subgraph).network = None
    with pytest.raises(ValueError, match="does not expose a parsed network"):
        from_scnetwork(_parent_network("block", subgraph))


def test_multiport_nested_node_requires_input_and_output_boundaries() -> None:
    """Reject a corrupted multi-port wrapper with an empty boundary set."""
    inner = _inner_network()
    subgraph = SCMultiPortSubgraphNode("block", inner)
    inner.input_nodes = []
    with pytest.raises(ValueError, match="at least one input and one output"):
        from_scnetwork(_parent_network("block", subgraph))


def test_single_port_nested_node_rejects_multiple_boundaries() -> None:
    """Reject a single-port wrapper whose network later gains a second input."""
    inner = _inner_network()
    subgraph = SCSubgraphNode("block", inner)
    inner.input_nodes = ["input", "second"]
    with pytest.raises(ValueError, match="exactly one input and one output"):
        from_scnetwork(_parent_network("block", subgraph))


def test_nested_namespace_collision_is_rejected() -> None:
    """Reject an outer node that already occupies a nested namespace."""
    subgraph = SCSubgraphNode("block", _inner_network())
    network = _parent_network("block", subgraph)
    network.nodes["block__input"] = SCInputNode("block__input", (1,))
    network._topo_order = ["outer_input", "block__input", "block", "lif", "output"]
    with pytest.raises(ValueError, match="would collide with existing nodes"):
        from_scnetwork(network)
