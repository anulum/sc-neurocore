# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Weighted connection resolution contracts

"""Exercise ambiguous connection paths through public neuron-graph conversion."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.nir_bridge import from_scnetwork
from sc_neurocore.nir_bridge.node_map import (
    SCAffineNode,
    SCFlattenNode,
    SCInputNode,
    SCLIFNode,
    SCScaleNode,
)
from sc_neurocore.nir_bridge.parser import SCNetwork


def _lif(name: str, width: int) -> SCLIFNode:
    """Build a deterministic parsed LIF node."""
    return SCLIFNode(
        name,
        width,
        tau=np.full(width, 20.0),
        r=np.ones(width),
        v_leak=np.zeros(width),
        v_threshold=np.ones(width),
        v_reset=np.zeros(width),
    )


def _affine(rows: int, columns: int) -> SCAffineNode:
    """Build a deterministic parsed affine node."""
    return SCAffineNode("affine", np.ones((rows, columns)), np.zeros(rows))


def test_source_width_must_match_upstream_flatten_input() -> None:
    """Reject a source population narrower than declared flatten metadata."""
    nodes = {
        "input": SCInputNode("input", (2,)),
        "flatten": SCFlattenNode("flatten", 0, -1, (3,), (3,)),
        "affine": _affine(2, 3),
        "lif": _lif("lif", 2),
    }
    network = SCNetwork(
        nodes=nodes,
        edges=[("input", "flatten"), ("flatten", "affine"), ("affine", "lif")],
    )
    with pytest.raises(ValueError, match="source 'input' width 2"):
        from_scnetwork(network)


def test_source_with_unknown_shape_remains_resolvable() -> None:
    """Retain an input boundary whose optional logical shape is absent."""
    input_node = SCInputNode("input", (2,))
    cast(Any, input_node).shape = None
    nodes = {"input": input_node, "affine": _affine(2, 2), "lif": _lif("lif", 2)}
    graph = from_scnetwork(SCNetwork(nodes=nodes, edges=[("input", "affine"), ("affine", "lif")]))
    assert graph.connections[0].src == "input"


def test_source_flatten_chain_must_have_consistent_intermediate_width() -> None:
    """Reject adjacent source flattens with incompatible boundary widths."""
    nodes = {
        "input": SCInputNode("input", (2,)),
        "first": SCFlattenNode("first", 0, -1, (2,), (2,)),
        "second": SCFlattenNode("second", 0, -1, (3,), (3,)),
        "affine": _affine(2, 3),
        "lif": _lif("lif", 2),
    }
    network = SCNetwork(
        nodes=nodes,
        edges=[
            ("input", "first"),
            ("first", "second"),
            ("second", "affine"),
            ("affine", "lif"),
        ],
    )
    with pytest.raises(ValueError, match="Flatten output width 2.*downstream source width 3"):
        from_scnetwork(network)


def test_source_pass_through_fan_in_is_rejected() -> None:
    """Reject a scale node that merges two sources without an explicit operator."""
    nodes = {
        "left": SCInputNode("left", (2,)),
        "right": SCInputNode("right", (2,)),
        "scale": SCScaleNode("scale", np.ones(2)),
        "affine": _affine(2, 2),
        "lif": _lif("lif", 2),
    }
    network = SCNetwork(
        nodes=nodes,
        edges=[
            ("left", "scale"),
            ("right", "scale"),
            ("scale", "affine"),
            ("affine", "lif"),
        ],
    )
    with pytest.raises(ValueError, match="has 2 upstream sources"):
        from_scnetwork(network)


def test_destination_input_or_terminal_output_is_not_a_population() -> None:
    """Ignore weight paths ending at graph boundaries instead of inventing a neuron."""
    for boundary in (SCInputNode("boundary", (2,)),):
        nodes = {"affine": _affine(2, 2), "boundary": boundary, "lif": _lif("lif", 1)}
        graph = from_scnetwork(
            SCNetwork(nodes=nodes, edges=[("affine", "boundary")], input_nodes=["boundary"])
        )
        assert graph.connections == []


def test_destination_flatten_chain_must_have_consistent_intermediate_width() -> None:
    """Reject adjacent destination flattens with incompatible boundary widths."""
    nodes = {
        "affine": _affine(2, 2),
        "first": SCFlattenNode("first", 0, -1, (2,), (2,)),
        "second": SCFlattenNode("second", 0, -1, (3,), (3,)),
        "lif": _lif("lif", 3),
    }
    network = SCNetwork(
        nodes=nodes,
        edges=[("affine", "first"), ("first", "second"), ("second", "lif")],
    )
    with pytest.raises(ValueError, match="Flatten input width 3.*upstream destination width 2"):
        from_scnetwork(network)


def test_destination_pass_through_fan_out_is_rejected() -> None:
    """Reject a scale node feeding two populations without explicit duplication."""
    nodes = {
        "affine": _affine(2, 2),
        "scale": SCScaleNode("scale", np.ones(2)),
        "left": _lif("left", 2),
        "right": _lif("right", 2),
    }
    network = SCNetwork(
        nodes=nodes,
        edges=[("affine", "scale"), ("scale", "left"), ("scale", "right")],
    )
    with pytest.raises(ValueError, match="has 2 downstream targets"):
        from_scnetwork(network)


def test_destination_scale_preserves_absent_affine_bias() -> None:
    """Scale weight rows while retaining a genuinely absent optional bias."""
    affine = _affine(2, 2)
    cast(Any, affine).bias = None
    nodes = {
        "affine": affine,
        "scale": SCScaleNode("scale", np.array([2.0, 3.0])),
        "lif": _lif("lif", 2),
    }
    graph = from_scnetwork(SCNetwork(nodes=nodes, edges=[("affine", "scale"), ("scale", "lif")]))
    assert graph.connections[0].bias is None
    np.testing.assert_array_equal(graph.connections[0].weights, [[2.0, 2.0], [3.0, 3.0]])
