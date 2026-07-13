# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR connection metadata validation contracts

"""Exercise delay, scale, threshold, and flatten validation through conversion."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.nir_bridge import from_scnetwork
from sc_neurocore.nir_bridge.node_map import (
    SCAffineNode,
    SCDelayNode,
    SCFlattenNode,
    SCInputNode,
    SCLIFNode,
    SCOutputNode,
    SCScaleNode,
    SCThresholdNode,
)
from sc_neurocore.nir_bridge.parser import SCNetwork


def _lif(name: str = "lif", width: int = 2) -> SCLIFNode:
    """Build a deterministic real parsed-node LIF destination."""
    return SCLIFNode(
        name,
        width,
        tau=np.full(width, 20.0),
        r=np.ones(width),
        v_leak=np.zeros(width),
        v_threshold=np.ones(width),
        v_reset=np.zeros(width),
    )


def _metadata_network(
    *,
    source_path: list[Any] | None = None,
    destination_path: list[Any] | None = None,
    input_shape: tuple[int, ...] = (2,),
    weights: np.ndarray[Any, Any] | None = None,
) -> SCNetwork:
    """Place pass-through metadata around one real affine connection."""
    source_path = [] if source_path is None else source_path
    destination_path = [] if destination_path is None else destination_path
    weight_values = np.eye(2, dtype=np.float32) if weights is None else weights
    affine = SCAffineNode("affine", weight_values, np.zeros(weight_values.shape[0]))
    lif = _lif(width=weight_values.shape[0])
    input_node = SCInputNode("input", input_shape)
    output_node = SCOutputNode("output", (weight_values.shape[0],))
    ordered = [input_node, *source_path, affine, *destination_path, lif, output_node]
    return SCNetwork(
        nodes={node.name: node for node in ordered},
        edges=[(left.name, right.name) for left, right in zip(ordered, ordered[1:])],
        input_nodes=["input"],
        output_nodes=["output"],
    )


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        (cast(Any, None), "does not expose delay_steps"),
        (np.array([], dtype=np.int64), "at least one delay"),
        (np.array([-1], dtype=np.int64), "negative delay_steps"),
        (np.array([1, 2, 3], dtype=np.int64), "does not match source width"),
    ],
)
def test_delay_metadata_rejects_invalid_public_networks(steps: Any, message: str) -> None:
    """Reject missing, empty, negative, or width-incompatible delay metadata."""
    delay = SCDelayNode("delay", np.array([0], dtype=np.int64))
    delay.delay_steps = steps
    with pytest.raises(ValueError, match=message):
        from_scnetwork(_metadata_network(source_path=[delay]))


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1], [2, 3], (3, 4)),
        ([1, 2], [3], (4, 5)),
        ([1, 2], [3, 4], (4, 6)),
        ([1, 2], [2, 1], 3),
        ([1, 1], [2, 2], 3),
    ],
)
def test_adjacent_delay_vectors_compose_through_public_conversion(
    left: list[int],
    right: list[int],
    expected: int | tuple[int, ...],
) -> None:
    """Compose scalar and per-channel delays without losing heterogeneity."""
    graph = from_scnetwork(
        _metadata_network(
            source_path=[
                SCDelayNode("left", np.asarray(left, dtype=np.int64)),
                SCDelayNode("right", np.asarray(right, dtype=np.int64)),
            ]
        )
    )
    assert graph.connections[0].delay_steps == expected


def test_adjacent_delay_vectors_reject_incompatible_lengths() -> None:
    """Reject delay vectors that cannot broadcast to one connection width."""
    network = _metadata_network(
        source_path=[
            SCDelayNode("left", np.array([1, 2], dtype=np.int64)),
            SCDelayNode("right", np.array([1, 2, 3], dtype=np.int64)),
        ],
        input_shape=(3,),
        weights=np.ones((2, 3), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="Incompatible delay vector lengths"):
        from_scnetwork(network)


@pytest.mark.parametrize(
    ("scale", "message"),
    [
        (cast(Any, None), "does not expose scale"),
        (np.array([], dtype=np.float32), "at least one scale"),
        (np.array([np.nan], dtype=np.float32), "non-finite scale"),
    ],
)
def test_scale_metadata_rejects_invalid_public_networks(scale: Any, message: str) -> None:
    """Reject absent, empty, and non-finite scale arrays."""
    with pytest.raises(ValueError, match=message):
        from_scnetwork(_metadata_network(source_path=[SCScaleNode("scale", scale)]))


def test_adjacent_scales_broadcast_or_reject_through_public_conversion() -> None:
    """Compose compatible scales and reject incompatible adjacent vectors."""
    graph = from_scnetwork(
        _metadata_network(
            source_path=[
                SCScaleNode("scalar", np.array([2.0], dtype=np.float32)),
                SCScaleNode("vector", np.array([3.0, 4.0], dtype=np.float32)),
            ]
        )
    )
    np.testing.assert_array_equal(graph.connections[0].weights, [[6.0, 0.0], [0.0, 8.0]])

    invalid = _metadata_network(
        source_path=[
            SCScaleNode("left", np.array([1.0, 2.0], dtype=np.float32)),
            SCScaleNode("right", np.array([1.0, 2.0, 3.0], dtype=np.float32)),
        ]
    )
    with pytest.raises(ValueError, match="incompatible shapes"):
        from_scnetwork(invalid)


def test_scalar_scale_broadcasts_across_source_columns() -> None:
    """Broadcast one source scale value across every weight column."""
    graph = from_scnetwork(_metadata_network(source_path=[SCScaleNode("scale", np.array([2.0]))]))
    np.testing.assert_array_equal(graph.connections[0].weights, np.eye(2) * 2.0)


@pytest.mark.parametrize(
    ("threshold", "message"),
    [
        (cast(Any, None), "does not expose threshold"),
        (np.array([], dtype=np.float32), "at least one threshold"),
        (np.array([np.inf], dtype=np.float32), "non-finite threshold"),
    ],
)
def test_threshold_metadata_rejects_invalid_public_networks(
    threshold: Any,
    message: str,
) -> None:
    """Reject absent, empty, and non-finite threshold arrays."""
    with pytest.raises(ValueError, match=message):
        from_scnetwork(_metadata_network(source_path=[SCThresholdNode("threshold", threshold)]))


@pytest.mark.parametrize("position", ["source", "destination"])
def test_multiple_thresholds_on_one_connection_are_rejected(position: str) -> None:
    """Reject ambiguous repeated threshold operations on either connection side."""
    thresholds = [
        SCThresholdNode("first", np.array([0.1], dtype=np.float32)),
        SCThresholdNode("second", np.array([0.2], dtype=np.float32)),
    ]
    with pytest.raises(ValueError, match="Multiple .* Threshold nodes"):
        if position == "source":
            from_scnetwork(_metadata_network(source_path=thresholds))
        else:
            from_scnetwork(_metadata_network(destination_path=thresholds))


def test_scalar_thresholds_broadcast_to_connection_width() -> None:
    """Broadcast scalar thresholds on both sides through the public graph API."""
    graph = from_scnetwork(
        _metadata_network(
            source_path=[SCThresholdNode("source_threshold", np.array([0.25]))],
            destination_path=[SCThresholdNode("destination_threshold", np.array([0.5]))],
        )
    )
    np.testing.assert_array_equal(graph.connections[0].source_threshold, [0.25, 0.25])
    np.testing.assert_array_equal(graph.connections[0].destination_threshold, [0.5, 0.5])


@pytest.mark.parametrize(
    ("input_shape", "output_shape", "message"),
    [
        (None, (2,), "lacks input shape"),
        ((2,), None, "lacks output shape"),
        ((0,), (0,), "invalid input shape"),
        ((2,), (3,), "changes element count"),
    ],
)
def test_flatten_metadata_rejects_invalid_shapes(
    input_shape: tuple[int, ...] | None,
    output_shape: tuple[int, ...] | None,
    message: str,
) -> None:
    """Reject absent, non-positive, or element-changing flatten metadata."""
    flatten = SCFlattenNode("flatten", 0, -1, input_shape, output_shape)
    with pytest.raises(ValueError, match=message):
        from_scnetwork(_metadata_network(source_path=[flatten]))


def test_scalar_shape_flatten_preserves_one_element() -> None:
    """Treat an empty NIR shape as one scalar element on both sides."""
    graph = from_scnetwork(
        _metadata_network(
            source_path=[SCFlattenNode("flatten", 0, -1, (), ())],
            input_shape=(),
            weights=np.ones((2, 1), dtype=np.float32),
        )
    )
    assert graph.connections[0].weights.shape == (2, 1)
