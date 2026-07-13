# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dense NIR lowering validation contracts

"""Exercise dense operator validation through the public graph conversion path."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.nir_bridge import from_scnetwork
from sc_neurocore.nir_bridge.node_map import (
    SCAffineNode,
    SCAvgPool2dNode,
    SCConv1dNode,
    SCConv2dNode,
    SCInputNode,
    SCLIFNode,
    SCOutputNode,
    SCSumPool2dNode,
)
from sc_neurocore.nir_bridge.parser import SCNetwork


def _lif(name: str, width: int) -> SCLIFNode:
    """Build a real parsed-node LIF destination with deterministic parameters."""
    return SCLIFNode(
        name=name,
        n_neurons=width,
        tau=np.full(width, 20.0),
        r=np.ones(width),
        v_leak=np.zeros(width),
        v_threshold=np.ones(width),
        v_reset=np.zeros(width),
    )


def _weight_network(weight_node: Any, input_width: int = 4, output_width: int = 4) -> SCNetwork:
    """Place one real parsed weight operator in the public conversion pipeline."""
    nodes = {
        "input": SCInputNode("input", (input_width,)),
        "weight": weight_node,
        "lif": _lif("lif", output_width),
        "output": SCOutputNode("output", (output_width,)),
    }
    return SCNetwork(
        nodes=nodes,
        edges=[("input", "weight"), ("weight", "lif"), ("lif", "output")],
        input_nodes=["input"],
        output_nodes=["output"],
    )


def _conv1d(**overrides: Any) -> SCConv1dNode:
    """Build a valid Conv1d parsed node before applying one test override."""
    values: dict[str, Any] = {
        "name": "weight",
        "weight": np.ones((2, 1, 2), dtype=np.float32),
        "bias": np.zeros(2, dtype=np.float32),
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "input_shape": 4,
    }
    values.update(overrides)
    return SCConv1dNode(**values)


def _conv2d(**overrides: Any) -> SCConv2dNode:
    """Build a valid Conv2d parsed node before applying one test override."""
    values: dict[str, Any] = {
        "name": "weight",
        "weight": np.ones((1, 1, 2, 2), dtype=np.float32),
        "bias": np.zeros(1, dtype=np.float32),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "groups": 1,
        "input_shape": (3, 3),
        "output_shape": (1, 2, 2),
    }
    values.update(overrides)
    return SCConv2dNode(**values)


@pytest.mark.parametrize(
    ("node", "message"),
    [
        (_conv1d(weight=np.ones((2, 2), dtype=np.float32)), "weight must have shape"),
        (_conv1d(input_shape=None), "requires input_shape"),
        (_conv1d(input_shape=0), "input_shape must be positive"),
        (_conv1d(padding=cast(Any, "same")), "string padding"),
        (_conv1d(stride=0), "invalid stride"),
        (_conv1d(padding=-1), "invalid stride"),
        (_conv1d(dilation=0), "invalid stride"),
        (_conv1d(groups=0), "invalid stride"),
        (
            _conv1d(weight=np.ones((3, 1, 2), dtype=np.float32), groups=2),
            "divisible by groups",
        ),
        (_conv1d(weight=np.ones((1, 1, 3), dtype=np.float32), input_shape=1), "not positive"),
        (_conv1d(bias=np.zeros(3, dtype=np.float32)), "bias length"),
    ],
)
def test_conv1d_validation_rejects_unrepresentable_nodes(
    node: SCConv1dNode,
    message: str,
) -> None:
    """Reject malformed Conv1d metadata before emitting a connection."""
    with pytest.raises(ValueError, match=message):
        from_scnetwork(_weight_network(node))


def test_conv1d_padding_and_missing_bias_lower_exactly() -> None:
    """Preserve padded taps and materialise a zero bias when NIR omits it."""
    node = _conv1d(
        weight=np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32),
        bias=cast(Any, None),
        padding=1,
        input_shape=2,
    )
    graph = from_scnetwork(_weight_network(node, input_width=2, output_width=2))

    np.testing.assert_array_equal(graph.connections[0].weights, [[2.0, 3.0], [1.0, 2.0]])
    np.testing.assert_array_equal(graph.connections[0].bias, [0.0, 0.0])


@pytest.mark.parametrize(
    ("node", "message"),
    [
        (_conv2d(weight=np.ones((1, 1, 2), dtype=np.float32)), "weight must have shape"),
        (_conv2d(input_shape=None), "requires input_shape"),
        (_conv2d(input_shape=cast(Any, (1, 2, 3))), r"must be \(height, width\)"),
        (_conv2d(input_shape=(0, 3)), "input_shape must be positive"),
        (_conv2d(padding=cast(Any, ("same", 0))), "string padding"),
        (_conv2d(stride=(0, 1)), "invalid stride"),
        (_conv2d(padding=(-1, 0)), "invalid stride"),
        (_conv2d(dilation=(0, 1)), "invalid stride"),
        (_conv2d(groups=0), "invalid stride"),
        (
            _conv2d(weight=np.ones((3, 1, 2, 2), dtype=np.float32), groups=2),
            "divisible by groups",
        ),
        (
            _conv2d(weight=np.ones((1, 1, 4, 4), dtype=np.float32), input_shape=(1, 1)),
            "not positive",
        ),
        (_conv2d(output_shape=(1, 3, 3)), "does not match"),
        (_conv2d(bias=np.zeros(2, dtype=np.float32)), "bias length"),
    ],
)
def test_conv2d_validation_rejects_unrepresentable_nodes(
    node: SCConv2dNode,
    message: str,
) -> None:
    """Reject malformed Conv2d metadata before emitting a connection."""
    with pytest.raises(ValueError, match=message):
        from_scnetwork(_weight_network(node))


def test_conv2d_padding_and_missing_bias_lower_exactly() -> None:
    """Preserve padded spatial taps and materialise zero bias values."""
    node = _conv2d(
        weight=np.ones((1, 1, 2, 2), dtype=np.float32),
        bias=cast(Any, None),
        padding=(1, 1),
        input_shape=(1, 1),
        output_shape=(1, 2, 2),
    )
    graph = from_scnetwork(_weight_network(node, input_width=1, output_width=4))

    np.testing.assert_array_equal(graph.connections[0].weights, np.ones((4, 1)))
    np.testing.assert_array_equal(graph.connections[0].bias, np.zeros(4))


def test_conv2d_without_declared_output_shape_uses_computed_shape() -> None:
    """Lower a valid Conv2d node when NIR omits optional output metadata."""
    node = _conv2d(output_shape=None)
    graph = from_scnetwork(_weight_network(node))
    assert graph.connections[0].weights.shape == (4, 9)


def _pool_node(node_type: Callable[..., Any], **overrides: Any) -> Any:
    """Build a valid parsed pooling node with one optional invalid override."""
    values: dict[str, Any] = {
        "name": "weight",
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "padding": (0, 0),
        "input_shape": (1, 3, 3),
        "output_shape": (1, 2, 2),
    }
    values.update(overrides)
    return node_type(**values)


@pytest.mark.parametrize(
    ("node", "message"),
    [
        (_pool_node(SCSumPool2dNode, input_shape=None), "requires input/output"),
        (_pool_node(SCSumPool2dNode, input_shape=(2, 2)), "requires CHW"),
        (_pool_node(SCSumPool2dNode, input_shape=(1, 0, 3)), "invalid input shape"),
        (_pool_node(SCSumPool2dNode, output_shape=(2, 2, 2)), "invalid output shape"),
        (_pool_node(SCSumPool2dNode, kernel_size=(0, 2)), "invalid kernel"),
        (_pool_node(SCSumPool2dNode, stride=(0, 1)), "invalid kernel"),
        (_pool_node(SCSumPool2dNode, padding=(-1, 0)), "invalid kernel"),
    ],
)
def test_pool_validation_rejects_unrepresentable_nodes(node: Any, message: str) -> None:
    """Reject pooling shapes and window metadata that hardware cannot represent."""
    with pytest.raises(ValueError, match=message):
        from_scnetwork(_weight_network(node))


def test_padded_sum_and_average_pooling_lower_through_public_conversion() -> None:
    """Retain boundary padding and distinguish sum from average coefficients."""
    sum_graph = from_scnetwork(
        _weight_network(
            _pool_node(
                SCSumPool2dNode,
                padding=(1, 1),
                input_shape=(1, 1, 1),
                output_shape=(1, 2, 2),
            ),
            input_width=1,
            output_width=4,
        )
    )
    average_graph = from_scnetwork(
        _weight_network(
            _pool_node(
                SCAvgPool2dNode,
                input_shape=(1, 2, 2),
                output_shape=(1, 1, 1),
            ),
            input_width=4,
            output_width=1,
        )
    )

    np.testing.assert_array_equal(sum_graph.connections[0].weights, np.ones((4, 1)))
    np.testing.assert_array_equal(
        average_graph.connections[0].weights,
        np.full((1, 4), 0.25),
    )


def test_affine_legacy_weights_attribute_and_missing_weights_contract() -> None:
    """Accept the historic plural attribute but reject a node with neither spelling."""
    affine = SCAffineNode("weight", np.eye(2, dtype=np.float32), np.zeros(2))
    dynamic_affine = cast(Any, affine)
    dynamic_affine.weights = dynamic_affine.weight
    del dynamic_affine.weight
    graph = from_scnetwork(_weight_network(affine, input_width=2, output_width=2))
    np.testing.assert_array_equal(graph.connections[0].weights, np.eye(2))

    del dynamic_affine.weights
    with pytest.raises(ValueError, match="does not expose weights"):
        from_scnetwork(_weight_network(affine, input_width=2, output_width=2))
