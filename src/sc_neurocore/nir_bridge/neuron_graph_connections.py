# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Weighted connection path resolution

"""Resolve weight endpoints and fold pass-through metadata into connections."""

from __future__ import annotations

from typing import Any

import numpy as np

from sc_neurocore.nir_bridge.neuron_graph_contracts import DelaySteps
from sc_neurocore.nir_bridge.neuron_graph_metadata import (
    _broadcast_scale,
    _compose_delay_steps,
    _compose_scale,
    _delay_steps,
    _fit_delay_steps_to_width,
    _flatten_widths,
    _scale_vector,
    _threshold_vector,
)
from sc_neurocore.nir_bridge.neuron_graph_nodes import (
    _DELAY_NODE_NAME,
    _FLATTEN_NODE_NAME,
    _SCALE_NODE_NAME,
    _SC_NODE_TO_TYPE,
    _SC_PASSTHROUGH_NODES,
    _THRESHOLD_NODE_NAME,
    _node_logical_width,
)

ResolvedSource = tuple[
    str,
    DelaySteps,
    np.ndarray[Any, Any] | None,
    int | None,
    np.ndarray[Any, Any] | None,
]
ResolvedDestination = tuple[
    str,
    np.ndarray[Any, Any] | None,
    int | None,
    np.ndarray[Any, Any] | None,
]


def _resolve_weight_source(
    node_name: str,
    *,
    nodes: dict[str, Any],
    predecessors: dict[str, list[str]],
    accumulated_delay_steps: DelaySteps = 0,
    accumulated_scale: np.ndarray[Any, Any] | None = None,
    accumulated_threshold: np.ndarray[Any, Any] | None = None,
    required_source_width: int | None = None,
    flatten_output_width: int | None = None,
) -> ResolvedSource | None:
    """Resolve one population or input feeding a weight node."""
    node = nodes[node_name]
    class_name = type(node).__name__
    if class_name in _SC_NODE_TO_TYPE or class_name == "SCInputNode":
        node_width = _node_logical_width(node)
        if (
            required_source_width is not None
            and node_width is not None
            and node_width != required_source_width
        ):
            raise ValueError(
                f"Flatten input width {required_source_width} does not match "
                f"source {node_name!r} width {node_width}"
            )
        width = node_width if node_width is not None else required_source_width
        resolved_delay_steps = accumulated_delay_steps
        if width is not None:
            resolved_delay_steps = _fit_delay_steps_to_width(
                accumulated_delay_steps,
                width,
                f"source {node_name!r}",
            )
        return (
            node_name,
            resolved_delay_steps,
            accumulated_scale,
            flatten_output_width,
            accumulated_threshold,
        )

    if class_name not in _SC_PASSTHROUGH_NODES:
        return None

    delay_steps = accumulated_delay_steps
    if class_name == _DELAY_NODE_NAME:
        delay_steps = _compose_delay_steps(delay_steps, _delay_steps(node, node_name))
    scale = accumulated_scale
    if class_name == _SCALE_NODE_NAME:
        scale = _compose_scale(scale, _scale_vector(node, node_name))
    threshold = accumulated_threshold
    if class_name == _THRESHOLD_NODE_NAME:
        if threshold is not None:
            raise ValueError(
                "Multiple source-side Threshold nodes on one connection require explicit "
                "pre-lowering before FPGA compilation"
            )
        threshold = _threshold_vector(node, node_name)
    next_required_source_width = required_source_width
    next_flatten_output_width = flatten_output_width
    if class_name == _FLATTEN_NODE_NAME:
        input_width, output_width = _flatten_widths(node, node_name)
        if required_source_width is not None and output_width != required_source_width:
            raise ValueError(
                f"Flatten output width {output_width} does not match downstream "
                f"source width {required_source_width}"
            )
        next_required_source_width = input_width
        next_flatten_output_width = (
            output_width if flatten_output_width is None else flatten_output_width
        )

    upstream = predecessors.get(node_name, [])
    if len(upstream) != 1:
        raise ValueError(
            f"Pass-through node {node_name!r} has {len(upstream)} upstream sources; "
            "explicit pre-lowering is required before FPGA compilation"
        )
    return _resolve_weight_source(
        upstream[0],
        nodes=nodes,
        predecessors=predecessors,
        accumulated_delay_steps=delay_steps,
        accumulated_scale=scale,
        accumulated_threshold=threshold,
        required_source_width=next_required_source_width,
        flatten_output_width=next_flatten_output_width,
    )


def _resolve_weight_destination(
    node_name: str,
    *,
    nodes: dict[str, Any],
    successors: dict[str, list[str]],
    accumulated_scale: np.ndarray[Any, Any] | None = None,
    accumulated_threshold: np.ndarray[Any, Any] | None = None,
    required_destination_width: int | None = None,
    flatten_input_width: int | None = None,
) -> ResolvedDestination | None:
    """Resolve one neuron destination fed by a weight node."""
    node = nodes[node_name]
    class_name = type(node).__name__
    if class_name in _SC_NODE_TO_TYPE:
        node_width = _node_logical_width(node)
        if (
            required_destination_width is not None
            and node_width is not None
            and node_width != required_destination_width
        ):
            raise ValueError(
                f"Flatten output width {required_destination_width} does not match "
                f"destination {node_name!r} width {node_width}"
            )
        return node_name, accumulated_scale, flatten_input_width, accumulated_threshold

    if class_name == "SCOutputNode" and not successors.get(node_name):
        return None
    if class_name not in _SC_PASSTHROUGH_NODES or class_name == "SCInputNode":
        return None

    scale = accumulated_scale
    if class_name == _SCALE_NODE_NAME:
        scale = _compose_scale(scale, _scale_vector(node, node_name))
    threshold = accumulated_threshold
    if class_name == _THRESHOLD_NODE_NAME:
        if threshold is not None:
            raise ValueError(
                "Multiple post-weight Threshold nodes on one connection require explicit "
                "pre-lowering before FPGA compilation"
            )
        threshold = _threshold_vector(node, node_name)
    next_required_destination_width = required_destination_width
    next_flatten_input_width = flatten_input_width
    if class_name == _FLATTEN_NODE_NAME:
        input_width, output_width = _flatten_widths(node, node_name)
        if required_destination_width is not None and input_width != required_destination_width:
            raise ValueError(
                f"Flatten input width {input_width} does not match upstream "
                f"destination width {required_destination_width}"
            )
        next_required_destination_width = output_width
        next_flatten_input_width = (
            input_width if flatten_input_width is None else flatten_input_width
        )

    downstream = successors.get(node_name, [])
    if len(downstream) != 1:
        raise ValueError(
            f"Pass-through node {node_name!r} has {len(downstream)} downstream targets; "
            "explicit pre-lowering is required before FPGA compilation"
        )
    return _resolve_weight_destination(
        downstream[0],
        nodes=nodes,
        successors=successors,
        accumulated_scale=scale,
        accumulated_threshold=threshold,
        required_destination_width=next_required_destination_width,
        flatten_input_width=next_flatten_input_width,
    )


def _fold_connection_scales(
    weights: np.ndarray[Any, Any],
    bias: np.ndarray[Any, Any] | None,
    *,
    source_scale: np.ndarray[Any, Any] | None,
    destination_scale: np.ndarray[Any, Any] | None,
    src: str,
    dst: str,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
    """Fold adjacent scale nodes into connection weights and bias."""
    folded_weights = np.asarray(weights, dtype=np.float32).copy()
    folded_bias = None if bias is None else np.asarray(bias, dtype=np.float32).copy()

    source_values = _broadcast_scale(
        source_scale,
        folded_weights.shape[1],
        f"source-side Scale for connection {src}->{dst}",
    )
    if source_values is not None:
        folded_weights *= source_values[np.newaxis, :]

    destination_values = _broadcast_scale(
        destination_scale,
        folded_weights.shape[0],
        f"post-weight Scale for connection {src}->{dst}",
    )
    if destination_values is not None:
        folded_weights *= destination_values[:, np.newaxis]
        if folded_bias is not None:
            folded_bias *= destination_values
    return folded_weights, folded_bias
