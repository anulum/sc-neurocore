# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR node classification for hardware graph lowering

"""Classify parsed NIR nodes and extract hardware graph parameters."""

from __future__ import annotations

from typing import Any

import numpy as np

_SC_NODE_TO_TYPE: dict[str, str] = {
    "SCLIFNode": "lif",
    "SCIFNode": "if",
    "SCLINode": "li",
    "SCCubaLIFNode": "cuba_lif",
    "SCCubaLINode": "cuba_li",
    "SCIntegratorNode": "integrator",
}

_SC_WEIGHT_NODES: set[str] = {
    "SCAffineNode",
    "SCLinearNode",
    "SCConv1dNode",
    "SCConv2dNode",
    "SCSumPool2dNode",
    "SCAvgPool2dNode",
}

_SC_PASSTHROUGH_NODES: set[str] = {
    "SCInputNode",
    "SCOutputNode",
    "SCScaleNode",
    "SCFlattenNode",
    "SCThresholdNode",
    "SCDelayNode",
    "_UnitDelayNode",
}

_DELAY_NODE_NAME = "SCDelayNode"
_SCALE_NODE_NAME = "SCScaleNode"
_FLATTEN_NODE_NAME = "SCFlattenNode"
_THRESHOLD_NODE_NAME = "SCThresholdNode"
_SINGLE_PORT_SUBGRAPH_NODE = "SCSubgraphNode"
_MULTIPORT_SUBGRAPH_NODE = "SCMultiPortSubgraphNode"


def _extract_neuron_params(node: Any, neuron_type: str) -> dict[str, np.ndarray[Any, Any]]:
    """Extract canonical hardware parameters from a parsed neuron node."""
    params: dict[str, np.ndarray[Any, Any]] = {}
    for attribute in ("tau", "r", "v_leak", "v_threshold", "v_reset"):
        value = getattr(node, attribute, None)
        if value is not None:
            params[attribute] = np.atleast_1d(np.asarray(value, dtype=np.float64))

    if neuron_type in ("cuba_lif", "cuba_li"):
        for attribute in ("tau_syn", "tau_mem", "w_in"):
            value = getattr(node, attribute)
            params[attribute] = np.atleast_1d(np.asarray(value, dtype=np.float64))

    if neuron_type == "if":
        params.pop("tau", None)
        params.pop("v_leak", None)
    if neuron_type == "integrator":
        for attribute in ("tau", "v_leak", "v_threshold", "v_reset"):
            params.pop(attribute, None)
    return params


def _node_logical_width(node: Any) -> int | None:
    """Return the flattened channel width for a graph boundary or neuron node."""
    class_name = type(node).__name__
    if class_name == "SCInputNode":
        shape = getattr(node, "shape", None)
        if shape is None:
            return None
        if not shape:
            return 1
        return int(np.prod(np.asarray(shape, dtype=np.int64)))
    return int(getattr(node, "n_neurons", 1))
