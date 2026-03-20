# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Export SC-NeuroCore networks to NIR format

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import nir
except ImportError as e:
    raise ImportError("pip install nir") from e

from .node_map import (
    SCInputNode,
    SCOutputNode,
    SCLIFNode,
    SCIFNode,
    SCLINode,
    SCIntegratorNode,
    SCAffineNode,
    SCLinearNode,
    SCScaleNode,
    SCThresholdNode,
    SCFlattenNode,
)


def _node_to_nir(name: str, node) -> nir.NIRNode | None:
    """Convert a single SC-NeuroCore node to its NIR equivalent."""
    if isinstance(node, SCInputNode):
        return nir.Input(input_type={"input": np.array(list(node.shape))})
    if isinstance(node, SCOutputNode):
        return nir.Output(output_type={"output": np.array(list(node.shape))})
    if isinstance(node, SCLIFNode):
        n = len(node.neurons)
        return nir.LIF(
            tau=np.array([neuron.tau_mem for neuron in node.neurons]),
            r=np.array([neuron.resistance for neuron in node.neurons]),
            v_leak=np.array([neuron.v_rest for neuron in node.neurons]),
            v_threshold=np.array([neuron.v_threshold for neuron in node.neurons]),
            v_reset=np.array([neuron.v_reset for neuron in node.neurons]),
        )
    if isinstance(node, SCIFNode):
        return nir.IF(
            r=node.r.copy(),
            v_threshold=node.v_threshold.copy(),
            v_reset=node.v_reset.copy(),
        )
    if isinstance(node, SCLINode):
        return nir.LI(
            tau=node.tau.copy(),
            r=node.r.copy(),
            v_leak=node.v_leak.copy(),
        )
    if isinstance(node, SCIntegratorNode):
        return nir.I(r=node.r.copy())
    if isinstance(node, SCAffineNode):
        return nir.Affine(weight=node.weight.copy(), bias=node.bias.copy())
    if isinstance(node, SCLinearNode):
        return nir.Linear(weight=node.weight.copy())
    if isinstance(node, SCScaleNode):
        return nir.Scale(scale=node.scale.copy())
    if isinstance(node, SCThresholdNode):
        return nir.Threshold(threshold=node.threshold.copy())
    if isinstance(node, SCFlattenNode):
        return nir.Flatten(start_dim=node.start_dim, end_dim=node.end_dim)
    return None


def to_nir(network, path: str | Path | None = None) -> nir.NIRGraph:
    """Export an SC-NeuroCore SCNetwork to NIR format.

    Parameters
    ----------
    network : SCNetwork
        The network to export.
    path : str or Path, optional
        If provided, write the NIR graph to this file.

    Returns
    -------
    nir.NIRGraph
    """
    from .parser import SCNetwork

    if not isinstance(network, SCNetwork):
        raise TypeError(f"Expected SCNetwork, got {type(network)}")

    nodes = {}
    for name, node in network.nodes.items():
        nir_node = _node_to_nir(name, node)
        if nir_node is None:
            raise ValueError(f"Cannot export node {name!r} of type {type(node).__name__} to NIR")
        nodes[name] = nir_node

    graph = nir.NIRGraph(nodes=nodes, edges=network.edges)

    if path is not None:
        nir.write(str(path), graph)

    return graph
