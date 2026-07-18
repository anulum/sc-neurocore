# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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
    SCAffineNode,
    SCAvgPool2dNode,
    SCConv1dNode,
    SCConv2dNode,
    SCCubaLIFNode,
    SCCubaLINode,
    SCDelayNode,
    SCFlattenNode,
    SCIFNode,
    SCInputNode,
    SCIntegratorNode,
    SCLIFNode,
    SCLINode,
    SCLinearNode,
    SCOutputNode,
    SCScaleNode,
    SCSumPool2dNode,
    SCThresholdNode,
)


def _node_to_nir(name: str, node) -> nir.NIRNode | None:  # type: ignore[no-untyped-def]
    """Convert a single SC-NeuroCore node to its NIR equivalent."""
    if isinstance(node, SCInputNode):
        return nir.Input(input_type={"input": np.array(list(node.shape))})
    if isinstance(node, SCOutputNode):
        return nir.Output(output_type={"output": np.array(list(node.shape))})
    if isinstance(node, SCLIFNode):
        return nir.LIF(
            tau=node.tau.copy(),
            r=node.r.copy(),
            v_leak=node.v_leak.copy(),
            v_threshold=node.v_threshold.copy(),
            v_reset=node.v_reset.copy(),
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
        if node.input_shape is None:
            raise ValueError(
                f"Cannot export Flatten node {node.name!r} to NIR: input_shape is unknown "
                "(NIR >= 1.0.8 requires Flatten.input_type). Import the source graph with "
                "shape metadata before exporting."
            )
        return nir.Flatten(
            input_type={"input": np.array(list(node.input_shape))},
            start_dim=node.start_dim,
            end_dim=node.end_dim,
        )
    if isinstance(node, SCCubaLIFNode):
        return nir.CubaLIF(
            tau_syn=node.tau_syn.copy(),
            tau_mem=node.tau_mem.copy(),
            r=node.r.copy(),
            v_leak=node.v_leak.copy(),
            v_threshold=node.v_threshold.copy(),
            w_in=node.w_in.copy(),
            v_reset=node.v_reset.copy(),
        )
    if isinstance(node, SCCubaLINode):
        return nir.CubaLI(
            tau_syn=node.tau_syn.copy(),
            tau_mem=node.tau_mem.copy(),
            r=node.r.copy(),
            v_leak=node.v_leak.copy(),
            w_in=node.w_in.copy(),
        )
    if isinstance(node, SCDelayNode):
        delay = (
            node.delay_time if node.delay_time is not None else node.delay_steps.astype(np.float64)
        )
        return nir.Delay(delay=delay)
    if isinstance(node, SCConv1dNode):
        return nir.Conv1d(
            input_shape=node.input_shape,
            weight=node.weight.copy(),
            stride=node.stride,
            padding=node.padding,
            dilation=node.dilation,
            groups=node.groups,
            bias=node.bias.copy() if node.bias is not None else None,
        )
    if isinstance(node, SCConv2dNode):
        return nir.Conv2d(
            input_shape=node.input_shape,
            weight=node.weight.copy(),
            stride=node.stride,
            padding=node.padding,
            dilation=node.dilation,
            groups=node.groups,
            bias=node.bias.copy() if node.bias is not None else None,
        )
    if isinstance(node, SCSumPool2dNode):
        return nir.SumPool2d(
            kernel_size=node.kernel_size,
            stride=node.stride,
            padding=node.padding,
        )
    if isinstance(node, SCAvgPool2dNode):
        return nir.AvgPool2d(
            kernel_size=node.kernel_size,
            stride=node.stride,
            padding=node.padding,
        )
    return None


def to_nir(network, path: str | Path | None = None) -> nir.NIRGraph:  # type: ignore[no-untyped-def]
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
    from .parser import SCMultiPortSubgraphNode, SCNetwork, SCSubgraphNode, _UnitDelayNode

    if not isinstance(network, SCNetwork):
        raise TypeError(f"Expected SCNetwork, got {type(network)}")

    # Ensure topo_order has been computed (triggers delay insertion)
    _ = network.topo_order

    nodes = {}
    edges = list(network.edges)

    for name, node in network.nodes.items():
        # Skip internal delay nodes — reconstruct as direct recurrent edges
        if isinstance(node, _UnitDelayNode):
            continue
        # Recursively export subgraphs
        if isinstance(node, (SCSubgraphNode, SCMultiPortSubgraphNode)):
            nodes[name] = to_nir(node.network)
            continue
        nir_node = _node_to_nir(name, node)
        if nir_node is None:
            raise ValueError(f"Cannot export node {name!r} of type {type(node).__name__} to NIR")
        nodes[name] = nir_node

    # Replace delay edges with original recurrent edges
    clean_edges = []
    for src, dst in edges:
        if src.startswith("_delay_") and src in network._recurrent_map:
            # Restore original back edge: recurrent_source -> dst
            original_src = network._recurrent_map[src]
            clean_edges.append((original_src, dst))
        elif dst.startswith("_delay_"):
            # Skip the edge feeding INTO the delay node (it's implicit)
            continue
        else:
            clean_edges.append((src, dst))

    graph = nir.NIRGraph(nodes=nodes, edges=clean_edges)

    if path is not None:
        nir.write(str(path), graph)

    return graph
