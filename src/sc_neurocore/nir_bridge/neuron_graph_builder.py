# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCNetwork to hardware neuron graph conversion

"""Build the hardware-targeted neuron graph from a parsed SCNetwork."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from sc_neurocore.nir_bridge.neuron_graph_connections import (
    _fold_connection_scales,
    _resolve_weight_destination,
    _resolve_weight_source,
)
from sc_neurocore.nir_bridge.neuron_graph_contracts import (
    ConnectionSpec,
    DelaySteps,
    NeuronGraph,
    NeuronSpec,
)
from sc_neurocore.nir_bridge.neuron_graph_dense import _weight_matrix_and_bias
from sc_neurocore.nir_bridge.neuron_graph_hierarchy import _inline_single_port_subgraphs
from sc_neurocore.nir_bridge.neuron_graph_metadata import _broadcast_threshold
from sc_neurocore.nir_bridge.neuron_graph_nodes import (
    _SC_NODE_TO_TYPE,
    _SC_PASSTHROUGH_NODES,
    _SC_WEIGHT_NODES,
    _extract_neuron_params,
)

logger = logging.getLogger("sc_neurocore.nir_bridge.neuron_graph")


def from_scnetwork(network: Any, dt: float | None = None) -> NeuronGraph:
    """Convert a parsed SCNetwork to the FPGA-targeted neuron graph.

    Parameters
    ----------
    network : SCNetwork
        Parsed SC-NeuroCore network returned by :func:`from_nir`.
    dt : float or None, optional
        Simulation timestep override. When omitted, each neuron node retains
        its imported timestep and the first population supplies the graph
        timestep.

    Returns
    -------
    NeuronGraph
        Ordered populations, lowered weighted connections, graph boundaries,
        and preserved nested hierarchy metadata.

    Raises
    ------
    ValueError
        If nested graph boundaries are ambiguous, pass-through metadata cannot
        be represented exactly, or no neuron population remains after lowering.
    """
    network.topo_order
    (
        nodes,
        edges,
        topo_order,
        boundary_inputs,
        boundary_outputs,
        recurrent_map,
        hierarchy,
    ) = _inline_single_port_subgraphs(network)

    successors: dict[str, list[str]] = {}
    predecessors: dict[str, list[str]] = {}
    for source, destination in edges:
        successors.setdefault(source, []).append(destination)
        predecessors.setdefault(destination, []).append(source)

    populations: list[NeuronSpec] = []
    connections: list[ConnectionSpec] = []
    input_pop = ""
    output_pop = ""
    pending_weights: dict[
        str,
        tuple[np.ndarray[Any, Any], np.ndarray[Any, Any] | None],
    ] = {}
    weight_source_for: dict[
        str,
        tuple[str, np.ndarray[Any, Any] | None, int | None, np.ndarray[Any, Any] | None],
    ] = {}

    for name in topo_order:
        node = nodes[name]
        class_name = type(node).__name__
        if class_name == "SCInputNode":
            if name in boundary_inputs or not input_pop:
                input_pop = name
            continue
        if class_name == "SCOutputNode":
            if name in boundary_outputs or not output_pop:
                output_pop = name
            continue
        if class_name in _SC_WEIGHT_NODES:
            weight, bias = _weight_matrix_and_bias(node, name)
            pending_weights[name] = (weight, bias)
            for successor in successors.get(name, []):
                resolved_destination = _resolve_weight_destination(
                    successor,
                    nodes=nodes,
                    successors=successors,
                )
                if resolved_destination is not None:
                    (
                        destination_name,
                        destination_scale,
                        destination_flatten_width,
                        destination_threshold,
                    ) = resolved_destination
                    weight_source_for[destination_name] = (
                        name,
                        destination_scale,
                        destination_flatten_width,
                        destination_threshold,
                    )
            continue
        if class_name in _SC_PASSTHROUGH_NODES:
            continue

        neuron_type = _SC_NODE_TO_TYPE.get(class_name)
        if neuron_type is None:
            logger.warning(
                "Skipping unsupported node type %s (%s) in FPGA compilation",
                class_name,
                name,
            )
            continue

        n_neurons = getattr(node, "n_neurons", 1)
        node_dt = dt if dt is not None else getattr(node, "dt", 1.0)
        populations.append(
            NeuronSpec(
                name=name,
                neuron_type=neuron_type,
                n_neurons=max(1, n_neurons),
                params=_extract_neuron_params(node, neuron_type),
                dt=node_dt,
            )
        )

    for population in populations:
        weight_source = weight_source_for.get(population.name)
        if weight_source is None:
            continue
        (
            weight_node_name,
            destination_scale,
            destination_flatten_width,
            destination_threshold,
        ) = weight_source
        weights, bias = pending_weights[weight_node_name]

        source_name = ""
        delay_steps: DelaySteps = 0
        source_scale: np.ndarray[Any, Any] | None = None
        source_flatten_width: int | None = None
        source_threshold: np.ndarray[Any, Any] | None = None
        for predecessor in predecessors.get(weight_node_name, []):
            resolved_source = _resolve_weight_source(
                predecessor,
                nodes=nodes,
                predecessors=predecessors,
            )
            if resolved_source is not None:
                (
                    source_name,
                    delay_steps,
                    source_scale,
                    source_flatten_width,
                    source_threshold,
                ) = resolved_source
                break

        if not source_name:
            candidate_predecessors = predecessors.get(weight_node_name, [])
            if candidate_predecessors:
                source_name = candidate_predecessors[0]
            else:
                source_name = input_pop or "input"

        if source_flatten_width is not None and source_flatten_width != int(weights.shape[1]):
            raise ValueError(
                f"Flatten output width {source_flatten_width} does not match "
                f"weight input width {int(weights.shape[1])} for connection "
                f"{source_name}->{population.name}"
            )
        if destination_flatten_width is not None and destination_flatten_width != int(
            weights.shape[0]
        ):
            raise ValueError(
                f"Flatten input width {destination_flatten_width} does not match "
                f"weight output width {int(weights.shape[0])} for connection "
                f"{source_name}->{population.name}"
            )
        source_threshold = _broadcast_threshold(
            source_threshold,
            int(weights.shape[1]),
            f"source-side Threshold for connection {source_name}->{population.name}",
        )
        destination_threshold = _broadcast_threshold(
            destination_threshold,
            int(weights.shape[0]),
            f"post-weight Threshold for connection {source_name}->{population.name}",
        )
        folded_weights, folded_bias = _fold_connection_scales(
            weights,
            bias,
            source_scale=source_scale,
            destination_scale=destination_scale,
            src=source_name,
            dst=population.name,
        )
        connections.append(
            ConnectionSpec(
                src=source_name,
                dst=population.name,
                weights=folded_weights,
                bias=folded_bias,
                delay_steps=delay_steps,
                source_threshold=source_threshold,
                destination_threshold=destination_threshold,
            )
        )

    for delay_name, recurrent_source in recurrent_map.items():
        weight_data = pending_weights.get(recurrent_source)
        if weight_data is None:
            continue

        source_name = ""
        for predecessor in predecessors.get(recurrent_source, []):
            if type(nodes[predecessor]).__name__ in _SC_NODE_TO_TYPE:
                source_name = predecessor
                break
        if not source_name:
            continue

        destination_names = [
            destination
            for destination in successors.get(delay_name, [])
            if type(nodes[destination]).__name__ in _SC_NODE_TO_TYPE
        ]
        if not destination_names:
            continue

        weights, bias = weight_data
        for destination_name in destination_names:
            connections.append(
                ConnectionSpec(
                    src=source_name,
                    dst=destination_name,
                    weights=weights,
                    bias=bias,
                    delay_steps=1,
                )
            )

    if not input_pop and populations:
        input_pop = populations[0].name
    if not output_pop and populations:
        output_pop = populations[-1].name
    if not populations:
        raise ValueError(
            "NeuronGraph requires at least one neuron population. "
            "The NIR graph may contain only pass-through nodes."
        )

    global_dt = dt if dt is not None else populations[0].dt
    graph = NeuronGraph(
        populations=populations,
        connections=connections,
        input_pop=input_pop,
        output_pop=output_pop,
        dt=global_dt,
        hierarchy=hierarchy,
    )
    logger.info(
        "Built NeuronGraph: %d populations, %d connections, %d neurons, %d synapses",
        len(populations),
        len(connections),
        graph.total_neurons,
        graph.total_synapses,
    )
    return graph
