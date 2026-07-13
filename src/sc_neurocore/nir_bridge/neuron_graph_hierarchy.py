# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Nested NIR graph flattening for hardware lowering

"""Inline nested NIR graphs while preserving explicit hierarchy provenance."""

from __future__ import annotations

from typing import Any

from sc_neurocore.nir_bridge.neuron_graph_contracts import HierarchyInstanceSpec
from sc_neurocore.nir_bridge.neuron_graph_nodes import (
    _MULTIPORT_SUBGRAPH_NODE,
    _SINGLE_PORT_SUBGRAPH_NODE,
)


def _topological_order(nodes: dict[str, Any], edges: list[tuple[str, str]]) -> list[str]:
    """Return a deterministic order for an already cycle-broken graph."""
    successors: dict[str, list[str]] = {name: [] for name in nodes}
    in_degree: dict[str, int] = {name: 0 for name in nodes}
    for source, destination in edges:
        if source not in nodes or destination not in nodes:
            raise ValueError(
                "Flattened nested NIRGraph edge references unknown node "
                f"{source!r}->{destination!r}"
            )
        successors[source].append(destination)
        in_degree[destination] += 1

    ready = [name for name, degree in in_degree.items() if degree == 0]
    order: list[str] = []
    while ready:
        name = ready.pop(0)
        order.append(name)
        for destination in successors[name]:
            in_degree[destination] -= 1
            if in_degree[destination] == 0:
                ready.append(destination)

    if len(order) != len(nodes):
        raise ValueError(
            "Flattened nested NIRGraph contains a cycle without an explicit delay node"
        )
    return order


def _hdl_identifier_fragment(value: str) -> str:
    """Return a conservative Verilog identifier fragment."""
    fragment = "".join(
        character if character.isascii() and (character.isalnum() or character == "_") else "_"
        for character in value
    ).strip("_")
    if not fragment:
        fragment = "instance"
    if not (fragment[0].isalpha() or fragment[0] == "_"):
        fragment = f"n_{fragment}"
    return fragment


def _inline_single_port_subgraphs(
    network: Any,
) -> tuple[
    dict[str, Any],
    list[tuple[str, str]],
    list[str],
    set[str],
    set[str],
    dict[str, str],
    tuple[HierarchyInstanceSpec, ...],
]:
    """Inline parser-executable nested graphs for SC-NIR and FPGA lowering."""
    nodes = dict(network.nodes)
    edges = list(network.edges)
    recurrent_map = dict(getattr(network, "_recurrent_map", {}))
    boundary_inputs = set(getattr(network, "input_nodes", ()))
    boundary_outputs = set(getattr(network, "output_nodes", ()))
    hierarchy: list[HierarchyInstanceSpec] = []

    changed = True
    while changed:
        changed = False
        for name in _topological_order(nodes, edges):
            node = nodes[name]
            class_name = type(node).__name__
            if class_name not in {_SINGLE_PORT_SUBGRAPH_NODE, _MULTIPORT_SUBGRAPH_NODE}:
                continue

            subnetwork = getattr(node, "network", None)
            if subnetwork is None:
                raise ValueError(f"Nested NIRGraph node {name!r} does not expose a parsed network")
            nested_inputs = tuple(str(item) for item in subnetwork.input_nodes)
            nested_outputs = tuple(str(item) for item in subnetwork.output_nodes)
            if not nested_inputs or not nested_outputs:
                raise ValueError(
                    f"Nested NIRGraph node {name!r} must expose at least one input and one output "
                    "for SC-NIR/FPGA lowering"
                )
            if class_name == _SINGLE_PORT_SUBGRAPH_NODE and (
                len(nested_inputs) != 1 or len(nested_outputs) != 1
            ):
                raise ValueError(
                    f"Nested NIRGraph node {name!r} must expose exactly one input and one output "
                    "for inline SC-NIR/FPGA lowering"
                )

            subnetwork.topo_order
            prefix = f"{name}__"
            prefixed_nodes = {
                f"{prefix}{inner_name}": inner_node
                for inner_name, inner_node in subnetwork.nodes.items()
            }
            collisions = sorted(set(nodes).intersection(prefixed_nodes))
            if collisions:
                raise ValueError(
                    f"Nested NIRGraph node {name!r} would collide with existing nodes: {collisions}"
                )

            incoming = [source for source, destination in edges if destination == name]
            outgoing = [destination for source, destination in edges if source == name]
            if len(incoming) != len(nested_inputs) or len(outgoing) != len(nested_outputs):
                raise ValueError(
                    f"Multi-port nested NIRGraph node {name!r} boundary mapping requires "
                    f"{len(nested_inputs)} incoming and {len(nested_outputs)} outgoing edges; "
                    f"got {len(incoming)} incoming and {len(outgoing)} outgoing"
                )
            hierarchy.append(
                HierarchyInstanceSpec(
                    instance_id=name,
                    module_name=f"scnir_{_hdl_identifier_fragment(name)}",
                    node_name_prefix=prefix,
                )
            )

            edges = [
                (source, destination)
                for source, destination in edges
                if source != name and destination != name
            ]
            edges.extend(
                (source, f"{prefix}{nested_input}")
                for source, nested_input in zip(incoming, nested_inputs, strict=True)
            )
            edges.extend(
                (f"{prefix}{nested_output}", destination)
                for nested_output, destination in zip(nested_outputs, outgoing, strict=True)
            )
            edges.extend(
                (f"{prefix}{source}", f"{prefix}{destination}")
                for source, destination in subnetwork.edges
            )
            nodes.pop(name)
            nodes.update(prefixed_nodes)
            recurrent_map.update(
                {
                    f"{prefix}{delay_name}": f"{prefix}{source_name}"
                    for delay_name, source_name in getattr(
                        subnetwork,
                        "_recurrent_map",
                        {},
                    ).items()
                }
            )
            changed = True
            break

    topo_order = _topological_order(nodes, edges)
    return (
        nodes,
        edges,
        topo_order,
        boundary_inputs,
        boundary_outputs,
        recurrent_map,
        tuple(hierarchy),
    )
