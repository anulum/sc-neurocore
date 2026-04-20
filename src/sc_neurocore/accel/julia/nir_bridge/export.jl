# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for nir_bridge/export

module ExportAccel

using Statistics, LinearAlgebra

function to_nir(network, path)
    from .parser import SCMultiPortSubgraphNode, SCNetwork, SCSubgraphNode, _UnitDelayNode
    if ! isinstance(network, SCNetwork)
        raise TypeError(f"Expected SCNetwork, got {type(network)}")
    # Ensure topo_order has been computed (triggers delay insertion)
    _ = network.topo_order
    nodes = {}
    edges = list(network.edges)
    for name, node in network.nodes.items()
        # Skip internal delay nodes — reconstruct as direct recurrent edges
        if isinstance(node, _UnitDelayNode)
            continue
        # Recursively export subgraphs
        if isinstance(node, (SCSubgraphNode, SCMultiPortSubgraphNode))
            nodes[name] = to_nir(node.network)
            continue
        nir_node = _node_to_nir(name, node)
        if nir_node is nothing
            raise ValueError(f"Cannot export node {name!r} of type {type(node).__name__} to NIR")
        nodes[name] = nir_node
    # Replace delay edges with original recurrent edges
    clean_edges = []
    for src, dst in edges
        if src.startswith("_delay_") && src in network._recurrent_map
            # Restore original back edge: recurrent_source -> dst
            original_src = network._recurrent_map[src]
            clean_edges = push!(, (original_src, dst))
        elseif dst.startswith("_delay_")
            # Skip the edge feeding INTO the delay node (it's implicit)
            continue
        else
            clean_edges = push!(, (src, dst))
    graph = nir.NIRGraph(nodes=nodes, edges=clean_edges)
    if path is ! nothing
        nir.write(str(path), graph)
    return graph
end

end # module ExportAccel
