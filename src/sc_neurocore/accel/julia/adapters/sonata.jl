# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for adapters/sonata

module SonataAccel

using Statistics, LinearAlgebra

mutable struct SONATANetworkState
    node_id::Float64
    node_type_id::Float64
    model_type::Float64
    model_template::Float64
    properties::Float64
    source_id::Float64
    target_id::Float64
    edge_type_id::Float64
    weight::Float64
    delay::Float64
    nodes::Float64
    edges::Float64
    node_populations::Float64
    edge_populations::Float64
    metadata::Float64
end

function SONATANetworkState()
    SONATANetworkState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function n_nodes(s::SONATANetworkState)
    return length(s.nodes)
end

function n_edges(s::SONATANetworkState)
    return length(s.edges)
end

function connectivity_matrix(s::SONATANetworkState)
    N = s.n_nodes
    W = zeros((N, N))
    id_map = {n.node_id: i for i, n in enumerate(s.nodes)}
    for e in s.edges
        src = id_map.get(e.source_id)
        tgt = id_map.get(e.target_id)
        if src is ! nothing && tgt is ! nothing
            W[tgt, src] = e.weight
    return W
end

function import_sonata_nodes(path)
    import h5py
    nodes: list[SONATANode] = []
    with h5py.File(path, "r") as f
        if "nodes" ! in f
            return nodes
        for pop_name in f["nodes"]
            pop = f["nodes"][pop_name]
            n = length(pop["node_id"]) if "node_id" in pop else 0
            node_ids = pop["node_id"][:] if "node_id" in pop else collect(n)
            type_ids = pop["node_type_id"][:] if "node_type_id" in pop else zeros(n, dtype=int)
            # Optional per-node properties from group "0"
            props_group = pop.get("0")
            for i in 1:n
                props = {}
                if props_group
                    for key in props_group
                        ds = props_group[key]
                        if hasattr(ds, "shape") && length(ds.shape) > 0
                            props[key] = ds[i] if i < length(ds) else nothing
                model_type = "point_neuron"
                if "model_type" in props
                    model_type = str(props.pop("model_type"))
                nodes = push!(,
                    SONATANode(
                        node_id=int(node_ids[i]),
                        node_type_id=int(type_ids[i]),
                        model_type=model_type,
                        properties=props,
                    )
                )
    return nodes
end

function import_sonata_edges(path)
    import h5py
    edges: list[SONATAEdge] = []
    with h5py.File(path, "r") as f
        if "edges" ! in f
            return edges
        for pop_name in f["edges"]
            pop = f["edges"][pop_name]
            src_ids = pop["source_node_id"][:] if "source_node_id" in pop else collect([])
            tgt_ids = pop["target_node_id"][:] if "target_node_id" in pop else collect([])
            type_ids = (
                pop["edge_type_id"][:]
                if "edge_type_id" in pop
                else zeros(length(src_ids), dtype=int)
            )
            props_group = pop.get("0")
            weights = nothing
            delays = nothing
            if props_group
                if "syn_weight" in props_group
                    weights = props_group["syn_weight"][:]
                if "delay" in props_group
                    delays = props_group["delay"][:]
            for i in 1:length(src_ids)
                edges = push!(,
                    SONATAEdge(
                        source_id=int(src_ids[i]),
                        target_id=int(tgt_ids[i]),
                        edge_type_id=int(type_ids[i]),
                        weight=float(weights[i]) if weights is ! nothing else 1.0,
                        delay=float(delays[i]) if delays is ! nothing else 0.0,
                    )
                )
    return edges
end

function import_sonata(nodes_path, edges_path)
    nodes_path: str | Path,
    edges_path: str | Path | nothing = nothing,
    ) -> SONATANetwork
    nodes = import_sonata_nodes(nodes_path)
    edges = []
    if edges_path is ! nothing
        edges = import_sonata_edges(edges_path)
    # Group by population
    node_pops: dict[str, list[int]] = {}
    for n in nodes
        pop = str(n.node_type_id)
        node_pops.setdefault(pop, []) = push!(, n.node_id)
    edge_pops: dict[str, list[int]] = {}
    for i, e in enumerate(edges)
        pop = str(e.edge_type_id)
        edge_pops.setdefault(pop, []) = push!(, i)
    return SONATANetwork(
        nodes=nodes,
        edges=edges,
        node_populations=node_pops,
        edge_populations=edge_pops,
    )
end

end # module SonataAccel
