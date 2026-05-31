# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia mirror for physics/wolfram_hypergraph contracts

module WolframHypergraphAccel

mutable struct WolframHypergraphState
    edges::Vector{Tuple{Vararg{Int}}}
    max_node_id::Int
end

function _validate_edges(edges)
    if !(edges isa Vector)
        throw(ArgumentError("edges must be a vector of integer tuples"))
    end
    for edge in edges
        if !(edge isa Tuple) || length(edge) == 0
            throw(ArgumentError("edges must contain non-empty integer tuples"))
        end
        if any(node -> !(node isa Int) || node < 0, edge)
            throw(ArgumentError("edge nodes must be non-negative integers"))
        end
        if length(unique(edge)) != length(edge)
            throw(ArgumentError("hyperedges must not repeat nodes"))
        end
    end
    return edges
end

function _validate_max_node_id(max_node_id::Int, edges)
    if max_node_id < 0
        throw(ArgumentError("max_node_id must be a non-negative integer"))
    end
    observed = isempty(edges) ? -1 : maximum(node for edge in edges for node in edge)
    if observed > max_node_id
        throw(ArgumentError("max_node_id must be at least the largest node in edges"))
    end
    return max_node_id
end

function WolframHypergraphState(edges::Vector{Tuple{Vararg{Int}}}, max_node_id::Int)
    clean_edges = _validate_edges(edges)
    clean_max = _validate_max_node_id(max_node_id, clean_edges)
    return WolframHypergraphState(copy(clean_edges), clean_max)
end

function evolve!(s::WolframHypergraphState, steps::Int=1)
    if steps < 0
        throw(ArgumentError("steps must be a non-negative integer"))
    end
    _validate_edges(s.edges)
    _validate_max_node_id(s.max_node_id, s.edges)

    for _ in 1:steps
        new_edges = Tuple{Vararg{Int}}[]
        matched = Set{Int}()
        for (i, e1) in enumerate(s.edges)
            if i in matched || length(e1) != 2
                continue
            end
            x, y = e1
            for (j, e2) in enumerate(s.edges)
                if i == j || j in matched || length(e2) != 2
                    continue
                end
                if e2[1] == y
                    z = e2[2]
                    w = s.max_node_id + 1
                    s.max_node_id = w
                    push!(new_edges, (x, z))
                    push!(new_edges, (x, w))
                    push!(new_edges, (y, w))
                    push!(matched, i)
                    push!(matched, j)
                    break
                end
            end
        end
        for (k, edge) in enumerate(s.edges)
            if !(k in matched)
                push!(new_edges, edge)
            end
        end
        s.edges = _validate_edges(new_edges)
        _validate_max_node_id(s.max_node_id, s.edges)
    end
    return s
end

end # module WolframHypergraphAccel
