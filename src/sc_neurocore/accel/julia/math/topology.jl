# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for math/topology

module TopologyAccel

export winding_number, ollivier_ricci_curvature, sheaf_consistency_defect, connection_curvature

function _validate_graph(knm::AbstractMatrix{<:Real})
    n, m = size(knm)
    n == m || throw(ArgumentError("knm must be a square coupling matrix"))
    n > 0 || throw(ArgumentError("knm must contain at least one node"))
    graph = Float64.(knm)
    all(isfinite, graph) || throw(ArgumentError("knm must contain only finite values"))
    any(graph .< 0.0) && throw(ArgumentError("knm must be non-negative for Ollivier-Ricci curvature"))
    return copy(graph)
end

function _validate_index(name::AbstractString, index::Integer, n::Integer)
    1 <= index <= n || throw(ArgumentError("$name out of range for coupling graph"))
    return Int(index)
end

function _shortest_path_distances(graph::Matrix{Float64})
    n = size(graph, 1)
    distances = fill(Inf, n, n)
    adjacency = graph .> 0.0
    for node in 1:n
        adjacency[node, node] = false
    end
    for source in 1:n
        distances[source, source] = 0.0
        queue = [source]
        head = 1
        while head <= length(queue)
            current = queue[head]
            head += 1
            next_distance = distances[source, current] + 1.0
            for target in findall(adjacency[current, :])
                if next_distance < distances[source, target]
                    distances[source, target] = next_distance
                    push!(queue, target)
                end
            end
        end
    end
    return distances
end

function _lazy_random_walk(graph::Matrix{Float64}, node::Integer; idleness::Float64=0.5)
    distribution = zeros(Float64, size(graph, 1))
    distribution[node] = idleness
    row = copy(graph[node, :])
    row[node] = 0.0
    row_sum = sum(row)
    if row_sum == 0.0
        distribution[node] = 1.0
        return distribution
    end
    distribution .+= (1.0 - idleness) .* row ./ row_sum
    return distribution
end

function _minimum_transport_cost(source::Vector{Float64}, target::Vector{Float64}, distances::Matrix{Float64})
    source_nodes = findall(source .> 0.0)
    target_nodes = findall(target .> 0.0)
    isempty(source_nodes) || isempty(target_nodes) && return 0.0
    costs = distances[source_nodes, target_nodes]
    all(isfinite, costs) || return Inf
    supply_count = length(source_nodes)
    demand_count = length(target_nodes)
    source_id = supply_count + demand_count + 1
    sink_id = source_id + 1
    node_count = sink_id
    residual = zeros(Float64, node_count, node_count)
    edge_cost = zeros(Float64, node_count, node_count)
    for (idx, node) in enumerate(source_nodes)
        residual[source_id, idx] = source[node]
    end
    for (idx, node) in enumerate(target_nodes)
        residual[supply_count + idx, sink_id] = target[node]
    end
    for s_idx in 1:supply_count, d_idx in 1:demand_count
        u = s_idx
        v = supply_count + d_idx
        residual[u, v] = Inf
        edge_cost[u, v] = costs[s_idx, d_idx]
        edge_cost[v, u] = -costs[s_idx, d_idx]
    end
    required = sum(source)
    transported = 0.0
    total_cost = 0.0
    tolerance = 1e-12
    while transported + tolerance < required
        dist = fill(Inf, node_count)
        parent = fill(0, node_count)
        dist[source_id] = 0.0
        for _ in 1:(node_count - 1)
            updated = false
            for u in 1:node_count
                isfinite(dist[u]) || continue
                for v in 1:node_count
                    residual[u, v] > tolerance || continue
                    candidate = dist[u] + edge_cost[u, v]
                    if candidate < dist[v] - tolerance
                        dist[v] = candidate
                        parent[v] = u
                        updated = true
                    end
                end
            end
            updated || break
        end
        parent[sink_id] != 0 || throw(ArgumentError("transport problem is infeasible"))
        increment = required - transported
        v = sink_id
        while v != source_id
            u = parent[v]
            increment = min(increment, residual[u, v])
            v = u
        end
        v = sink_id
        while v != source_id
            u = parent[v]
            residual[u, v] -= increment
            residual[v, u] += increment
            total_cost += increment * edge_cost[u, v]
            v = u
        end
        transported += increment
    end
    return total_cost
end

function winding_number(phases::AbstractVector{<:Real})
    values = Float64.(phases)
    all(isfinite, values) || throw(ArgumentError("phases must be finite"))
    diffs = diff(values)
    diffs = ifelse.(diffs .> pi, diffs .- 2pi, diffs)
    diffs = ifelse.(diffs .< -pi, diffs .+ 2pi, diffs)
    return Int(round(sum(diffs) / (2pi)))
end

function ollivier_ricci_curvature(knm::AbstractMatrix{<:Real}, i::Integer, j::Integer)
    graph = _validate_graph(knm)
    n = size(graph, 1)
    source = _validate_index("i", i, n)
    target = _validate_index("j", j, n)
    source == target && return 0.0
    distances = _shortest_path_distances(graph)
    graph_distance = distances[source, target]
    (!isfinite(graph_distance) || graph_distance <= 0.0) && return 0.0
    mu_source = _lazy_random_walk(graph, source)
    mu_target = _lazy_random_walk(graph, target)
    w1 = _minimum_transport_cost(mu_source, mu_target, distances)
    isfinite(w1) || return 0.0
    return 1.0 - w1 / graph_distance
end

function sheaf_consistency_defect(phases::AbstractVector{<:Real}, knm::AbstractMatrix{<:Real})
    values = Float64.(phases)
    graph = Float64.(knm)
    n = length(values)
    diffs = reshape(values, 1, n) .- reshape(values, n, 1)
    cost = abs.(graph) .* (1.0 .- cos.(diffs))
    return sum(cost) / (n * n)
end

function connection_curvature(phases::AbstractVector{<:Real}, knm::AbstractMatrix{<:Real})
    values = Float64.(phases)
    n = length(values)
    diffs = reshape(values, 1, n) .- reshape(values, n, 1)
    return Float64.(knm) .* cos.(diffs)
end

end # module TopologyAccel
