# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo Ollivier-Ricci curvature (parity with math/topology.py)
#
# Build:
#   mojo build --emit shared-lib -o libtopology.so topology.mojo
#
# Parity contract: `ollivier_ricci_curvature_c` reproduces the value of
# `sc_neurocore.math.topology.ollivier_ricci_curvature` within float64
# round-off for the same coupling matrix and node pair. The min-cost-flow
# loop follows the same Bellman-Ford ascending-node iteration order as the
# Python, Rust, and Go references so the chosen augmenting paths — and the
# floating-point accumulation of the transport cost — match.
#
# Reference: Ollivier (2009), J. Functional Analysis 256(3): 810-864.
#
# Mojo FFI rules (per feedback_mojo_026_ffi_pattern): @export rejects
# parametric signatures, so the coupling matrix is a raw `Int` address +
# size scalars; the pointer is reconstructed inside. The matrix is flat
# row-major Float64 (matches the Python/Rust/Julia/Go convention).

from std.memory import UnsafePointer, alloc

alias IDLENESS: Float64 = 0.5
alias TOLERANCE: Float64 = 1e-12


@always_inline
fn _ptr(addr: Int) -> UnsafePointer[Float64, MutAnyOrigin]:
    return UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=addr)


fn _alloc(n: Int) -> UnsafePointer[Float64, MutAnyOrigin]:
    var raw = alloc[Float64](n)
    return UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=Int(raw))


fn _alloc_int(n: Int) -> UnsafePointer[Int, MutAnyOrigin]:
    var raw = alloc[Int](n)
    return UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=Int(raw))


fn _free(p: UnsafePointer[Float64, MutAnyOrigin]):
    var raw = UnsafePointer[Float64, MutExternalOrigin](unsafe_from_address=Int(p))
    raw.free()


fn _free_int(p: UnsafePointer[Int, MutAnyOrigin]):
    var raw = UnsafePointer[Int, MutExternalOrigin](unsafe_from_address=Int(p))
    raw.free()


@always_inline
fn _inf() -> Float64:
    var big: Float64 = 1.0e308
    return big * 10.0


@always_inline
fn _is_inf(x: Float64) -> Bool:
    return x > 1.0e307


# ─── BFS hop-count all-pairs distances (flat n*n, row-major) ──────

fn _shortest_path_distances(
    graph: UnsafePointer[Float64, MutAnyOrigin], n: Int
) -> UnsafePointer[Float64, MutAnyOrigin]:
    var distances = _alloc(n * n)
    for k in range(n * n):
        distances[k] = _inf()
    var queue = _alloc_int(n)
    for source in range(n):
        distances[source * n + source] = 0.0
        var head = 0
        var tail = 0
        queue[tail] = source
        tail += 1
        while head < tail:
            var current = queue[head]
            head += 1
            var next_distance = distances[source * n + current] + 1.0
            for target in range(n):
                if target == current or graph[current * n + target] <= 0.0:
                    continue
                if next_distance < distances[source * n + target]:
                    distances[source * n + target] = next_distance
                    queue[tail] = target
                    tail += 1
    _free_int(queue)
    return distances


fn _lazy_random_walk(
    graph: UnsafePointer[Float64, MutAnyOrigin], n: Int, node: Int
) -> UnsafePointer[Float64, MutAnyOrigin]:
    var distribution = _alloc(n)
    for k in range(n):
        distribution[k] = 0.0
    distribution[node] = IDLENESS
    var row_sum: Float64 = 0.0
    for k in range(n):
        if k != node:
            row_sum += graph[node * n + k]
    if row_sum == 0.0:
        distribution[node] = 1.0
        return distribution
    for k in range(n):
        if k != node:
            distribution[k] += (1.0 - IDLENESS) * graph[node * n + k] / row_sum
    return distribution


# ─── exact Wasserstein-1 via successive-shortest-path min-cost flow ──
# Returns NaN on an infeasible transport sub-problem.

fn _minimum_transport_cost(
    source: UnsafePointer[Float64, MutAnyOrigin],
    target: UnsafePointer[Float64, MutAnyOrigin],
    distances: UnsafePointer[Float64, MutAnyOrigin],
    n: Int,
) -> Float64:
    var source_nodes = _alloc_int(n)
    var target_nodes = _alloc_int(n)
    var total_supply = 0
    var total_demand = 0
    for k in range(n):
        if source[k] > 0.0:
            source_nodes[total_supply] = k
            total_supply += 1
        if target[k] > 0.0:
            target_nodes[total_demand] = k
            total_demand += 1
    if total_supply == 0 or total_demand == 0:
        _free_int(source_nodes)
        _free_int(target_nodes)
        return 0.0

    var costs = _alloc(total_supply * total_demand)
    for s_idx in range(total_supply):
        for d_idx in range(total_demand):
            var cost = distances[source_nodes[s_idx] * n + target_nodes[d_idx]]
            if _is_inf(cost):
                _free(costs)
                _free_int(source_nodes)
                _free_int(target_nodes)
                return _inf()
            costs[s_idx * total_demand + d_idx] = cost

    var source_id = total_supply + total_demand
    var sink_id = source_id + 1
    var node_count = sink_id + 1
    var residual = _alloc(node_count * node_count)
    var edge_cost = _alloc(node_count * node_count)
    for k in range(node_count * node_count):
        residual[k] = 0.0
        edge_cost[k] = 0.0

    for idx in range(total_supply):
        residual[source_id * node_count + idx] = source[source_nodes[idx]]
    for idx in range(total_demand):
        residual[(total_supply + idx) * node_count + sink_id] = target[target_nodes[idx]]
    for s_idx in range(total_supply):
        for d_idx in range(total_demand):
            var u = s_idx
            var v = total_supply + d_idx
            var cost = costs[s_idx * total_demand + d_idx]
            residual[u * node_count + v] = _inf()
            edge_cost[u * node_count + v] = cost
            edge_cost[v * node_count + u] = -cost

    var required: Float64 = 0.0
    for k in range(n):
        required += source[k]
    var transported: Float64 = 0.0
    var total_cost: Float64 = 0.0

    var dist = _alloc(node_count)
    var parent = _alloc_int(node_count)

    var infeasible = False
    while transported + TOLERANCE < required:
        for k in range(node_count):
            dist[k] = _inf()
            parent[k] = -1
        dist[source_id] = 0.0
        for _iter in range(node_count - 1):
            var updated = False
            for u in range(node_count):
                if _is_inf(dist[u]):
                    continue
                for v in range(node_count):
                    if residual[u * node_count + v] <= TOLERANCE:
                        continue
                    var candidate = dist[u] + edge_cost[u * node_count + v]
                    if candidate < dist[v] - TOLERANCE:
                        dist[v] = candidate
                        parent[v] = u
                        updated = True
            if not updated:
                break
        if parent[sink_id] == -1:
            infeasible = True
            break

        var increment = required - transported
        var v = sink_id
        while v != source_id:
            var u = parent[v]
            if residual[u * node_count + v] < increment:
                increment = residual[u * node_count + v]
            v = u
        v = sink_id
        while v != source_id:
            var u = parent[v]
            residual[u * node_count + v] -= increment
            residual[v * node_count + u] += increment
            total_cost += increment * edge_cost[u * node_count + v]
            v = u
        transported += increment

    _free(dist)
    _free_int(parent)
    _free(costs)
    _free(residual)
    _free(edge_cost)
    _free_int(source_nodes)
    _free_int(target_nodes)

    if infeasible:
        var zero: Float64 = 0.0
        return zero / zero  # NaN signals an infeasible transport sub-problem
    return total_cost


@export
fn ollivier_ricci_curvature_c(knm_addr: Int, n: Int, i: Int, j: Int) -> Float64:
    var graph = _ptr(knm_addr)
    if i == j:
        return 0.0
    var distances = _shortest_path_distances(graph, n)
    var graph_distance = distances[i * n + j]
    if _is_inf(graph_distance) or graph_distance <= 0.0:
        _free(distances)
        return 0.0
    var mu_i = _lazy_random_walk(graph, n, i)
    var mu_j = _lazy_random_walk(graph, n, j)
    var w1 = _minimum_transport_cost(mu_i, mu_j, distances, n)
    _free(mu_i)
    _free(mu_j)
    _free(distances)
    return 1.0 - w1 / graph_distance
