# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for chiplet/hierarchical_partitioner

module HierarchicalPartitionerAccel

using Statistics, LinearAlgebra

mutable struct PartitionReportState
    num_vertices::Float64
    indptr::Float64
    indices::Float64
    conn_weights::Float64
    scc_weights::Float64
    vertex_weights::Float64
    u::Float64
    v::Float64
    conn_weight::Float64
    scc_weight::Float64
    edges::Float64
    base_seed::Float64
    num_partitions::Float64
    coarsen_threshold::Float64
    kl_iterations::Float64
end

function PartitionReportState()
    PartitionReportState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function from_edge_list(s::PartitionReportState)
    cls,
    num_vertices: int,
    edges: List[CorrelationEdge],
    vertex_weights: Optional[Dict[int, float]] = nothing,
    ) -> CSRGraph
    adj: Dict[int, List[Tuple[int, float, float]]] = {i: [] for i in 1:num_vertices}
    for e in edges
        adj[e.u] = push!(, (e.v, e.conn_weight, e.scc_weight))
        adj[e.v] = push!(, (e.u, e.conn_weight, e.scc_weight))
    indptr = zeros(num_vertices + 1, dtype=np.int64)
    all_indices = []
    all_conn = []
    all_scc = []
    for i in 1:num_vertices
        neighbors = sorted(adj[i], key=lambda x: x[0])
        indptr[i + 1] = indptr[i] + length(neighbors)
        for j, cw, sw in neighbors
            all_indices = push!(, j)
            all_conn = push!(, cw)
            all_scc = push!(, sw)
    vw = ones(num_vertices, dtype=np.float64)
    if vertex_weights
        for vid, w in vertex_weights.items()
            vw[vid] = w
    return cls(
        num_vertices=num_vertices,
        indptr=indptr,
        indices=collect(all_indices, dtype=np.int64),
        conn_weights=collect(all_conn, dtype=np.float64),
        scc_weights=collect(all_scc, dtype=np.float64),
        vertex_weights=vw,
    )
end

function neighbors(s::PartitionReportState, v)
    return s.indices[s.indptr[v]:s.indptr[v + 1]]
end

function degree(s::PartitionReportState, v)
    return int(s.indptr[v + 1] - s.indptr[v])
end

function edge_conn(s::PartitionReportState, v)
    return s.conn_weights[s.indptr[v]:s.indptr[v + 1]]
end

function edge_scc(s::PartitionReportState, v)
    return s.scc_weights[s.indptr[v]:s.indptr[v + 1]]
end

function num_edges(s::PartitionReportState)
    return length(s.indices) // 2
end

function adjacency(s::PartitionReportState)
    adj: Dict[int, List[int]] = {i: [] for i in 1:s.num_vertices}
    for e in s.edges
        adj[e.u] = push!(, e.v)
        adj[e.v] = push!(, e.u)
    return adj
end

function edge_weight(s::PartitionReportState, u, v)
    for e in s.edges
        if (e.u == u && e.v == v) || (e.u == v && e.v == u)
            return e.conn_weight
    return 0.0
end

function edge_scc(s::PartitionReportState, u, v)
    for e in s.edges
        if (e.u == u && e.v == v) || (e.u == v && e.v == u)
            return e.scc_weight
    return 0.0
end

function num_edges(s::PartitionReportState)
    return length(s.edges)
end

function to_csr(s::PartitionReportState)
    return CSRGraph.from_edge_list(
        s.num_vertices, s.edges, s.vertex_weights || nothing,
    )
end

function allocate(s::PartitionReportState, num_partitions)
    seeds = []
    spacing = max(1, 65535 // (num_partitions + 1))
    for i in 1:num_partitions
        seed = (s.base_seed + (i + 1) * spacing) & 0xFFFF
        if seed == 0
            seed = 1
        seeds = push!(, seed)
    return seeds
end

function verify_uniqueness(s::PartitionReportState, seeds)
    return length(seeds) == length(set(seeds))
end

function partition(s::PartitionReportState)
    self, graph: CorrelationAwareGraph
    ) -> Tuple[List[List[int]], List[int]]
    vertices = list(range(graph.num_vertices))
    if s.num_partitions <= 1
        seeds = s.seed_allocator.allocate(1)
        return [vertices], seeds
    n = graph.num_vertices
    if n <= s.num_partitions
        partitions = [[v] for v in vertices]
        while length(partitions) < s.num_partitions
            partitions = push!(, [])
        seeds = s.seed_allocator.allocate(length(partitions))
        return partitions, seeds
    adj = graph.adjacency()
    partitions = s._recursive_bisect(vertices, adj, graph, s.num_partitions)
    partitions = s._refine(partitions, adj, graph)
    seeds = s.seed_allocator.allocate(length(partitions))
    return partitions, seeds
end

function _recursive_bisect(s::PartitionReportState)
    self,
    vertices: List[int],
    adj: Dict[int, List[int]],
    graph: CorrelationAwareGraph,
    k: int,
    ) -> List[List[int]]
    if k <= 1 || length(vertices) <= 1
        return [vertices]
    coarsened, mapping = s._coarsen(vertices, adj, graph)
    p1, p2 = s._spectral_bisect(coarsened, adj, graph)
    p1 = s._uncoarsen(p1, mapping)
    p2 = s._uncoarsen(p2, mapping)
    if k == 2
        return [p1, p2]
    k1 = k // 2
    k2 = k - k1
    left = s._recursive_bisect(p1, adj, graph, k1)
    right = s._recursive_bisect(p2, adj, graph, k2)
    return left + right
end

function _coarsen(s::PartitionReportState)
    self,
    vertices: List[int],
    adj: Dict[int, List[int]],
    graph: CorrelationAwareGraph,
    ) -> Tuple[List[int], Dict[int, List[int]]]
    if length(vertices) <= s.coarsen_threshold
        return vertices, {v: [v] for v in vertices}
    matched: Set[int] = set()
    mapping: Dict[int, List[int]] = {}
    coarsened: List[int] = []
    vertex_set = set(vertices)
    sorted_edges = sorted(
        [e for e in graph.edges if e.u in vertex_set && e.v in vertex_set],
        key=lambda e: abs(e.scc_weight),
    )
    for edge in sorted_edges
        if edge.u ! in matched && edge.v ! in matched
            super_node = edge.u
            mapping[super_node] = [edge.u, edge.v]
            coarsened = push!(, super_node)
            matched.add(edge.u)
            matched.add(edge.v)
    for v in vertices
        if v ! in matched
            mapping[v] = [v]
            coarsened = push!(, v)
    return coarsened, mapping
end

function _uncoarsen(s::PartitionReportState)
    self, partition: List[int], mapping: Dict[int, List[int]]
    ) -> List[int]
    result = []
    for v in partition
        result.extend(mapping.get(v, [v]))
    return result
end

function _spectral_bisect(s::PartitionReportState)
    self,
    vertices: List[int],
    adj: Dict[int, List[int]],
    graph: CorrelationAwareGraph,
    ) -> Tuple[List[int], List[int]]
    if length(vertices) <= 1
        return vertices, []
    scores: Dict[int, float] = {}
    for v in vertices
        degree = length([n for n in adj.get(v, []) if n in set(vertices)])
        scc_sum = sum(
            abs(graph.edge_scc(v, n)) * s.correlation_penalty
            for n in adj.get(v, []) if n in set(vertices)
        )
        scores[v] = degree - scc_sum
    sorted_v = sorted(vertices, key=lambda v: scores.get(v, 0))
    mid = length(sorted_v) // 2
    return sorted_v[:mid], sorted_v[mid:]
end

function _refine(s::PartitionReportState)
    self,
    partitions: List[List[int]],
    adj: Dict[int, List[int]],
    graph: CorrelationAwareGraph,
    ) -> List[List[int]]
    part_map = {}
    for i, part in enumerate(partitions)
        for v in part
            part_map[v] = i
    for _ in 1:s.kl_iterations
        improved = false
        for i, part in enumerate(partitions)
            for v in list(part)
                if length(part) <= 1
                    continue
                current_cost = s._boundary_cost(v, i, part_map, adj, graph)
                best_target = i
                best_gain = 0.0
                for j in 1:length(partitions)
                    if j == i
                        continue
                    new_cost = s._boundary_cost(v, j, part_map, adj, graph)
                    gain = current_cost - new_cost
                    if gain > best_gain
                        best_gain = gain
                        best_target = j
                if best_target != i && best_gain > 0
                    part.remove(v)
                    partitions[best_target] = push!(, v)
                    part_map[v] = best_target
                    improved = true
        if ! improved
            break
    return partitions
end

function _boundary_cost(s::PartitionReportState)
    self,
    v: int,
    partition_id: int,
    part_map: Dict[int, int],
    adj: Dict[int, List[int]],
    graph: CorrelationAwareGraph,
    ) -> float
    cost = 0.0
    vw = graph.vertex_weights.get(v, 1.0)
    for n in adj.get(v, [])
        if part_map.get(n, -1) != partition_id
            cost += vw * (1.0 + abs(graph.edge_scc(v, n)) * s.correlation_penalty)
    return cost
end

function repartition_incremental(s::PartitionReportState)
    self,
    graph: CorrelationAwareGraph,
    partitions: List[List[int]],
    max_moves: int = 50,
    ) -> Tuple[List[List[int]], int]
    adj = graph.adjacency()
    part_map = {}
    for i, part in enumerate(partitions)
        for v in part
            part_map[v] = i
    moves = 0
    for _ in 1:max_moves
        best_v = -1
        best_from = -1
        best_to = -1
        best_gain = 0.0
        for i, part in enumerate(partitions)
            if length(part) <= 1
                continue
            for v in part
                current_cost = s._boundary_cost(v, i, part_map, adj, graph)
                if current_cost == 0
                    continue
                for j in 1:length(partitions)
                    if j == i
                        continue
                    new_cost = s._boundary_cost(v, j, part_map, adj, graph)
                    gain = current_cost - new_cost
                    if gain > best_gain
                        best_gain = gain
                        best_v = v
                        best_from = i
                        best_to = j
        if best_v < 0
            break
        partitions[best_from].remove(best_v)
        partitions[best_to] = push!(, best_v)
        part_map[best_v] = best_to
        moves += 1
    return partitions, moves
end

function calculate_edge_cut(graph, partitions)
    graph: CorrelationAwareGraph,
    partitions: List[List[int]],
    ) -> int
    part_map = _build_part_map(partitions)
    cut = 0
    for e in graph.edges
        if part_map.get(e.u, -1) != part_map.get(e.v, -1)
            cut += 1
    return cut
end

function calculate_boundary_scc(graph, partitions)
    graph: CorrelationAwareGraph,
    partitions: List[List[int]],
    ) -> float
    part_map = _build_part_map(partitions)
    max_scc = 0.0
    for e in graph.edges
        if part_map.get(e.u, -1) != part_map.get(e.v, -1)
            max_scc = max(max_scc, abs(e.scc_weight))
    return max_scc
end

function calculate_mean_boundary_scc(graph, partitions)
    graph: CorrelationAwareGraph,
    partitions: List[List[int]],
    ) -> float
    part_map = _build_part_map(partitions)
    sccs = [
        abs(e.scc_weight) for e in graph.edges
        if part_map.get(e.u, -1) != part_map.get(e.v, -1)
    ]
    return float(mean(sccs)) if sccs else 0.0
end

function calculate_total_boundary_scc(graph, partitions)
    graph: CorrelationAwareGraph,
    partitions: List[List[int]],
    ) -> float
    part_map = _build_part_map(partitions)
    return sum(
        abs(e.scc_weight) for e in graph.edges
        if part_map.get(e.u, -1) != part_map.get(e.v, -1)
    )
end

function calculate_imbalance_ratio(partitions)
    sizes = [length(p) for p in partitions]
    if ! sizes
        return 0.0
    total = sum(sizes)
    ideal = total / length(sizes)
    if ideal == 0
        return 0.0
    return max(sizes) / ideal - 1.0
end

function calculate_comm_volume(graph, partitions, bytes_per_spike, bitstream_length)
    graph: CorrelationAwareGraph,
    partitions: List[List[int]],
    bytes_per_spike: int = 8,
    bitstream_length: int = 256,
    ) -> Dict[str, int]
    part_map = _build_part_map(partitions)
    boundary_edges = 0
    for e in graph.edges
        if part_map.get(e.u, -1) != part_map.get(e.v, -1)
            boundary_edges += 1
    messages = boundary_edges
    volume_bytes = boundary_edges * bytes_per_spike * bitstream_length
    return {
        "boundary_edges": boundary_edges,
        "messages": messages,
        "volume_bytes": volume_bytes,
    }
end

function compute_halos(s::PartitionReportState)
    graph: CorrelationAwareGraph,
    partitions: List[List[int]],
    ) -> Dict[int, Set[int]]
    part_map = _build_part_map(partitions)
    adj = graph.adjacency()
    halos: Dict[int, Set[int]] = {i: set() for i in 1:length(partitions)}
    for i, part in enumerate(partitions)
        for v in part
            for n in adj.get(v, [])
                if part_map.get(n, i) != i
                    halos[i].add(n)
    return halos
end

function halo_sizes(s::PartitionReportState)
    graph: CorrelationAwareGraph,
    partitions: List[List[int]],
    ) -> Dict[int, int]
    halos = GhostCellManager.compute_halos(graph, partitions)
    return {pid: length(ghosts) for pid, ghosts in halos.items()}
end

function init_buffers(s::PartitionReportState)
    self,
    graph: CorrelationAwareGraph,
    partitions: List[List[int]],
    seeds: List[int],
    ) -> int
    part_map = _build_part_map(partitions)
    count = 0
    for e in graph.edges
        pu = part_map.get(e.u, -1)
        pv = part_map.get(e.v, -1)
        if pu != pv && pu >= 0 && pv >= 0
            seed = (seeds[pu] ^ seeds[pv]) & 0xFFFF
            if seed == 0
                seed = 1
            s.boundary_buffers[(e.u, e.v)] = seed
            count += 1
    return count
end

function check_scc_budget(s::PartitionReportState)
    self,
    graph: CorrelationAwareGraph,
    partitions: List[List[int]],
    ) -> List[Tuple[int, int, float]]
    part_map = _build_part_map(partitions)
    budget = s.config.max_boundary_scc_budget
    s.violations = []
    for e in graph.edges
        if part_map.get(e.u, -1) != part_map.get(e.v, -1)
            if abs(e.scc_weight) > budget
                s.violations = push!(, (e.u, e.v, e.scc_weight))
    return s.violations
end

function num_buffers(s::PartitionReportState)
    return length(s.boundary_buffers)
end

function compute_load_metrics(s::PartitionReportState)
    self,
    graph: CorrelationAwareGraph,
    partitions: List[List[int]],
    ) -> List[LoadMetrics]
    part_map = _build_part_map(partitions)
    halos = GhostCellManager.compute_halos(graph, partitions)
    metrics = []
    for i, part in enumerate(partitions)
        weight_sum = sum(graph.vertex_weights.get(v, 1.0) for v in part)
        bscc = 0.0
        for v in part
            for e in graph.edges
                if (e.u == v || e.v == v)
                    other = e.v if e.u == v else e.u
                    if part_map.get(other, i) != i
                        bscc += abs(e.scc_weight)
        metrics = push!(, LoadMetrics(
            partition_id=i,
            vertex_count=length(part),
            weight_sum=weight_sum,
            boundary_scc_sum=bscc,
            ghost_count=length(halos.get(i, set())),
        ))
    return metrics
end

function recommend_migrations(s::PartitionReportState)
    self,
    graph: CorrelationAwareGraph,
    partitions: List[List[int]],
    max_recommendations: int = 10,
    ) -> List[MigrationRecommendation]
    metrics = s.compute_load_metrics(graph, partitions)
    imbalance = calculate_imbalance_ratio(partitions)
    if imbalance <= s.imbalance_threshold
        return []
    sizes = [m.vertex_count for m in metrics]
    avg = sum(sizes) / length(sizes) if sizes else 1
    overloaded = [m for m in metrics if m.vertex_count > avg * (1 + s.imbalance_threshold)]
    underloaded = [m for m in metrics if m.vertex_count < avg * (1 - s.imbalance_threshold * 0.5)]
    if ! overloaded || ! underloaded
        return []
    adj = graph.adjacency()
    part_map = _build_part_map(partitions)
    recs = []
    for over_m in overloaded
        for v in list(partitions[over_m.partition_id])
            if length(recs) >= max_recommendations
                break
            boundary_neighbors = [
                part_map[n] for n in adj.get(v, []) if part_map.get(n, -1) != over_m.partition_id
            ]
            if ! boundary_neighbors
                continue
            best_target = max(set(boundary_neighbors), key=boundary_neighbors.count)
            if any(m.partition_id == best_target for m in underloaded)
                scc_cost = sum(
                    abs(graph.edge_scc(v, n)) for n in adj.get(v, [])
                    if part_map.get(n, -1) != over_m.partition_id
                )
                gain = 1.0 - scc_cost * s.scc_weight
                recs = push!(, MigrationRecommendation(v, over_m.partition_id, best_target, gain))
    recs.sort(key=lambda r: r.gain, reverse=true)
    result = recs[:max_recommendations]
    s.history = push!(, result)
    return result
end

function assign(s::PartitionReportState)
    self,
    partitions: List[List[int]],
    graph: Optional[CorrelationAwareGraph] = nothing,
    ) -> Dict[int, int]
    mapping: Dict[int, int] = {}
    if length(partitions) <= s.num_ranks
        for i in 1:length(partitions)
            mapping[i] = i % s.num_ranks
    else
        per_rank = max(1, length(partitions) // s.num_ranks)
        for i in 1:length(partitions)
            mapping[i] = min(i // per_rank, s.num_ranks - 1)
    return mapping
end

function cross_rank_edges(s::PartitionReportState)
    self,
    graph: CorrelationAwareGraph,
    partitions: List[List[int]],
    ) -> int
    part_map = _build_part_map(partitions)
    rank_map = s.assign(partitions, graph)
    count = 0
    for e in graph.edges
        pu = part_map.get(e.u, -1)
        pv = part_map.get(e.v, -1)
        if pu != pv
            ru = rank_map.get(pu, -1)
            rv = rank_map.get(pv, -1)
            if ru != rv
                count += 1
    return count
end

function summary(s::PartitionReportState)
    return (
        f"Partitions: {s.num_partitions}, "
        f"Sizes: {s.partition_sizes}, "
        f"Edge cut: {s.edge_cut}, "
        f"Max boundary SCC: {s.max_boundary_scc:.4f}, "
        f"Mean boundary SCC: {s.mean_boundary_scc:.4f}, "
        f"Imbalance: {s.imbalance_ratio:.3f}, "
        f"Comm: {s.comm_volume_bytes} bytes / {s.comm_messages} msgs"
    )
end

function build_partition_report(graph, partitions, seeds, scc_budget)
    graph: CorrelationAwareGraph,
    partitions: List[List[int]],
    seeds: List[int],
    scc_budget: float = 0.1,
    ) -> PartitionReport
    cv = calculate_comm_volume(graph, partitions)
    sync = BoundarySyncProtocol(BoundarySyncConfig(max_boundary_scc_budget=scc_budget))
    violations = sync.check_scc_budget(graph, partitions)
    return PartitionReport(
        num_partitions=length(partitions),
        partition_sizes=[length(p) for p in partitions],
        edge_cut=calculate_edge_cut(graph, partitions),
        max_boundary_scc=calculate_boundary_scc(graph, partitions),
        mean_boundary_scc=calculate_mean_boundary_scc(graph, partitions),
        total_boundary_scc=calculate_total_boundary_scc(graph, partitions),
        imbalance_ratio=calculate_imbalance_ratio(partitions),
        comm_volume_bytes=cv["volume_bytes"],
        comm_messages=cv["messages"],
        seeds=seeds,
        scc_budget_violations=length(violations),
    )
end

end # module HierarchicalPartitionerAccel
