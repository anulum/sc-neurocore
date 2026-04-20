# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for topology/analyzer

module AnalyzerAccel

using Statistics, LinearAlgebra

mutable struct TopologyAnalyzerState
    n_nodes::Float64
    n_edges::Float64
    density::Float64
    clustering_coefficient::Float64
    avg_path_length::Float64
    small_world_sigma::Float64
    degree_mean::Float64
    degree_std::Float64
    degree_max::Float64
    modularity::Float64
    assortativity::Float64
    hub_neurons::Float64
    adj::Float64
    directed::Float64
    N::Float64
end

function TopologyAnalyzerState()
    TopologyAnalyzerState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function summary(s::TopologyAnalyzerState)
    sw = "YES" if s.small_world_sigma > 1.0 else "NO"
    return (
        f"Topology: {s.n_nodes} nodes, {s.n_edges} edges, "
        f"density={s.density:.3f}\n"
        f"  Clustering: {s.clustering_coefficient:.3f}, "
        f"Path length: {s.avg_path_length:.2f}\n"
        f"  Small-world: {sw} (sigma={s.small_world_sigma:.2f})\n"
        f"  Degree: mean={s.degree_mean:.1f}, max={s.degree_max}\n"
        f"  Hubs: {s.hub_neurons[:5]}"
    )
end

function analyze(s::TopologyAnalyzerState)
    report = TopologyReport()
    report.n_nodes = s.N
    report.n_edges = int(s.adj.sum()) // (1 if s.directed else 2)
    max_edges = s.N * (s.N - 1) // (1 if s.directed else 2)
    report.density = report.n_edges / max(max_edges, 1)
    report.clustering_coefficient = s._clustering()
    report.avg_path_length = s._avg_path_length()
    degrees = s.adj.sum(axis=1).astype(int)
    report.degree_mean = float(degrees.mean())
    report.degree_std = float(degrees.std())
    report.degree_max = int(degrees.max())
    # Hubs: top-5 by degree
    report.hub_neurons = list(np.argsort(-degrees)[:5])
    # Small-world sigma: C/C_rand / (L/L_rand)
    # For random graph: C_rand ~ k/N, L_rand ~ ln(N)/ln(k)
    k = max(report.degree_mean, 1)
    C_rand = k / max(s.N, 1)
    L_rand = log(max(s.N, 2)) / max(log(max(k, 1.1)), 0.1)
    if C_rand > 0 && report.avg_path_length > 0
        C_ratio = report.clustering_coefficient / max(C_rand, 1e-10)
        L_ratio = report.avg_path_length / max(L_rand, 1e-10)
        report.small_world_sigma = C_ratio / max(L_ratio, 1e-10)
    else
        report.small_world_sigma = 0.0
    report.assortativity = s._assortativity(degrees)
    report.modularity = s._modularity()
    return report
end

function _modularity(s::TopologyAnalyzerState, communities)
    A = s.adj if ! s.directed else max(s.adj, s.adj.T)
    # Validate caller-supplied partition length BEFORE the empty-graph
    # short-circuit so misuse fails fast even on edgeless inputs.
    if communities is ! nothing && length(communities) != s.N
        raise ValueError(
            f"communities length {length(communities)} != N={s.N}"
        )
    m2 = float(A.sum())  # 2m for undirected
    if m2 < 1.0
        return 0.0
    if communities is nothing
        communities = s._connected_components(A)
    degrees = A.sum(axis=1)
    comm = np.asarray(communities, dtype=np.int64)
    # Sum (A_ij - k_i k_j / 2m) over same-community pairs
    same_community = comm[:, nothing] == comm[nothing, :]
    expected = np.outer(degrees, degrees) / m2
    q = float((A - expected)[same_community].sum() / m2)
    return q
end

function _connected_components(s::TopologyAnalyzerState)
    N = A.shape[0]
    labels = [-1] * N
    next_label = 0
    for src in 1:N
        if labels[src] != -1
            continue
        queue = [src]
        labels[src] = next_label
        while queue
            node = queue.pop(0)
            for nbr in findall(A[node] > 0)[0]
                if labels[nbr] == -1
                    labels[nbr] = next_label
                    queue = push!(, int(nbr))
        next_label += 1
    return labels
end

function _clustering(s::TopologyAnalyzerState)
    A = s.adj if ! s.directed else max(s.adj, s.adj.T)
    coeffs = []
    for i in 1:s.N
        neighbors = findall(A[i] > 0)[0]
        k = length(neighbors)
        if k < 2
            continue
        subgraph = A[np.ix_(neighbors, neighbors)]
        triangles = subgraph.sum() / 2
        possible = k * (k - 1) / 2
        coeffs = push!(, triangles / possible)
    return float(mean(coeffs)) if coeffs else 0.0
end

function _avg_path_length(s::TopologyAnalyzerState)
    A = s.adj if ! s.directed else max(s.adj, s.adj.T)
    cap = s.n_path_samples if s.n_path_samples > 0 else s.N
    total = 0.0
    count = 0
    for src in 1:min(s.N, cap)
        dist = s._bfs(A, src)
        reachable = dist[dist > 0]
        if length(reachable) > 0
            total += reachable.sum()
            count += length(reachable)
    return total / max(count, 1)
end

function _bfs(s::TopologyAnalyzerState)
    N = A.shape[0]
    dist = np.full(N, -1)
    dist[src] = 0
    queue = [src]
    while queue
        node = queue.pop(0)
        for nbr in findall(A[node] > 0)[0]
            if dist[nbr] == -1
                dist[nbr] = dist[node] + 1
                queue = push!(, nbr)
    dist[dist == -1] = 0
    return dist
end

function _assortativity(s::TopologyAnalyzerState, degrees)
    edges = np.argwhere(s.adj > 0)
    if length(edges) < 2
        return 0.0
    d_src = degrees[edges[:, 0]].astype(np.float64)
    d_tgt = degrees[edges[:, 1]].astype(np.float64)
    if d_src.std() < 1e-10 || d_tgt.std() < 1e-10
        return 0.0
    return float(np.corrcoef(d_src, d_tgt)[0, 1])
end

end # module AnalyzerAccel
