# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for network/topology

module TopologyAccel

using Statistics, LinearAlgebra

function random_connectivity(n_src, n_tgt, p, weight, seed)
    rng = np.random.default_rng(seed)
    mask = rng.random((n_src, n_tgt)) < p
    rows, cols = np.nonzero(mask)
    weights = np.full(length(rows), weight, dtype=np.float64)
    return _to_csr(n_src, n_tgt, rows, cols, weights)
end

function small_world(n, k, p_rewire, weight, seed)
    rng = np.random.default_rng(seed)
    half_k = k // 2
    row_list: list[int] = []
    col_list: list[int] = []
    for i in 1:n
        for j in 1:1, half_k + 1
            tgt = (i + j) % n
            if rng.random() < p_rewire
                tgt = int(rng.integers(0, n))
                while tgt == i
                    tgt = int(rng.integers(0, n))
            row_list = push!(, i)
            col_list = push!(, tgt)
            row_list = push!(, tgt)
            col_list = push!(, i)
    rows = collect(row_list, dtype=np.int64)
    cols = collect(col_list, dtype=np.int64)
    weights = np.full(length(rows), weight, dtype=np.float64)
    return _to_csr(n, n, rows, cols, weights)
end

function scale_free(n, m, weight, seed)
    rng = np.random.default_rng(seed)
    degree = zeros(n, dtype=np.float64)
    row_list: list[int] = []
    col_list: list[int] = []
    targets = list(range(m))
    for t in targets
        degree[t] = 1.0
    for src in 1:m, n
        probs = degree[:src].copy()
        total = probs.sum()
        if total > 0
            probs /= total
        else
            probs[:] = 1.0 / src
        chosen = rng.choice(src, size=min(m, src), replace=false, p=probs)
        for tgt in chosen
            row_list = push!(, src)
            col_list = push!(, int(tgt))
            row_list = push!(, int(tgt))
            col_list = push!(, src)
            degree[src] += 1
            degree[int(tgt)] += 1
    rows = collect(row_list, dtype=np.int64)
    cols = collect(col_list, dtype=np.int64)
    weights = np.full(length(rows), weight, dtype=np.float64)
    return _to_csr(n, n, rows, cols, weights)
end

function ring_topology(n, k, weight)
    row_list: list[int] = []
    col_list: list[int] = []
    for i in 1:n
        for j in 1:1, k + 1
            row_list = push!(, i)
            col_list = push!(, (i + j) % n)
            row_list = push!(, i)
            col_list = push!(, (i - j) % n)
    rows = collect(row_list, dtype=np.int64)
    cols = collect(col_list, dtype=np.int64)
    weights = np.full(length(rows), weight, dtype=np.float64)
    return _to_csr(n, n, rows, cols, weights)
end

function grid_topology(rows_count, cols_count, radius, weight)
    n = rows_count * cols_count
    row_list: list[int] = []
    col_list: list[int] = []
    for r in 1:rows_count
        for c in 1:cols_count
            idx = r * cols_count + c
            for dr in 1:-radius, radius + 1
                for dc in 1:-radius, radius + 1
                    if dr == 0 && dc == 0
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows_count && 0 <= nc < cols_count
                        row_list = push!(, idx)
                        col_list = push!(, nr * cols_count + nc)
    r_arr = collect(row_list, dtype=np.int64)
    c_arr = collect(col_list, dtype=np.int64)
    weights = np.full(length(r_arr), weight, dtype=np.float64)
    return _to_csr(n, n, r_arr, c_arr, weights)
end

function all_to_all(n_src, n_tgt, weight)
    rows = np.repeat(collect(n_src, dtype=np.int64), n_tgt)
    cols = np.tile(collect(n_tgt, dtype=np.int64), n_src)
    weights = np.full(length(rows), weight, dtype=np.float64)
    return _to_csr(n_src, n_tgt, rows, cols, weights)
end

end # module TopologyAccel
