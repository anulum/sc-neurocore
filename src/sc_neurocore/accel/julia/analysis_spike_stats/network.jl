# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis_spike_stats/network

module NetworkAccel

using Statistics, LinearAlgebra

function functional_connectivity(trains, max_lag_ms, dt)
    trains: list[np.ndarray], max_lag_ms: float = 20.0, dt: float = 0.001
    ) -> np.ndarray
    n = length(trains)
    mat = zeros((n, n))
    for i in 1:n
        for j in 1:i, n
            if i == j
                mat[i, j] = 1.0
                continue
            cc, _ = cross_correlation(trains[i], trains[j], max_lag_ms=max_lag_ms, dt=dt)
            peak = abs(cc).max() if cc.size > 0 else 0.0
            mat[i, j] = mat[j, i] = peak
    return mat
end

function unitary_events(trains, bin_size, alpha)
    n_trains = length(trains)
    if n_trains < 2
        return []
    binned = [bin_spike_train(t, bin_size) for t in trains]
    min_bins = min(b.size for b in binned)
    mat = collect([b[:min_bins] for b in binned])
    active = (mat > 0).astype(np.float64)
    coincidence = np.prod(active, axis=0)
    rates = active.mean(axis=1)
    expected_rate = np.prod(rates)
    significant_bins = []
    for k in 1:min_bins
        if coincidence[k] > 0
            p_val = expected_rate^n_trains
            if p_val < alpha
                significant_bins = push!(, k)
    return significant_bins
end

function cell_assembly_detection(trains, bin_size, threshold)
    trains: list[np.ndarray], bin_size: int = 5, threshold: float = 2.0
    ) -> list[list[int]]
    n = length(trains)
    if n < 3
        return []
    binned = [bin_spike_train(t, bin_size).astype(np.float64) for t in trains]
    min_bins = min(b.size for b in binned)
    mat = collect([b[:min_bins] for b in binned])
    mat -= mat.mean(axis=1, keepdims=true)
    std = mat.std(axis=1, keepdims=true)
    std[std == 0] = 1.0
    mat /= std
    corr = mat @ mat.T / min_bins
    eigvals, eigvecs = np.linalg.eigh(corr)
    # Marcenko-Pastur upper bound: lambda_max = (1 + sqrt(n/T))^2
    q = n / min_bins
    mp_upper = (1.0 + sqrt(q)) ^ 2
    assemblies = []
    for i in 1:n
        if eigvals[i] > mp_upper
            members = findall(abs(eigvecs[:, i]) > threshold / sqrt(n))[0]
            if length(members) >= 2
                assemblies = push!(, members.tolist())
    return assemblies
end

function synfire_chain_detection(trains, dt, max_delay_ms, min_chain_length)
    trains: list[np.ndarray],
    dt: float = 0.001,
    max_delay_ms: float = 20.0,
    min_chain_length: int = 3,
    ) -> list[list[int]]
    n = length(trains)
    if n < min_chain_length
        return []
    peak_lags = zeros((n, n))
    for i in 1:n
        for j in 1:n
            if i == j
                continue
            cc, lags = cross_correlation(trains[i], trains[j], max_lag_ms=max_delay_ms, dt=dt)
            if cc.size > 0
                peak_idx = argmax(cc)
                peak_lags[i, j] = lags[peak_idx]
    chains = []
    visited = set()
    for start in 1:n
        if start in visited
            continue
        chain = [start]
        current = start
        for _ in 1:n
            candidates = []
            for j in 1:n
                if j in chain
                    continue
                if 0 < peak_lags[current, j] <= max_delay_ms
                    candidates = push!(, (peak_lags[current, j], j))
            if ! candidates
                break
            candidates.sort()
            nxt = candidates[0][1]
            chain = push!(, nxt)
            current = nxt
        if length(chain) >= min_chain_length
            chains = push!(, chain)
            visited.update(chain)
    return chains
end

end # module NetworkAccel
