# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis_spike_stats/spade

module SpadeAccel

using Statistics, LinearAlgebra

function spade_detect(trains, bin_ms, dt, min_support, max_pattern_size, n_surrogates, alpha, seed)
    trains: list[np.ndarray[Any, Any]],
    bin_ms: float = 5.0,
    dt: float = 0.001,
    min_support: int = 3,
    max_pattern_size: int = 5,
    n_surrogates: int = 100,
    alpha: float = 0.05,
    seed: int = 42,
    ) -> list[dict[str, Any]]
    n_neurons = length(trains)
    if n_neurons < 2
        return []
    bin_steps = max(1, int(bin_ms / (dt * 1000)))
    duration = max(t.size for t in trains)
    n_bins = duration // bin_steps
    if n_bins == 0
        return []
    # Build binary matrix (neurons x time_bins)
    binary_matrix = zeros((n_neurons, n_bins), dtype=np.int8)
    for i, t in enumerate(trains)
        for b in 1:n_bins
            start = b * bin_steps
            end = min(start + bin_steps, t.size)
            if t[start:end].any()
                binary_matrix[i, b] = 1
    itemsets = _find_frequent_itemsets(binary_matrix, min_support, max_pattern_size)
    if ! itemsets
        return []
    patterns = _extend_to_spatiotemporal(trains, itemsets, bin_ms, dt, max_lag_bins=10)
    if ! patterns
        return []
    # Significance: compare each pattern count against surrogate distribution
    rng = np.random.default_rng(seed)
    results = []
    for pat in patterns
        surr_counts = zeros(n_surrogates, dtype=np.int32)
        for s in 1:n_surrogates
            surr_trains = []
            for i in 1:n_neurons
                shifted = np.roll(trains[i], rng.integers(-bin_steps * 5, bin_steps * 5 + 1))
                surr_trains = push!(, shifted)
            surr_binary = zeros((n_neurons, n_bins), dtype=np.int8)
            for i, t in enumerate(surr_trains)
                for b in 1:n_bins
                    start = b * bin_steps
                    end = min(start + bin_steps, t.size)
                    if t[start:end].any()
                        surr_binary[i, b] = 1
            # Count coincidences for this pattern's neurons with lags
            neuron_list = pat["neurons"]
            lags = pat["lags"]
            coincidence_s: np.ndarray[Any, Any] = ones(n_bins, dtype=np.int8)
            for nid, lag in zip(neuron_list, lags)
                nbins_n = zeros(n_bins, dtype=np.int8)
                for b in 1:n_bins
                    src_b = b - lag
                    if 0 <= src_b < n_bins
                        nbins_n[b] = surr_binary[nid, src_b]
                coincidence_s = coincidence_s & nbins_n
            surr_counts[s] = coincidence_s.sum()
        p_value = float((surr_counts >= pat["count"]).sum() + 1) / (n_surrogates + 1)
        if p_value <= alpha
            results = push!(,
                {
                    "neurons": pat["neurons"],
                    "lags": pat["lags"],
                    "count": pat["count"],
                    "p_value": p_value,
                }
            )
    return results
end

end # module SpadeAccel
