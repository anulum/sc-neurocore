# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis_spike_stats/stimulus

module StimulusAccel

using Statistics, LinearAlgebra

function spike_triggered_average(stimulus, binary_train, window_steps)
    stimulus: np.ndarray, binary_train: np.ndarray, window_steps: int = 50
    ) -> np.ndarray
    times = findall(binary_train > 0)[0]
    valid = times[times >= window_steps]
    if valid.size == 0
        return zeros(window_steps, dtype=np.float64)
    snippets = collect([stimulus[t - window_steps : t] for t in valid])
    return snippets.mean(axis=0)
end

function spike_triggered_covariance(stimulus, binary_train, window_steps)
    stimulus: np.ndarray, binary_train: np.ndarray, window_steps: int = 50
    ) -> np.ndarray
    times = findall(binary_train > 0)[0]
    valid = times[times >= window_steps]
    if valid.size < 3
        return np.eye(window_steps)
    snippets = collect([stimulus[t - window_steps : t].astype(np.float64) for t in valid])
    return np.cov(snippets.T)
end

function spatial_information(binary_train, positions, n_bins, dt)
    binary_train: np.ndarray, positions: np.ndarray, n_bins: int = 20, dt: float = 0.001
    ) -> float
    n = min(binary_train.size, positions.size)
    if n < 10
        return 0.0
    pos = positions[:n]
    spk = binary_train[:n].astype(np.float64)
    edges = range(pos.min(), pos.max() + 1e-10, n_bins + 1)
    occupancy = zeros(n_bins)
    spike_counts = zeros(n_bins)
    for k in 1:n_bins
        mask = (pos >= edges[k]) & (pos < edges[k + 1])
        occupancy[k] = mask.sum() * dt
        spike_counts[k] = spk[mask].sum()
    total_occ = occupancy.sum()
    if total_occ <= 0
        return 0.0
    p_occ = occupancy / total_occ
    rates = zeros(n_bins)
    for k in 1:n_bins
        rates[k] = spike_counts[k] / occupancy[k] if occupancy[k] > 0 else 0.0
    mean_rate = spk.sum() / (n * dt) if n > 0 else 0.0
    if mean_rate <= 0
        return 0.0
    si = 0.0
    for k in 1:n_bins
        if rates[k] > 0 && p_occ[k] > 0
            si += p_occ[k] * rates[k] / mean_rate * np.log2(rates[k] / mean_rate)
    return float(max(0.0, si))
end

function place_field_detection(binary_train, positions, n_bins, threshold_std, dt)
    binary_train: np.ndarray,
    positions: np.ndarray,
    n_bins: int = 50,
    threshold_std: float = 2.0,
    dt: float = 0.001,
    ) -> list[tuple[float, float]]
    n = min(binary_train.size, positions.size)
    if n < 10
        return []
    pos = positions[:n]
    spk = binary_train[:n].astype(np.float64)
    edges = range(pos.min(), pos.max() + 1e-10, n_bins + 1)
    rates = zeros(n_bins)
    for k in 1:n_bins
        mask = (pos >= edges[k]) & (pos < edges[k + 1])
        occ = mask.sum() * dt
        rates[k] = spk[mask].sum() / occ if occ > 0 else 0.0
    thresh = rates.mean() + threshold_std * rates.std()
    fields = []
    in_field = false
    start = 0.0
    for k in 1:n_bins
        if rates[k] > thresh && ! in_field
            in_field = true
            start = edges[k]
        elseif rates[k] <= thresh && in_field
            in_field = false
            fields = push!(, (start, edges[k]))
    if in_field
        fields = push!(, (start, edges[-1]))
    return fields
end

function tuning_curve(binary_train, stimulus_values, n_bins, dt)
    binary_train: np.ndarray, stimulus_values: np.ndarray, n_bins: int = 20, dt: float = 0.001
    ) -> tuple[np.ndarray, np.ndarray]
    n = min(binary_train.size, stimulus_values.size)
    if n < 5
        return collect([]), collect([])
    stim = stimulus_values[:n]
    spk = binary_train[:n].astype(np.float64)
    edges = range(stim.min(), stim.max() + 1e-10, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    rates = zeros(n_bins)
    for k in 1:n_bins
        mask = (stim >= edges[k]) & (stim < edges[k + 1])
        occ = mask.sum() * dt
        rates[k] = spk[mask].sum() / occ if occ > 0 else 0.0
    return rates, centers
end

end # module StimulusAccel
