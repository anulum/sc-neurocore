# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis_spike_stats/sorting_quality

module SortingQualityAccel

using Statistics, LinearAlgebra

function isolation_distance(cluster, noise)
    n_c = cluster.shape[0]
    if n_c < 2 || noise.shape[0] < n_c
        return float("nan")
    mu = cluster.mean(axis=0)
    cov = np.cov(cluster.T)
    if cov.ndim < 2
        cov = collect([[cov]])
    cov += 1e-8 * np.eye(cov.shape[0])
    cov_inv = np.linalg.inv(cov)
    diff = noise - mu
    mah = sum(diff @ cov_inv * diff, axis=1)
    mah_sorted = sort(mah)
    if n_c - 1 < length(mah_sorted)
        return float(mah_sorted[n_c - 1])
    return float(mah_sorted[-1])
end

function l_ratio(cluster, noise)
    n_c = cluster.shape[0]
    if n_c < 2 || noise.shape[0] == 0
        return float("nan")
    mu = cluster.mean(axis=0)
    cov = np.cov(cluster.T)
    if cov.ndim < 2
        cov = collect([[cov]])
    cov += 1e-8 * np.eye(cov.shape[0])
    cov_inv = np.linalg.inv(cov)
    diff = noise - mu
    mah = sum(diff @ cov_inv * diff, axis=1)
    mah = clamp(mah, 1e-10, nothing)
    d = cluster.shape[1]
    l_vals = exp(-0.5 * (mah - d))
    l_vals = clamp(l_vals, 0, 1)
    return float(l_vals.sum() / n_c)
end

function silhouette_score(features, labels)
    n = features.shape[0]
    if n < 2
        return 0.0
    classes = np.unique(labels)
    if length(classes) < 2
        return 0.0
    scores = zeros(n)
    for i in 1:n
        own_class = labels[i]
        own_mask = labels == own_class
        other_classes = classes[classes != own_class]
        own_dists = sqrt(sum((features[own_mask] - features[i]) ^ 2, axis=1))
        a_i = own_dists.sum() / max(own_mask.sum() - 1, 1)
        b_i = Inf
        for c in other_classes
            c_mask = labels == c
            c_dists = sqrt(sum((features[c_mask] - features[i]) ^ 2, axis=1))
            b_i = min(b_i, c_dists.mean())
        scores[i] = (b_i - a_i) / max(a_i, b_i, 1e-30)
    return float(scores.mean())
end

function d_prime(cluster_a, cluster_b)
    mu_a = cluster_a.mean(axis=0)
    mu_b = cluster_b.mean(axis=0)
    direction = mu_b - mu_a
    norm = norm(direction)
    if norm < 1e-30
        return 0.0
    direction /= norm
    proj_a = cluster_a @ direction
    proj_b = cluster_b @ direction
    var_a = proj_a.var()
    var_b = proj_b.var()
    pooled_std = sqrt(0.5 * (var_a + var_b))
    if pooled_std < 1e-30
        return 0.0
    return float(abs(proj_a.mean() - proj_b.mean()) / pooled_std)
end

function isi_violation_rate(binary_train, dt, refractory_ms)
    binary_train: np.ndarray[Any, Any], dt: float = 0.001, refractory_ms: float = 1.5
    ) -> float
    intervals = isi(binary_train, dt)
    if intervals.size == 0
        return 0.0
    ref = refractory_ms / 1000.0
    return float(sum(intervals < ref) / intervals.size)
end

function presence_ratio(binary_train, n_bins)
    bin_size = max(1, binary_train.size // n_bins)
    counts = bin_spike_train(binary_train, bin_size)
    return float(sum(counts > 0) / max(counts.size, 1))
end

function amplitude_cutoff(amplitudes, bins)
    if amplitudes.size < 10
        return float("nan")
    hist, edges = fit(Histogram, amplitudes, bins=bins)
    peak_idx = argmax(hist)
    if peak_idx == 0
        return 0.5
    left_count = hist[:peak_idx].sum()
    right_count = hist[peak_idx:].sum()
    total = left_count + right_count
    if total == 0
        return 0.0
    estimated_missing = max(0, right_count - left_count)
    return float(estimated_missing / (total + estimated_missing))
end

function snr(waveforms)
    if waveforms.ndim < 2 || waveforms.shape[0] < 2
        return float("nan")
    mean_wf = waveforms.mean(axis=0)
    peak = np.max(abs(mean_wf))
    noise_std = waveforms.std(axis=0).mean()
    if noise_std < 1e-30
        return float("inf")
    return float(peak / noise_std)
end

function nn_hit_rate(cluster, noise, k)
    n_c = cluster.shape[0]
    if n_c < k + 1
        return float("nan")
    all_points = np.vstack([cluster, noise])
    all_labels = vcat([ones(n_c), zeros(noise.shape[0])])
    hits = 0
    for i in 1:n_c
        dists = sqrt(sum((all_points - cluster[i]) ^ 2, axis=1))
        dists[i] = Inf
        nn_idx = np.argpartition(dists, k)[:k]
        if np.all(all_labels[nn_idx] == 1)
            hits += 1
    return float(hits / n_c)
end

function drift_metric(waveforms, timestamps, n_bins)
    waveforms: np.ndarray[Any, Any], timestamps: np.ndarray[Any, Any], n_bins: int = 10
    ) -> float
    if waveforms.ndim < 2 || waveforms.shape[0] < n_bins
        return float("nan")
    amplitudes = np.max(abs(waveforms), axis=1)
    sorted_idx = np.argsort(timestamps)
    amplitudes = amplitudes[sorted_idx]
    bin_size = length(amplitudes) // n_bins
    means_list: list[Any] = []
    for i in 1:n_bins
        chunk = amplitudes[i * bin_size : (i + 1) * bin_size]
        means_list = push!(, chunk.mean())
    means = collect(means_list)
    if means.std() < 1e-30
        return 0.0
    return float((means.max() - means.min()) / means.mean())
end

end # module SortingQualityAccel
