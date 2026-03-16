# SPDX-License-Identifier: AGPL-3.0-or-later
"""Spike sorting quality metrics."""

from __future__ import annotations

import numpy as np

from .basic import isi, bin_spike_train


def isolation_distance(cluster: np.ndarray, noise: np.ndarray) -> float:
    """Isolation distance. Harris et al. 2001.

    Mahalanobis distance at which the number of noise points equals cluster size.
    cluster: (n_cluster, n_features). noise: (n_noise, n_features).
    """
    n_c = cluster.shape[0]
    if n_c < 2 or noise.shape[0] < n_c:
        return float("nan")
    mu = cluster.mean(axis=0)
    cov = np.cov(cluster.T)
    if cov.ndim < 2:
        cov = np.array([[cov]])
    cov += 1e-8 * np.eye(cov.shape[0])
    cov_inv = np.linalg.inv(cov)
    diff = noise - mu
    mah = np.sum(diff @ cov_inv * diff, axis=1)
    mah_sorted = np.sort(mah)
    if n_c - 1 < len(mah_sorted):
        return float(mah_sorted[n_c - 1])
    return float(mah_sorted[-1])


def l_ratio(cluster: np.ndarray, noise: np.ndarray) -> float:
    """L-ratio. Schmitzer-Torbert et al. 2005.

    Sum of inverse-chi2 CDF values for noise Mahalanobis distances, normalized by cluster size.
    Approximated here as sum(1/mah_dist) / n_cluster.
    """
    n_c = cluster.shape[0]
    if n_c < 2 or noise.shape[0] == 0:
        return float("nan")
    mu = cluster.mean(axis=0)
    cov = np.cov(cluster.T)
    if cov.ndim < 2:
        cov = np.array([[cov]])
    cov += 1e-8 * np.eye(cov.shape[0])
    cov_inv = np.linalg.inv(cov)
    diff = noise - mu
    mah = np.sum(diff @ cov_inv * diff, axis=1)
    mah = np.clip(mah, 1e-10, None)
    d = cluster.shape[1]
    l_vals = np.exp(-0.5 * (mah - d))
    l_vals = np.clip(l_vals, 0, 1)
    return float(l_vals.sum() / n_c)


def silhouette_score(features: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette score. Rousseeuw 1987.

    Measures cluster separation: s_i = (b_i - a_i) / max(a_i, b_i).
    """
    n = features.shape[0]
    if n < 2:
        return 0.0
    classes = np.unique(labels)
    if len(classes) < 2:
        return 0.0
    scores = np.zeros(n)
    for i in range(n):
        own_class = labels[i]
        own_mask = labels == own_class
        other_classes = classes[classes != own_class]
        own_dists = np.sqrt(np.sum((features[own_mask] - features[i]) ** 2, axis=1))
        a_i = own_dists.sum() / max(own_mask.sum() - 1, 1)
        b_i = np.inf
        for c in other_classes:
            c_mask = labels == c
            c_dists = np.sqrt(np.sum((features[c_mask] - features[i]) ** 2, axis=1))
            b_i = min(b_i, c_dists.mean())
        scores[i] = (b_i - a_i) / max(a_i, b_i, 1e-30)
    return float(scores.mean())


def d_prime(cluster_a: np.ndarray, cluster_b: np.ndarray) -> float:
    """d-prime (sensitivity index) between two clusters. Green & Swets 1966.

    Uses first principal axis for projection.
    """
    mu_a = cluster_a.mean(axis=0)
    mu_b = cluster_b.mean(axis=0)
    direction = mu_b - mu_a
    norm = np.linalg.norm(direction)
    if norm < 1e-30:
        return 0.0
    direction /= norm
    proj_a = cluster_a @ direction
    proj_b = cluster_b @ direction
    var_a = proj_a.var()
    var_b = proj_b.var()
    pooled_std = np.sqrt(0.5 * (var_a + var_b))
    if pooled_std < 1e-30:
        return 0.0
    return float(abs(proj_a.mean() - proj_b.mean()) / pooled_std)


def isi_violation_rate(binary_train: np.ndarray, dt: float = 0.001,
                       refractory_ms: float = 1.5) -> float:
    """ISI violation rate: fraction of ISIs below refractory period. Hill et al. 2011."""
    intervals = isi(binary_train, dt)
    if intervals.size == 0:
        return 0.0
    ref = refractory_ms / 1000.0
    return float(np.sum(intervals < ref) / intervals.size)


def presence_ratio(binary_train: np.ndarray, n_bins: int = 100) -> float:
    """Presence ratio: fraction of time bins containing at least one spike. IBL 2019."""
    bin_size = max(1, binary_train.size // n_bins)
    counts = bin_spike_train(binary_train, bin_size)
    return float(np.sum(counts > 0) / max(counts.size, 1))


def amplitude_cutoff(amplitudes: np.ndarray, bins: int = 100) -> float:
    """Amplitude cutoff estimate. Hill et al. 2011.

    Fraction of spikes estimated to be missing below the amplitude histogram peak.
    """
    if amplitudes.size < 10:
        return float("nan")
    hist, edges = np.histogram(amplitudes, bins=bins)
    peak_idx = np.argmax(hist)
    if peak_idx == 0:
        return 0.5
    left_count = hist[:peak_idx].sum()
    right_count = hist[peak_idx:].sum()
    total = left_count + right_count
    if total == 0:
        return 0.0
    estimated_missing = max(0, right_count - left_count)
    return float(estimated_missing / (total + estimated_missing))


def snr(waveforms: np.ndarray) -> float:
    """Signal-to-noise ratio of spike waveforms. Suner et al. 2005.

    waveforms: (n_spikes, n_samples). SNR = peak_amplitude / noise_std.
    """
    if waveforms.ndim < 2 or waveforms.shape[0] < 2:
        return float("nan")
    mean_wf = waveforms.mean(axis=0)
    peak = np.max(np.abs(mean_wf))
    noise_std = waveforms.std(axis=0).mean()
    if noise_std < 1e-30:
        return float("inf")
    return float(peak / noise_std)


def nn_hit_rate(cluster: np.ndarray, noise: np.ndarray, k: int = 4) -> float:
    """Nearest-neighbor hit rate. Chung et al. 2017.

    Fraction of cluster points whose k nearest neighbors are also in the cluster.
    """
    n_c = cluster.shape[0]
    if n_c < k + 1:
        return float("nan")
    all_points = np.vstack([cluster, noise])
    all_labels = np.concatenate([np.ones(n_c), np.zeros(noise.shape[0])])
    hits = 0
    for i in range(n_c):
        dists = np.sqrt(np.sum((all_points - cluster[i]) ** 2, axis=1))
        dists[i] = np.inf
        nn_idx = np.argpartition(dists, k)[:k]
        if np.all(all_labels[nn_idx] == 1):
            hits += 1
    return float(hits / n_c)


def drift_metric(waveforms: np.ndarray, timestamps: np.ndarray, n_bins: int = 10) -> float:
    """Waveform drift metric. IBL 2019.

    Measures change in mean waveform amplitude over time.
    """
    if waveforms.ndim < 2 or waveforms.shape[0] < n_bins:
        return float("nan")
    amplitudes = np.max(np.abs(waveforms), axis=1)
    sorted_idx = np.argsort(timestamps)
    amplitudes = amplitudes[sorted_idx]
    bin_size = len(amplitudes) // n_bins
    means = []
    for i in range(n_bins):
        chunk = amplitudes[i * bin_size:(i + 1) * bin_size]
        means.append(chunk.mean())
    means = np.array(means)
    if means.std() < 1e-30:
        return 0.0
    return float((means.max() - means.min()) / means.mean())
