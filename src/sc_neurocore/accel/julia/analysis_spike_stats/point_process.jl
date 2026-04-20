# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis_spike_stats/point_process

module PointProcessAccel

using Statistics, LinearAlgebra

function conditional_intensity(binary_train, dt, window_ms)
    binary_train: np.ndarray, dt: float = 0.001, window_ms: float = 50.0
    ) -> np.ndarray
    w = max(1, int(window_ms / (dt * 1000)))
    x = binary_train.astype(np.float64)
    kernel = ones(w) / (w * dt)
    return np.convolve(x, kernel, mode="same")
end

function isi_hazard_function(binary_train, dt, bins)
    binary_train: np.ndarray, dt: float = 0.001, bins: int = 30
    ) -> tuple[np.ndarray, np.ndarray]
    intervals = isi(binary_train, dt)
    if intervals.size < 5
        return collect([]), collect([])
    hist, edges = fit(Histogram, intervals, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    pdf = hist.astype(np.float64) / (intervals.size * (edges[1] - edges[0]))
    survivor = 1.0 - cumsum(pdf) * (edges[1] - edges[0])
    survivor = clamp(survivor, 1e-30, nothing)
    hazard = pdf / survivor
    return hazard, centers
end

function isi_survivor_function(binary_train, dt, bins)
    binary_train: np.ndarray, dt: float = 0.001, bins: int = 30
    ) -> tuple[np.ndarray, np.ndarray]
    intervals = isi(binary_train, dt)
    if intervals.size < 2
        return collect([]), collect([])
    sorted_isi = sort(intervals)
    n = sorted_isi.size
    edges = range(0, sorted_isi[-1], bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    survivor = collect([sum(sorted_isi > t) / n for t in centers])
    return survivor, centers
end

function renewal_density(binary_train, dt, bins)
    binary_train: np.ndarray, dt: float = 0.001, bins: int = 30
    ) -> tuple[np.ndarray, np.ndarray]
    intervals = isi(binary_train, dt)
    if intervals.size < 5
        return collect([]), collect([])
    hist, edges = fit(Histogram, intervals, bins=bins, density=true)
    centers = (edges[:-1] + edges[1:]) / 2
    mean_rate = 1.0 / intervals.mean() if intervals.mean() > 0 else 1.0
    return hist / mean_rate, centers
end

end # module PointProcessAccel
