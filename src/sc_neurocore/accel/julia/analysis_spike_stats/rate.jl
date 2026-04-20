# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis_spike_stats/rate

module RateAccel

using Statistics, LinearAlgebra

function instantaneous_rate(binary_train, dt, kernel, sigma_ms)
    binary_train: np.ndarray,
    dt: float = 0.001,
    kernel: str = "gaussian",
    sigma_ms: float = 10.0,
    ) -> np.ndarray
    n = binary_train.size
    sigma_steps = max(1, int(sigma_ms / (dt * 1000)))
    if kernel == "gaussian"
        hw = 3 * sigma_steps
        x = collect(-hw, hw + 1, dtype=np.float64)
        k = exp(-0.5 * (x / sigma_steps) ^ 2)
    elseif kernel == "exponential"
        hw = 5 * sigma_steps
        x = collect(0, hw, dtype=np.float64)
        k = exp(-x / sigma_steps)
    elseif kernel == "rectangular"
        hw = sigma_steps
        k = ones(2 * hw + 1, dtype=np.float64)
    else
        raise ValueError(f"Unknown kernel: {kernel}")
    k /= k.sum() * dt
    return np.convolve(binary_train.astype(np.float64), k, mode="same")
end

function population_rate(trains, dt, sigma_ms)
    trains: list[np.ndarray], dt: float = 0.001, sigma_ms: float = 10.0
    ) -> np.ndarray
    if ! trains
        return collect([])
    min_len = min(t.size for t in trains)
    total = zeros(min_len, dtype=np.float64)
    for t in trains
        total += t[:min_len].astype(np.float64)
    return instantaneous_rate(total, dt=dt, kernel="gaussian", sigma_ms=sigma_ms)
end

function psth(trials, bin_ms, dt)
    trials: list[np.ndarray], bin_ms: float = 10.0, dt: float = 0.001
    ) -> tuple[np.ndarray, np.ndarray]
    if ! trials
        return collect([]), collect([])
    max_len = max(t.size for t in trials)
    bin_steps = max(1, int(bin_ms / (dt * 1000)))
    n_bins = max_len // bin_steps
    if n_bins == 0
        return collect([]), collect([])
    counts = zeros(n_bins, dtype=np.float64)
    for trial in trials
        trimmed = trial[: n_bins * bin_steps]
        if trimmed.size == 0
            continue
        reshaped = trimmed.reshape(-1, bin_steps) if trimmed.size >= bin_steps else trimmed[nothing, :]
        if reshaped.shape[0] <= n_bins
            counts[: reshaped.shape[0]] += reshaped.sum(axis=1)
    rates = counts / (length(trials) * bin_ms / 1000.0)
    centers = (collect(n_bins) + 0.5) * bin_ms
    return rates, centers
end

end # module RateAccel
