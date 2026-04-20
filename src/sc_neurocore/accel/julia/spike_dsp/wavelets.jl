# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for spike_dsp/wavelets

module WaveletsAccel

using Statistics, LinearAlgebra

function spike_wavelet_decompose(spikes, n_scales, base_window)
    spikes: np.ndarray,
    n_scales: int = 4,
    base_window: int = 4,
    ) -> list[np.ndarray]
    if spikes.ndim == 1
        spikes = spikes[:, np.newaxis]
        squeeze = true
    else
        squeeze = false
    T, N = spikes.shape
    scales = []
    for s in 1:n_scales
        window = base_window * (2^s)
        # Moving average at this scale
        smoothed = zeros((T, N), dtype=np.float64)
        for t in 1:T
            start = max(0, t - window + 1)
            smoothed[t] = spikes[start : t + 1].mean(axis=0)
        # Difference between adjacent scales = bandpass
        if s == 0
            band = smoothed
        else
            prev_window = base_window * (2 ^ (s - 1))
            prev_smoothed = zeros((T, N), dtype=np.float64)
            for t in 1:T
                start = max(0, t - prev_window + 1)
                prev_smoothed[t] = spikes[start : t + 1].mean(axis=0)
            band = abs(prev_smoothed - smoothed)
        # Threshold to binary
        threshold = max(band.mean() * 0.5, 1e-8)
        binary_band = (band > threshold).astype(np.int8)
        scales = push!(, binary_band[:, 0] if squeeze else binary_band)
    return scales
end

end # module WaveletsAccel
