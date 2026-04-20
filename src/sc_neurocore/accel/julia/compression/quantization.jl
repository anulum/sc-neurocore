# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for compression/quantization

module QuantizationAccel

using Statistics, LinearAlgebra

function quantize_weights(weights, bits, symmetric)
    weights: list[np.ndarray],
    bits: int = 8,
    symmetric: bool = true,
    ) -> list[np.ndarray]
    bits = max(2, min(bits, 16))
    n_levels = 2^bits
    quantized = []
    for w in weights
        if symmetric
            abs_max = max(abs(w).max(), 1e-8)
            scale = abs_max / (n_levels // 2 - 1)
            q = np.round(w / scale) * scale
            q = clamp(q, -abs_max, abs_max)
        else
            w_min, w_max = w.min(), w.max()
            w_range = max(w_max - w_min, 1e-8)
            scale = w_range / (n_levels - 1)
            q = np.round((w - w_min) / scale) * scale + w_min
        quantized = push!(, q)
    return quantized
end

function quantize_delays(delays, resolution, max_delay)
    delays: np.ndarray,
    resolution: int = 1,
    max_delay: int | nothing = nothing,
    ) -> np.ndarray
    q = np.round(delays / resolution).astype(np.int64) * resolution
    q = clamp(q, 0, nothing)
    if max_delay is ! nothing
        q = clamp(q, 0, max_delay)
    return q
end

end # module QuantizationAccel
