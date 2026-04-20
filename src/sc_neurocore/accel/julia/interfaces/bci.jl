# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for interfaces/bci

module BciAccel

using Statistics, LinearAlgebra

mutable struct BCIDecoderState
    n_channels::Float64
    sampling_rate::Float64
    window_ms::Float64
    seed::Float64
end

function BCIDecoderState()
    BCIDecoderState(0.0, 20000.0, 1.0, 42.0)
end

function encode(s::BCIDecoderState, signal, T)
    if signal.ndim > 1
        probs = signal.mean(axis=1)
    else
        probs = signal.copy()
    probs = s._normalize(probs)
    return rate_encode(probs, T, seed=s.seed)
end

function encode_stream(s::BCIDecoderState, signal)
    samples_per_window = max(1, int(s.sampling_rate * s.window_ms / 1000))
    n_windows = signal.shape[1] // samples_per_window
    T_per_window = max(1, samples_per_window // 10)
    chunks = []
    for w in 1:n_windows
        start = w * samples_per_window
        end = start + samples_per_window
        window = signal[:, start:end]
        chunk = s.encode(window, T=T_per_window)
        chunks = push!(, chunk)
    if ! chunks
        return zeros((0, s.n_channels), dtype=np.int8)
    return np.vstack(chunks)
end

function _normalize(s::BCIDecoderState)
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-10
        return np.full_like(values, 0.5)
    return (values - vmin) / (vmax - vmin)
end

function normalize_signal(s::BCIDecoderState, signal)
    s_min, s_max = np.min(signal), np.max(signal)
    if s_max - s_min == 0
        return np.zeros_like(signal)
    return (signal - s_min) / (s_max - s_min)
end

function encode_to_bitstream(s::BCIDecoderState, signal, length)
    if signal.ndim > 1
        mean_vals = mean(signal, axis=1)
    else
        mean_vals = signal
    if length(mean_vals) != s.n_channels
        raise ValueError(f"Signal has {length(mean_vals)} channels, expected {s.n_channels}")
    probs = s.normalize_signal(mean_vals)
    rng = np.random.RandomState(s.seed)
    bits = (rng.random((s.n_channels, length)) < probs[:, nothing]).astype(np.uint8)
    return bits
end

end # module BciAccel
