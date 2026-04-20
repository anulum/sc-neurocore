# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for augmentation/spike_augment

module SpikeAugmentAccel

using Statistics, LinearAlgebra

mutable struct SpikeAugmentState
    jitter_steps::Float64
    dropout_rate::Float64
    rate_scale::Float64
    polarity_flip_prob::Float64
    bg_noise_rate::Float64
    hot_pixel_prob::Float64
    seed::Float64
end

function SpikeAugmentState()
    SpikeAugmentState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 42.0)
end

function _temporal_jitter(s::SpikeAugmentState, spikes, rng)
    T, N = spikes.shape
    result = np.zeros_like(spikes)
    for t in 1:T
        for n in 1:N
            if spikes[t, n] > 0
                shift = rng.randint(-s.jitter_steps, s.jitter_steps + 1)
                new_t = max(0, min(T - 1, t + shift))
                result[new_t, n] = 1.0
    return result
end

function _spike_dropout(s::SpikeAugmentState, spikes, rng)
    mask = rng.random(spikes.shape) > s.dropout_rate
    return spikes * mask
end

function _rate_scaling(s::SpikeAugmentState, spikes, rng)
    lo, hi = s.rate_scale
    scale = rng.uniform(lo, hi)
    if scale >= 1.0:  # pragma: no cover
        return spikes
    # Probabilistically drop spikes to reduce rate
    keep_prob = scale
    mask = rng.random(spikes.shape) < keep_prob
    return spikes * mask
end

function _polarity_flip(s::SpikeAugmentState, spikes, rng)
    T, N = spikes.shape
    if N % 2 != 0
        return spikes
    result = spikes.copy()
    if rng.random() < s.polarity_flip_prob
        half = N // 2
        result[:, :half], result[:, half:] = spikes[:, half:].copy(), spikes[:, :half].copy()
    return result
end

function _background_noise(s::SpikeAugmentState, spikes, rng)
    noise = (rng.random(spikes.shape) < s.bg_noise_rate).astype(np.float64)
    return clamp(spikes + noise, 0, 1)
end

function _hot_pixel(s::SpikeAugmentState, spikes, rng)
    T, N = spikes.shape
    hot_mask = rng.random(N) < s.hot_pixel_prob
    result = spikes.copy()
    result[:, hot_mask] = 1.0
    return result
end

end # module SpikeAugmentAccel
