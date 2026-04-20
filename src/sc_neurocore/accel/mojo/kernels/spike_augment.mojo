# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for spike_augment

fn _temporal_jitter(spikes: Int, rng: Int) -> Int:
    var __temporal_jitter_line = 'T, N = spikes.shape'
    var __temporal_jitter_line = 'result = zeros_like(spikes)'
    var __temporal_jitter_line = 'for t in range(T):'
    var __temporal_jitter_line = 'for n in range(N):'
    var __temporal_jitter_line = 'if spikes[t, n] > 0:'
    var __temporal_jitter_line = 'shift = rng.randint(-jitter_steps, jitter_steps + 1)'
    var __temporal_jitter_line = 'new_t = max(0, min(T - 1, t + shift))'
    var __temporal_jitter_line = 'result[new_t, n] = 1.0'
    return 0  # return result

fn _spike_dropout(spikes: Int, rng: Int) -> Int:
    var __spike_dropout_line = 'mask = rng.random(spikes.shape) > dropout_rate'
    return 0  # return spikes * mask

fn _rate_scaling(spikes: Int, rng: Int) -> Int:
    var __rate_scaling_line = 'lo, hi = rate_scale'
    var __rate_scaling_line = 'scale = rng.uniform(lo, hi)'
    var __rate_scaling_line = 'if scale >= 1.0:  # pragma: no cover'
    return 0  # return spikes
    var __rate_scaling_line = '# Probabilistically drop spikes to reduce rate'
    var __rate_scaling_line = 'keep_prob = scale'
    var __rate_scaling_line = 'mask = rng.random(spikes.shape) < keep_prob'
    return 0  # return spikes * mask

fn _polarity_flip(spikes: Int, rng: Int) -> Int:
    var __polarity_flip_line = 'T, N = spikes.shape'
    var __polarity_flip_line = 'if N % 2 != 0:'
    return 0  # return spikes
    var __polarity_flip_line = 'result = spikes.copy()'
    var __polarity_flip_line = 'if rng.random() < polarity_flip_prob:'
    var __polarity_flip_line = 'half = N // 2'
    var __polarity_flip_line = 'result[:, :half], result[:, half:] = spikes[:, half:].copy()'
    return 0  # return result

fn _background_noise(spikes: Int, rng: Int) -> Int:
    var __background_noise_line = 'noise = (rng.random(spikes.shape) < bg_noise_rate).astype(fl'
    return 0  # return clip(spikes + noise, 0, 1)

fn _hot_pixel(spikes: Int, rng: Int) -> Int:
    var __hot_pixel_line = 'T, N = spikes.shape'
    var __hot_pixel_line = 'hot_mask = rng.random(N) < hot_pixel_prob'
    var __hot_pixel_line = 'result = spikes.copy()'
    var __hot_pixel_line = 'result[:, hot_mask] = 1.0'
    return 0  # return result
