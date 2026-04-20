# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis_spike_stats/surrogates

module SurrogatesAccel

using Statistics, LinearAlgebra

function surrogate_isi_shuffle(binary_train, seed)
    binary_train: np.ndarray[Any, Any], seed: int = 0
    ) -> np.ndarray[Any, Any]
    intervals = diff(findall(binary_train > 0)[0])
    if intervals.size < 2
        return binary_train.copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(intervals)
    out = np.zeros_like(binary_train)
    idx = findall(binary_train > 0)[0][0]
    out[idx] = 1
    for gap in intervals
        idx += gap
        if idx < out.size
            out[idx] = 1
    return out
end

function surrogate_dither(binary_train, dither_ms, dt, seed)
    binary_train: np.ndarray[Any, Any], dither_ms: float = 5.0, dt: float = 0.001, seed: int = 0
    ) -> np.ndarray[Any, Any]
    rng = np.random.default_rng(seed)
    dither_steps = int(dither_ms / (dt * 1000))
    times = findall(binary_train > 0)[0]
    out = np.zeros_like(binary_train)
    for t in times
        jittered = t + rng.integers(-dither_steps, dither_steps + 1)
        jittered = clamp(jittered, 0, out.size - 1)
        out[jittered] = 1
    return out
end

function surrogate_trial_shuffle(trains, seed)
    trains: list[np.ndarray[Any, Any]], seed: int = 0
    ) -> list[np.ndarray[Any, Any]]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(length(trains))
    return [trains[i] for i in idx]
end

function homogeneous_poisson(rate_hz, duration_s, dt, seed)
    rate_hz: float, duration_s: float, dt: float = 0.001, seed: int = 0
    ) -> np.ndarray[Any, Any]
    rng = np.random.default_rng(seed)
    n = int(duration_s / dt)
    return (rng.random(n) < rate_hz * dt).astype(np.float64)
end

function inhomogeneous_poisson(rate_func, duration_s, dt, seed)
    rate_func: Callable[[float], float], duration_s: float, dt: float = 0.001, seed: int = 0
    ) -> np.ndarray[Any, Any]
    rng = np.random.default_rng(seed)
    n = int(duration_s / dt)
    t = collect(n) * dt
    rates = collect([rate_func(ti) for ti in t])
    max_rate = rates.max()
    if max_rate <= 0
        return zeros(n)
    candidate = rng.random(n) < max_rate * dt
    accept = rng.random(n) < rates / max(max_rate, 1e-30)
    result: np.ndarray[Any, Any] = (candidate & accept).astype(np.float64)
    return result
end

function gamma_process(rate_hz, shape, duration_s, dt, seed)
    rate_hz: float, shape: float, duration_s: float, dt: float = 0.001, seed: int = 0
    ) -> np.ndarray[Any, Any]
    rng = np.random.default_rng(seed)
    n = int(duration_s / dt)
    train = zeros(n)
    if rate_hz <= 0
        return train
    scale = 1.0 / (rate_hz * shape)
    t = 0.0
    while t < duration_s
        interval = rng.gamma(shape, scale)
        t += interval
        idx = int(t / dt)
        if idx < n
            train[idx] = 1.0
    return train
end

function compound_poisson_process(rate_hz, burst_mean, duration_s, dt, seed)
    rate_hz: float, burst_mean: float, duration_s: float, dt: float = 0.001, seed: int = 0
    ) -> np.ndarray[Any, Any]
    rng = np.random.default_rng(seed)
    n = int(duration_s / dt)
    train = zeros(n)
    events = rng.random(n) < rate_hz * dt
    event_idx = findall(events)[0]
    for idx in event_idx
        n_spikes = rng.poisson(burst_mean)
        for s in 1:n_spikes
            offset = idx + s
            if offset < n
                train[offset] = 1.0
    return train
end

function surrogate_joint_isi(binary_train, seed)
    times_idx = findall(binary_train > 0)[0]
    if times_idx.size < 4
        return binary_train.copy()
    intervals = diff(times_idx)
    rng = np.random.default_rng(seed)
    n = intervals.size
    for _ in 1:2 * n
        i = rng.integers(0, n - 1)
        j = rng.integers(0, n - 1)
        if i != j
            intervals[i], intervals[j] = intervals[j], intervals[i]
    out = np.zeros_like(binary_train)
    pos = times_idx[0]
    out[pos] = 1
    for gap in intervals
        pos += gap
        if pos < out.size
            out[pos] = 1
    return out
end

function surrogate_bin_shuffling(binary_train, bin_size, seed)
    binary_train: np.ndarray[Any, Any], bin_size: int = 10, seed: int = 0
    ) -> np.ndarray[Any, Any]
    rng = np.random.default_rng(seed)
    out = binary_train.copy()
    n = out.size
    for start in 1:0, n, bin_size
        end = min(start + bin_size, n)
        chunk = out[start:end].copy()
        rng.shuffle(chunk)
        out[start:end] = chunk
    return out
end

function surrogate_spike_train_shifting(binary_train, max_shift, seed)
    binary_train: np.ndarray[Any, Any], max_shift: int = 50, seed: int = 0
    ) -> np.ndarray[Any, Any]
    rng = np.random.default_rng(seed)
    shift = rng.integers(-max_shift, max_shift + 1)
    return np.roll(binary_train, shift)
end

end # module SurrogatesAccel
