# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for surrogates

fn surrogate_isi_shuffle(binary_train: Int, seed: Int) -> Int:
    var _surrogate_isi_shuffle_line = 'binary_train: ndarray[Any, Any], seed: int = 0'
    var _surrogate_isi_shuffle_line = ') -> ndarray[Any, Any]:'
    var _surrogate_isi_shuffle_line = 'intervals = diff(where(binary_train > 0)[0])'
    var _surrogate_isi_shuffle_line = 'if intervals.size < 2:'
    return 0  # return binary_train.copy()
    var _surrogate_isi_shuffle_line = 'rng = random.default_rng(seed)'
    var _surrogate_isi_shuffle_line = 'rng.shuffle(intervals)'
    var _surrogate_isi_shuffle_line = 'out = zeros_like(binary_train)'
    var _surrogate_isi_shuffle_line = 'idx = where(binary_train > 0)[0][0]'
    var _surrogate_isi_shuffle_line = 'out[idx] = 1'
    var _surrogate_isi_shuffle_line = 'for gap in intervals:'
    var _surrogate_isi_shuffle_line = 'idx += gap'
    var _surrogate_isi_shuffle_line = 'if idx < out.size:'
    var _surrogate_isi_shuffle_line = 'out[idx] = 1'
    return 0  # return out

fn surrogate_dither(binary_train: Int, dither_ms: Int, dt: Int, seed: Int) -> Int:
    var _surrogate_dither_line = 'binary_train: ndarray[Any, Any], dither_ms: float = 5.0, dt:'
    var _surrogate_dither_line = ') -> ndarray[Any, Any]:'
    var _surrogate_dither_line = 'rng = random.default_rng(seed)'
    var _surrogate_dither_line = 'dither_steps = int(dither_ms / (dt * 1000))'
    var _surrogate_dither_line = 'times = where(binary_train > 0)[0]'
    var _surrogate_dither_line = 'out = zeros_like(binary_train)'
    var _surrogate_dither_line = 'for t in times:'
    var _surrogate_dither_line = 'jittered = t + rng.integers(-dither_steps, dither_steps + 1)'
    var _surrogate_dither_line = 'jittered = clip(jittered, 0, out.size - 1)'
    var _surrogate_dither_line = 'out[jittered] = 1'
    return 0  # return out

fn surrogate_trial_shuffle(trains: Int, seed: Int) -> Int:
    var _surrogate_trial_shuffle_line = 'trains: list[ndarray[Any, Any]], seed: int = 0'
    var _surrogate_trial_shuffle_line = ') -> list[ndarray[Any, Any]]:'
    var _surrogate_trial_shuffle_line = 'rng = random.default_rng(seed)'
    var _surrogate_trial_shuffle_line = 'idx = rng.permutation(len(trains))'
    return 0  # return [trains[i] for i in idx]

fn homogeneous_poisson(rate_hz: Int, duration_s: Int, dt: Int, seed: Int) -> Int:
    var _homogeneous_poisson_line = 'rate_hz: float, duration_s: float, dt: float = 0.001, seed: '
    var _homogeneous_poisson_line = ') -> ndarray[Any, Any]:'
    var _homogeneous_poisson_line = 'rng = random.default_rng(seed)'
    var _homogeneous_poisson_line = 'n = int(duration_s / dt)'
    return 0  # return (rng.random(n) < rate_hz * dt).astype(float

fn inhomogeneous_poisson(rate_func: Int, duration_s: Int, dt: Int, seed: Int) -> Int:
    var _inhomogeneous_poisson_line = 'rate_func: Callable[[float], float], duration_s: float, dt: '
    var _inhomogeneous_poisson_line = ') -> ndarray[Any, Any]:'
    var _inhomogeneous_poisson_line = 'rng = random.default_rng(seed)'
    var _inhomogeneous_poisson_line = 'n = int(duration_s / dt)'
    var _inhomogeneous_poisson_line = 't = arange(n) * dt'
    var _inhomogeneous_poisson_line = 'rates = array([rate_func(ti) for ti in t])'
    var _inhomogeneous_poisson_line = 'max_rate = rates.max()'
    var _inhomogeneous_poisson_line = 'if max_rate <= 0:'
    return 0  # return zeros(n)
    var _inhomogeneous_poisson_line = 'candidate = rng.random(n) < max_rate * dt'
    var _inhomogeneous_poisson_line = 'accept = rng.random(n) < rates / max(max_rate, 1e-30)'
    var _inhomogeneous_poisson_line = 'result: ndarray[Any, Any] = (candidate & accept).astype(floa'
    return 0  # return result

fn gamma_process(rate_hz: Int, shape: Int, duration_s: Int, dt: Int, seed: Int) -> Int:
    var _gamma_process_line = 'rate_hz: float, shape: float, duration_s: float, dt: float ='
    var _gamma_process_line = ') -> ndarray[Any, Any]:'
    var _gamma_process_line = 'rng = random.default_rng(seed)'
    var _gamma_process_line = 'n = int(duration_s / dt)'
    var _gamma_process_line = 'train = zeros(n)'
    var _gamma_process_line = 'if rate_hz <= 0:'
    return 0  # return train
    var _gamma_process_line = 'scale = 1.0 / (rate_hz * shape)'
    var _gamma_process_line = 't = 0.0'
    var _gamma_process_line = 'while t < duration_s:'
    var _gamma_process_line = 'interval = rng.gamma(shape, scale)'
    var _gamma_process_line = 't += interval'
    var _gamma_process_line = 'idx = int(t / dt)'
    var _gamma_process_line = 'if idx < n:'
    var _gamma_process_line = 'train[idx] = 1.0'
    return 0  # return train

fn compound_poisson_process(rate_hz: Int, burst_mean: Int, duration_s: Int, dt: Int, seed: Int) -> Int:
    var _compound_poisson_process_line = 'rate_hz: float, burst_mean: float, duration_s: float, dt: fl'
    var _compound_poisson_process_line = ') -> ndarray[Any, Any]:'
    var _compound_poisson_process_line = 'rng = random.default_rng(seed)'
    var _compound_poisson_process_line = 'n = int(duration_s / dt)'
    var _compound_poisson_process_line = 'train = zeros(n)'
    var _compound_poisson_process_line = 'events = rng.random(n) < rate_hz * dt'
    var _compound_poisson_process_line = 'event_idx = where(events)[0]'
    var _compound_poisson_process_line = 'for idx in event_idx:'
    var _compound_poisson_process_line = 'n_spikes = rng.poisson(burst_mean)'
    var _compound_poisson_process_line = 'for s in range(n_spikes):'
    var _compound_poisson_process_line = 'offset = idx + s'
    var _compound_poisson_process_line = 'if offset < n:'
    var _compound_poisson_process_line = 'train[offset] = 1.0'
    return 0  # return train

fn surrogate_joint_isi(binary_train: Int, seed: Int) -> Int:
    var _surrogate_joint_isi_line = 'times_idx = where(binary_train > 0)[0]'
    var _surrogate_joint_isi_line = 'if times_idx.size < 4:'
    return 0  # return binary_train.copy()
    var _surrogate_joint_isi_line = 'intervals = diff(times_idx)'
    var _surrogate_joint_isi_line = 'rng = random.default_rng(seed)'
    var _surrogate_joint_isi_line = 'n = intervals.size'
    var _surrogate_joint_isi_line = 'for _ in range(2 * n):'
    var _surrogate_joint_isi_line = 'i = rng.integers(0, n - 1)'
    var _surrogate_joint_isi_line = 'j = rng.integers(0, n - 1)'
    var _surrogate_joint_isi_line = 'if i != j:'
    var _surrogate_joint_isi_line = 'intervals[i], intervals[j] = intervals[j], intervals[i]'
    var _surrogate_joint_isi_line = 'out = zeros_like(binary_train)'
    var _surrogate_joint_isi_line = 'pos = times_idx[0]'
    var _surrogate_joint_isi_line = 'out[pos] = 1'
    var _surrogate_joint_isi_line = 'for gap in intervals:'
    var _surrogate_joint_isi_line = 'pos += gap'
    var _surrogate_joint_isi_line = 'if pos < out.size:'
    var _surrogate_joint_isi_line = 'out[pos] = 1'
    return 0  # return out

fn surrogate_bin_shuffling(binary_train: Int, bin_size: Int, seed: Int) -> Int:
    var _surrogate_bin_shuffling_line = 'binary_train: ndarray[Any, Any], bin_size: int = 10, seed: i'
    var _surrogate_bin_shuffling_line = ') -> ndarray[Any, Any]:'
    var _surrogate_bin_shuffling_line = 'rng = random.default_rng(seed)'
    var _surrogate_bin_shuffling_line = 'out = binary_train.copy()'
    var _surrogate_bin_shuffling_line = 'n = out.size'
    var _surrogate_bin_shuffling_line = 'for start in range(0, n, bin_size):'
    var _surrogate_bin_shuffling_line = 'end = min(start + bin_size, n)'
    var _surrogate_bin_shuffling_line = 'chunk = out[start:end].copy()'
    var _surrogate_bin_shuffling_line = 'rng.shuffle(chunk)'
    var _surrogate_bin_shuffling_line = 'out[start:end] = chunk'
    return 0  # return out

fn surrogate_spike_train_shifting(binary_train: Int, max_shift: Int, seed: Int) -> Int:
    var _surrogate_spike_train_shifting_line = 'binary_train: ndarray[Any, Any], max_shift: int = 50, seed: '
    var _surrogate_spike_train_shifting_line = ') -> ndarray[Any, Any]:'
    var _surrogate_spike_train_shifting_line = 'rng = random.default_rng(seed)'
    var _surrogate_spike_train_shifting_line = 'shift = rng.integers(-max_shift, max_shift + 1)'
    return 0  # return roll(binary_train, shift)

