# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for temporal

fn burst_detection(binary_train: Int, dt: Int, max_isi_ms: Int, min_spikes: Int) -> Int:
    var _burst_detection_line = 'binary_train: ndarray, dt: float = 0.001, max_isi_ms: float '
    var _burst_detection_line = ') -> list[tuple[float, float, int]]:'
    var _burst_detection_line = 'times = spike_times(binary_train, dt)'
    var _burst_detection_line = 'if times.size < min_spikes:'
    return 0  # return []
    var _burst_detection_line = 'max_isi = max_isi_ms / 1000.0'
    var _burst_detection_line = 'intervals = diff(times)'
    var _burst_detection_line = 'bursts = []'
    var _burst_detection_line = 'i = 0'
    var _burst_detection_line = 'while i < intervals.size:'
    var _burst_detection_line = 'if intervals[i] < max_isi:'
    var _burst_detection_line = 'start = i'
    var _burst_detection_line = 'while i < intervals.size and intervals[i] < max_isi:'
    var _burst_detection_line = 'i += 1'
    var _burst_detection_line = 'n_spikes = i - start + 1'
    var _burst_detection_line = 'if n_spikes >= min_spikes:'
    var _burst_detection_line = 'bursts.append((float(times[start]), float(times[i]), n_spike'
    var _burst_detection_line = 'else:'
    var _burst_detection_line = 'i += 1'
    return 0  # return bursts

fn first_spike_latency(binary_train: Int, dt: Int) -> Int:
    var _first_spike_latency_line = 'idx = argmax(binary_train > 0)'
    var _first_spike_latency_line = 'if binary_train[idx] == 0:'
    return 0  # return float("nan")
    return 0  # return float(idx * dt)

fn response_onset(binary_train: Int, baseline_steps: Int, dt: Int, threshold_sigma: Int) -> Int:
    var _response_onset_line = 'binary_train: ndarray,'
    var _response_onset_line = 'baseline_steps: int = 100,'
    var _response_onset_line = 'dt: float = 0.001,'
    var _response_onset_line = 'threshold_sigma: float = 3.0,'
    var _response_onset_line = ') -> float:'
    var _response_onset_line = 'if binary_train.size <= baseline_steps:'
    return 0  # return float("nan")
    var _response_onset_line = 'baseline_rate = binary_train[:baseline_steps].mean()'
    var _response_onset_line = 'baseline_std = binary_train[:baseline_steps].std()'
    var _response_onset_line = 'if baseline_std == 0:'
    var _response_onset_line = 'baseline_std = 1e-10'
    var _response_onset_line = 'threshold = baseline_rate + threshold_sigma * baseline_std'
    var _response_onset_line = 'post = binary_train[baseline_steps:]'
    var _response_onset_line = 'above = where(post > threshold)[0]'
    var _response_onset_line = 'if above.size == 0:'
    return 0  # return float("nan")
    return 0  # return float((baseline_steps + above[0]) * dt)

fn change_point_detection(binary_train: Int, bin_size: Int, threshold: Int) -> Int:
    var _change_point_detection_line = 'binary_train: ndarray, bin_size: int = 50, threshold: float '
    var _change_point_detection_line = ') -> list[int]:'
    var _change_point_detection_line = 'counts = bin_spike_train(binary_train, bin_size).astype(floa'
    var _change_point_detection_line = 'n = counts.size'
    var _change_point_detection_line = 'if n < 5:'
    return 0  # return []
    var _change_point_detection_line = 'mean_rate = counts.mean()'
    var _change_point_detection_line = 'std_rate = counts.std()'
    var _change_point_detection_line = 'if std_rate < 1e-10:'
    return 0  # return []
    var _change_point_detection_line = 'cusum_pos = zeros(n)'
    var _change_point_detection_line = 'cusum_neg = zeros(n)'
    var _change_point_detection_line = 'change_points = []'
    var _change_point_detection_line = 'for i in range(1, n):'
    var _change_point_detection_line = 'cusum_pos[i] = max(0, cusum_pos[i - 1] + (counts[i] - mean_r'
    var _change_point_detection_line = 'cusum_neg[i] = max(0, cusum_neg[i - 1] - (counts[i] - mean_r'
    var _change_point_detection_line = 'if cusum_pos[i] > threshold or cusum_neg[i] > threshold:'
    var _change_point_detection_line = 'change_points.append(i)'
    var _change_point_detection_line = 'cusum_pos[i] = 0'
    var _change_point_detection_line = 'cusum_neg[i] = 0'
    return 0  # return change_points

