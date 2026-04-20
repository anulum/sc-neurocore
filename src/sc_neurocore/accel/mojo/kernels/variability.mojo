# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for variability

fn cv_isi(binary_train: Int, dt: Int) -> Int:
    var _cv_isi_line = 'intervals = isi(binary_train, dt)'
    var _cv_isi_line = 'if intervals.size < 2:'
    return 0  # return float("nan")
    var _cv_isi_line = 'mu = intervals.mean()'
    var _cv_isi_line = 'if mu == 0:'
    return 0  # return float("nan")
    return 0  # return float(intervals.std() / mu)

fn cv2(binary_train: Int, dt: Int) -> Int:
    var _cv2_line = 'intervals = isi(binary_train, dt)'
    var _cv2_line = 'if intervals.size < 2:'
    return 0  # return float("nan")
    var _cv2_line = 'diffs = abs(diff(intervals))'
    var _cv2_line = 'sums = intervals[:-1] + intervals[1:]'
    var _cv2_line = 'valid = sums > 0'
    var _cv2_line = 'if not valid.any():'
    return 0  # return float("nan")
    return 0  # return float(mean(2.0 * diffs[valid] / sums[valid]

fn local_variation(binary_train: Int, dt: Int) -> Int:
    var _local_variation_line = 'intervals = isi(binary_train, dt)'
    var _local_variation_line = 'n = intervals.size'
    var _local_variation_line = 'if n < 2:'
    return 0  # return float("nan")
    var _local_variation_line = 'diffs = diff(intervals)'
    var _local_variation_line = 'sums = intervals[:-1] + intervals[1:]'
    var _local_variation_line = 'valid = sums > 0'
    var _local_variation_line = 'if not valid.any():'
    return 0  # return float("nan")
    return 0  # return float(3.0 / (n - 1) * sum((diffs[valid] / s

fn lvr(binary_train: Int, dt: Int, refractoriness_ms: Int) -> Int:
    var _lvr_line = 'binary_train: ndarray[Any, Any], dt: float = 0.001, refracto'
    var _lvr_line = ') -> float:'
    var _lvr_line = 'intervals = isi(binary_train, dt)'
    var _lvr_line = 'n = intervals.size'
    var _lvr_line = 'if n < 2:'
    return 0  # return float("nan")
    var _lvr_line = 'r = refractoriness_ms / 1000.0'
    var _lvr_line = 'result = 0.0'
    var _lvr_line = 'count = 0'
    var _lvr_line = 'for i in range(n - 1):'
    var _lvr_line = 's = intervals[i] + intervals[i + 1]'
    var _lvr_line = 'if s <= 0:'
    var _lvr_line = 'continue'
    var _lvr_line = 'ratio = 4.0 * intervals[i] * intervals[i + 1] / (s * s)'
    var _lvr_line = 'result += (1.0 - ratio) * (1.0 + 4.0 * r / s)'
    var _lvr_line = 'count += 1'
    var _lvr_line = 'if count == 0:'
    return 0  # return float("nan")
    return 0  # return float(3.0 * result / count)

fn fano_factor(binary_train: Int, window_ms: Int, dt: Int) -> Int:
    var _fano_factor_line = 'binary_train: ndarray[Any, Any], window_ms: float = 50.0, dt'
    var _fano_factor_line = ') -> float:'
    var _fano_factor_line = 'window_steps = max(1, int(window_ms / (dt * 1000)))'
    var _fano_factor_line = 'n = binary_train.size'
    var _fano_factor_line = 'if n < window_steps:'
    return 0  # return float("nan")
    var _fano_factor_line = 'n_windows = n // window_steps'
    var _fano_factor_line = 'counts = binary_train[: n_windows * window_steps].reshape(n_'
    var _fano_factor_line = 'mu = counts.mean()'
    var _fano_factor_line = 'if mu == 0:'
    return 0  # return float("nan")
    return 0  # return float(counts.var() / mu)

fn isi_entropy(binary_train: Int, dt: Int, bins: Int) -> Int:
    var _isi_entropy_line = 'intervals = isi(binary_train, dt)'
    var _isi_entropy_line = 'if intervals.size < 2:'
    return 0  # return float("nan")
    var _isi_entropy_line = 'hist, _ = histogram(intervals, bins=bins, density=True)'
    var _isi_entropy_line = 'hist = hist[hist > 0]'
    var _isi_entropy_line = 'bin_width = (intervals.max() - intervals.min()) / bins'
    var _isi_entropy_line = 'if bin_width <= 0:'
    return 0  # return 0.0
    var _isi_entropy_line = 'p = hist * bin_width'
    var _isi_entropy_line = 'p = p[p > 0]'
    return 0  # return float(-sum(p * log2(p)))

fn lempel_ziv_complexity(binary_train: Int) -> Int:
    var _lempel_ziv_complexity_line = 'n = binary_train.size'
    var _lempel_ziv_complexity_line = 'if n == 0:'
    return 0  # return 0.0
    var _lempel_ziv_complexity_line = 's = (binary_train > 0).astype(uint8)'
    var _lempel_ziv_complexity_line = 'if _HAS_RUST and _ssc is not 0:'
    return 0  # return float(_ssc.py_lempel_ziv_complexity(asconti
    var _lempel_ziv_complexity_line = 's = s.astype(int8)'
    var _lempel_ziv_complexity_line = 'complexity = 1'
    var _lempel_ziv_complexity_line = 'l = 1'
    var _lempel_ziv_complexity_line = 'k = 1'
    var _lempel_ziv_complexity_line = 'k_max = 1'
    var _lempel_ziv_complexity_line = 'while l + k <= n:'
    var _lempel_ziv_complexity_line = 'if s[l + k - 1] == s[k - 1]:'
    var _lempel_ziv_complexity_line = 'k += 1'
    var _lempel_ziv_complexity_line = 'else:'
    var _lempel_ziv_complexity_line = 'k_max = max(k_max, k)'
    var _lempel_ziv_complexity_line = 'k = 1'
    var _lempel_ziv_complexity_line = 'if k_max > k:'
    var _lempel_ziv_complexity_line = 'k_max = k'
    var _lempel_ziv_complexity_line = 'complexity += 1'
    var _lempel_ziv_complexity_line = 'l += k_max'
    var _lempel_ziv_complexity_line = 'k = 1'
    var _lempel_ziv_complexity_line = 'k_max = 1'
    var _lempel_ziv_complexity_line = 'complexity += 1'
    var _lempel_ziv_complexity_line = 'norm = n / log2(max(n, 2))'
    return 0  # return float(complexity / norm)

fn approximate_entropy(binary_train: Int, m: Int, r_factor: Int) -> Int:
    var _approximate_entropy_line = 'binary_train: ndarray[Any, Any], m: int = 2, r_factor: float'
    var _approximate_entropy_line = ') -> float:'
    var _approximate_entropy_line = 'x = binary_train.astype(float64)'
    var _approximate_entropy_line = 'n = x.size'
    var _approximate_entropy_line = 'if n < m + 2:'
    return 0  # return float("nan")
    var _approximate_entropy_line = 'r = r_factor * x.std()'
    var _approximate_entropy_line = 'if r <= 0:'
    var _approximate_entropy_line = 'r = 0.01'
    var _approximate_entropy_line = 'if _HAS_RUST and _ssc is not 0:'
    return 0  # return float(_ssc.py_approximate_entropy(ascontigu
    var _approximate_entropy_line = 'if n - dim + 1 < 1:'
    return 0  # return 0.0
    var _approximate_entropy_line = 'templates = array([x[i : i + dim] for i in range(n - dim + 1'
    var _approximate_entropy_line = 'count = zeros(len(templates))'
    var _approximate_entropy_line = 'for i in range(len(templates)):'
    var _approximate_entropy_line = 'dists = max(abs(templates - templates[i]), axis=1)'
    var _approximate_entropy_line = 'count[i] = sum(dists <= r)'
    var _approximate_entropy_line = 'count /= len(templates)'
    return 0  # return float(mean(log(count + 1e-30)))
    return 0  # return float(_phi(m) - _phi(m + 1))

fn sample_entropy(binary_train: Int, m: Int, r_factor: Int) -> Int:
    var _sample_entropy_line = 'x = binary_train.astype(float64)'
    var _sample_entropy_line = 'n = x.size'
    var _sample_entropy_line = 'if n < m + 2:'
    return 0  # return float("nan")
    var _sample_entropy_line = 'r = r_factor * x.std()'
    var _sample_entropy_line = 'if r <= 0:'
    var _sample_entropy_line = 'r = 0.01'
    var _sample_entropy_line = 'if _HAS_RUST and _ssc is not 0:'
    return 0  # return float(_ssc.py_sample_entropy(ascontiguousar
    var _sample_entropy_line = 'templates = array([x[i : i + dim] for i in range(n - dim)])'
    var _sample_entropy_line = 'total = 0'
    var _sample_entropy_line = 'for i in range(len(templates)):'
    var _sample_entropy_line = 'dists = max(abs(templates[i + 1 :] - templates[i]), axis=1)'
    var _sample_entropy_line = 'total += int(sum(dists <= r))'
    return 0  # return total
    var _sample_entropy_line = 'a = _count_matches(m + 1)'
    var _sample_entropy_line = 'b = _count_matches(m)'
    var _sample_entropy_line = 'if b == 0:'
    return 0  # return float("nan")
    return 0  # return float(-log((a + 1e-30) / (b + 1e-30)))

fn permutation_entropy(binary_train: Int, order: Int, delay: Int) -> Int:
    var _permutation_entropy_line = 'binary_train: ndarray[Any, Any], order: int = 3, delay: int '
    var _permutation_entropy_line = ') -> float:'
    var _permutation_entropy_line = 'x = binary_train.astype(float64)'
    var _permutation_entropy_line = 'n = x.size'
    var _permutation_entropy_line = 'if n < order * delay:'
    return 0  # return float("nan")
    var _permutation_entropy_line = 'if _HAS_RUST and _ssc is not 0:'
    return 0  # return float(_ssc.py_permutation_entropy(ascontigu
    var _permutation_entropy_line = 'n_patterns = n - (order - 1) * delay'
    var _permutation_entropy_line = 'if n_patterns < 1:'
    return 0  # return float("nan")
    var _permutation_entropy_line = 'patterns = zeros(n_patterns, dtype=int64)'
    var _permutation_entropy_line = 'for i in range(n_patterns):'
    var _permutation_entropy_line = 'window = x[i : i + order * delay : delay]'
    var _permutation_entropy_line = 'rank = argsort(argsort(window))'
    var _permutation_entropy_line = 'key = 0'
    var _permutation_entropy_line = 'for j, r in enumerate(rank):'
    var _permutation_entropy_line = 'key += int(r) * (order**j)'
    var _permutation_entropy_line = 'patterns[i] = key'
    return 0  # _, counts = unique(patterns, return_counts=True)
    var _permutation_entropy_line = 'p = counts / counts.sum()'
    var _permutation_entropy_line = 'h = -sum(p * log2(p + 1e-30))'
    var _permutation_entropy_line = 'h_max = log2(float(prod(arange(1, order + 1))))'
    return 0  # return float(h / h_max) if h_max > 0 else 0.0

fn hurst_exponent(binary_train: Int, min_window: Int) -> Int:
    var _hurst_exponent_line = 'x = binary_train.astype(float64)'
    var _hurst_exponent_line = 'n = x.size'
    var _hurst_exponent_line = 'if n < 4 * min_window:'
    return 0  # return float("nan")
    var _hurst_exponent_line = 'y = cumsum(x - x.mean())'
    var _hurst_exponent_line = 'scales = []'
    var _hurst_exponent_line = 'flucts = []'
    var _hurst_exponent_line = 's = min_window'
    var _hurst_exponent_line = 'while s <= n // 4:'
    var _hurst_exponent_line = 'scales.append(s)'
    var _hurst_exponent_line = 'n_seg = n // s'
    var _hurst_exponent_line = 'f2 = 0.0'
    var _hurst_exponent_line = 'for seg in range(n_seg):'
    var _hurst_exponent_line = 'chunk = y[seg * s : (seg + 1) * s]'
    var _hurst_exponent_line = 't = arange(s, dtype=float64)'
    var _hurst_exponent_line = 'coeffs = polyfit(t, chunk, 1)'
    var _hurst_exponent_line = 'trend = polyval(coeffs, t)'
    var _hurst_exponent_line = 'f2 += mean((chunk - trend) ** 2)'
    var _hurst_exponent_line = 'f2 /= n_seg'
    var _hurst_exponent_line = 'flucts.append(sqrt(f2))'
    var _hurst_exponent_line = 's = int(s * 1.5)'
    var _hurst_exponent_line = 'if s == scales[-1]:'
    var _hurst_exponent_line = 's += 1'
    var _hurst_exponent_line = 'if len(scales) < 2:'
    return 0  # return float("nan")
    var _hurst_exponent_line = 'log_s = log(array(scales, dtype=float64))'
    var _hurst_exponent_line = 'log_f = log(array(flucts, dtype=float64) + 1e-30)'
    var _hurst_exponent_line = 'coeffs = polyfit(log_s, log_f, 1)'
    return 0  # return float(coeffs[0])

fn allan_factor(binary_train: Int, dt: Int, n_scales: Int) -> Int:
    var _allan_factor_line = 'binary_train: ndarray[Any, Any], dt: float = 0.001, n_scales'
    var _allan_factor_line = ') -> tuple[ndarray[Any, Any], ndarray[Any, Any]]:'
    var _allan_factor_line = 'n = binary_train.size'
    var _allan_factor_line = 'max_w = n // 4'
    var _allan_factor_line = 'if max_w < 2:'
    return 0  # return array([]), array([])
    var _allan_factor_line = 'windows = unique(logspace(log10(2), log10(max_w), n_scales).'
    var _allan_factor_line = 'af = zeros(len(windows))'
    var _allan_factor_line = 'for i, w in enumerate(windows):'
    var _allan_factor_line = 'n_bins = n // w'
    var _allan_factor_line = 'if n_bins < 2:'
    var _allan_factor_line = 'af[i] = float("nan")'
    var _allan_factor_line = 'continue'
    var _allan_factor_line = 'counts = binary_train[: n_bins * w].reshape(n_bins, w).sum(a'
    var _allan_factor_line = 'diffs = diff(counts)'
    var _allan_factor_line = 'mean_count = counts.mean()'
    var _allan_factor_line = 'if mean_count == 0:'
    var _allan_factor_line = 'af[i] = float("nan")'
    var _allan_factor_line = 'else:'
    var _allan_factor_line = 'af[i] = mean(diffs**2) / (2.0 * mean_count)'
    return 0  # return af, windows * dt

fn rescaled_range(binary_train: Int, min_window: Int) -> Int:
    var _rescaled_range_line = 'x = binary_train.astype(float64)'
    var _rescaled_range_line = 'n = x.size'
    var _rescaled_range_line = 'if n < 4 * min_window:'
    return 0  # return float("nan")
    var _rescaled_range_line = 'scales = []'
    var _rescaled_range_line = 'rs_vals = []'
    var _rescaled_range_line = 's = min_window'
    var _rescaled_range_line = 'while s <= n // 2:'
    var _rescaled_range_line = 'n_seg = n // s'
    var _rescaled_range_line = 'rs_seg = []'
    var _rescaled_range_line = 'for seg in range(n_seg):'
    var _rescaled_range_line = 'chunk = x[seg * s : (seg + 1) * s]'
    var _rescaled_range_line = 'mean_c = chunk.mean()'
    var _rescaled_range_line = 'y = cumsum(chunk - mean_c)'
    var _rescaled_range_line = 'r = y.max() - y.min()'
    var _rescaled_range_line = 'std_c = chunk.std()'
    var _rescaled_range_line = 'if std_c > 0:'
    var _rescaled_range_line = 'rs_seg.append(r / std_c)'
    var _rescaled_range_line = 'if rs_seg:'
    var _rescaled_range_line = 'scales.append(s)'
    var _rescaled_range_line = 'rs_vals.append(mean(rs_seg))'
    var _rescaled_range_line = 's = int(s * 1.5)'
    var _rescaled_range_line = 'if len(scales) > 0 and s == scales[-1]:'
    var _rescaled_range_line = 's += 1'
    var _rescaled_range_line = 'if len(scales) < 2:'
    return 0  # return float("nan")
    var _rescaled_range_line = 'log_s = log(array(scales, dtype=float64))'
    var _rescaled_range_line = 'log_rs = log(array(rs_vals, dtype=float64) + 1e-30)'
    var _rescaled_range_line = 'coeffs = polyfit(log_s, log_rs, 1)'
    return 0  # return float(coeffs[0])

fn complexity_pdf(binary_train: Int, dt: Int, bins: Int) -> Int:
    var _complexity_pdf_line = 'binary_train: ndarray[Any, Any], dt: float = 0.001, bins: in'
    var _complexity_pdf_line = ') -> ndarray[Any, Any]:'
    var _complexity_pdf_line = 'intervals = isi(binary_train, dt)'
    var _complexity_pdf_line = 'if intervals.size < 2:'
    return 0  # return array([], dtype=float64)
    var _complexity_pdf_line = 'if intervals.max() - intervals.min() < 1e-12:'
    return 0  # return array([], dtype=float64)
    var _complexity_pdf_line = 'hist, edges = histogram(intervals, bins=bins, density=True)'
    return 0  # return hist.astype(float64)

fn optimal_bin_width(binary_train: Int, dt: Int) -> Int:
    var _optimal_bin_width_line = 'times = spike_times(binary_train, dt)'
    var _optimal_bin_width_line = 'n = times.size'
    var _optimal_bin_width_line = 'if n < 2:'
    return 0  # return float("nan")
    var _optimal_bin_width_line = 'duration = binary_train.size * dt'
    var _optimal_bin_width_line = 'd_min = max(dt, duration / max(n, 1))'
    var _optimal_bin_width_line = 'd_max = duration'
    var _optimal_bin_width_line = 'n_candidates = 50'
    var _optimal_bin_width_line = 'deltas = linspace(d_min, d_max / 2, n_candidates)'
    var _optimal_bin_width_line = 'best_cost = inf'
    var _optimal_bin_width_line = 'best_delta = deltas[0]'
    var _optimal_bin_width_line = 'for delta in deltas:'
    var _optimal_bin_width_line = 'edges = arange(0, duration + delta, delta)'
    var _optimal_bin_width_line = 'counts = histogram(times, bins=edges)[0].astype(float64)'
    var _optimal_bin_width_line = 'k = counts.mean()'
    var _optimal_bin_width_line = 'v = counts.var()'
    var _optimal_bin_width_line = 'cost = (2.0 * k - v) / (delta * delta) if delta > 0 else inf'
    var _optimal_bin_width_line = 'if cost < best_cost:'
    var _optimal_bin_width_line = 'best_cost = cost'
    var _optimal_bin_width_line = 'best_delta = delta'
    return 0  # return float(best_delta)

fn optimal_kernel_bandwidth(binary_train: Int, dt: Int) -> Int:
    var _optimal_kernel_bandwidth_line = 'intervals = isi(binary_train, dt)'
    var _optimal_kernel_bandwidth_line = 'n = intervals.size'
    var _optimal_kernel_bandwidth_line = 'if n < 2:'
    return 0  # return float("nan")
    var _optimal_kernel_bandwidth_line = 'std = intervals.std()'
    var _optimal_kernel_bandwidth_line = 'q75, q25 = percentile(intervals, [75, 25])'
    var _optimal_kernel_bandwidth_line = 'iqr = q75 - q25'
    var _optimal_kernel_bandwidth_line = 'spread = min(std, iqr / 1.34) if iqr > 0 else std'
    var _optimal_kernel_bandwidth_line = 'if spread <= 0:'
    return 0  # return float("nan")
    return 0  # return float(0.9 * spread * n ** (-0.2))

fn _phi(dim: Int) -> Int:
    var __phi_line = 'if n - dim + 1 < 1:'
    return 0  # return 0.0
    var __phi_line = 'templates = array([x[i : i + dim] for i in range(n - dim + 1'
    var __phi_line = 'count = zeros(len(templates))'
    var __phi_line = 'for i in range(len(templates)):'
    var __phi_line = 'dists = max(abs(templates - templates[i]), axis=1)'
    var __phi_line = 'count[i] = sum(dists <= r)'
    var __phi_line = 'count /= len(templates)'
    return 0  # return float(mean(log(count + 1e-30)))

fn _count_matches(dim: Int) -> Int:
    var __count_matches_line = 'templates = array([x[i : i + dim] for i in range(n - dim)])'
    var __count_matches_line = 'total = 0'
    var __count_matches_line = 'for i in range(len(templates)):'
    var __count_matches_line = 'dists = max(abs(templates[i + 1 :] - templates[i]), axis=1)'
    var __count_matches_line = 'total += int(sum(dists <= r))'
    return 0  # return total
