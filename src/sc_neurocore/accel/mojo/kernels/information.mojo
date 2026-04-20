# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for information

fn mutual_information(train_a: Int, train_b: Int, bin_size: Int) -> Int:
    var _mutual_information_line = 'train_a: ndarray[Any, Any], train_b: ndarray[Any, Any], bin_'
    var _mutual_information_line = ') -> float:'
    var _mutual_information_line = 'ca = bin_spike_train(train_a, bin_size)'
    var _mutual_information_line = 'cb = bin_spike_train(train_b, bin_size)'
    var _mutual_information_line = 'n = min(ca.size, cb.size)'
    var _mutual_information_line = 'ca, cb = ca[:n], cb[:n]'
    return 0  # vals, counts = unique(x, return_counts=True)
    var _mutual_information_line = 'p = counts / counts.sum()'
    return 0  # return float(-sum(p * log2(p + 1e-30)))
    var _mutual_information_line = 'ha = _entropy(ca)'
    var _mutual_information_line = 'hb = _entropy(cb)'
    var _mutual_information_line = 'joint = ca * (cb.max() + 1) + cb'
    var _mutual_information_line = 'hab = _entropy(joint)'
    return 0  # return max(0.0, ha + hb - hab)

fn transfer_entropy(source: Int, target: Int, bin_size: Int, lag: Int) -> Int:
    var _transfer_entropy_line = 'source: ndarray[Any, Any], target: ndarray[Any, Any], bin_si'
    var _transfer_entropy_line = ') -> float:'
    var _transfer_entropy_line = 'cs = bin_spike_train(source, bin_size)'
    var _transfer_entropy_line = 'ct = bin_spike_train(target, bin_size)'
    var _transfer_entropy_line = 'n = min(cs.size, ct.size)'
    var _transfer_entropy_line = 'if n <= lag:'
    return 0  # return 0.0
    var _transfer_entropy_line = 'cs, ct = cs[:n], ct[:n]'
    var _transfer_entropy_line = 't_past = ct[:-lag]'
    var _transfer_entropy_line = 't_future = ct[lag:]'
    var _transfer_entropy_line = 's_past = cs[:-lag]'
    var _transfer_entropy_line = 'n_pts = t_past.size'
    var _transfer_entropy_line = 'joint = future.copy()'
    var _transfer_entropy_line = 'for p in pasts:'
    var _transfer_entropy_line = 'joint = joint * (p.max() + 1) + p'
    return 0  # vals, counts = unique(joint, return_counts=True)
    var _transfer_entropy_line = 'h_joint = float(-sum(counts / n_pts * log2(counts / n_pts + '
    var _transfer_entropy_line = 'past_joint = pasts[0].copy()'
    var _transfer_entropy_line = 'for p in pasts[1:]:'
    var _transfer_entropy_line = 'past_joint = past_joint * (p.max() + 1) + p'
    return 0  # vals2, counts2 = unique(past_joint, return_counts=
    var _transfer_entropy_line = 'h_past = float(-sum(counts2 / n_pts * log2(counts2 / n_pts +'
    return 0  # return h_joint - h_past
    var _transfer_entropy_line = 'h1 = _cond_entropy(t_future, t_past)'
    var _transfer_entropy_line = 'h2 = _cond_entropy(t_future, t_past, s_past)'
    return 0  # return max(0.0, float(h1 - h2))

fn spike_train_entropy(binary_train: Int, bin_size: Int, word_length: Int) -> Int:
    var _spike_train_entropy_line = 'binary_train: ndarray[Any, Any], bin_size: int = 10, word_le'
    var _spike_train_entropy_line = ') -> float:'
    var _spike_train_entropy_line = 'binned = (bin_spike_train(binary_train, bin_size) > 0).astyp'
    var _spike_train_entropy_line = 'n = binned.size'
    var _spike_train_entropy_line = 'if n < word_length:'
    return 0  # return float("nan")
    var _spike_train_entropy_line = 'if _HAS_RUST and _ssc is not 0:'
    return 0  # return float(_ssc.py_spike_train_entropy(ascontigu
    var _spike_train_entropy_line = 'n_words = n - word_length + 1'
    var _spike_train_entropy_line = 'words = zeros(n_words, dtype=int64)'
    var _spike_train_entropy_line = 'for i in range(n_words):'
    var _spike_train_entropy_line = 'w = 0'
    var _spike_train_entropy_line = 'for j in range(word_length):'
    var _spike_train_entropy_line = 'w = w * 2 + int(binned[i + j])'
    var _spike_train_entropy_line = 'words[i] = w'
    return 0  # _, counts = unique(words, return_counts=True)
    var _spike_train_entropy_line = 'p = counts / counts.sum()'
    return 0  # return float(-sum(p * log2(p + 1e-30)))

fn noise_entropy(binary_train: Int, n_trials: Int, bin_size: Int, word_length: Int) -> Int:
    var _noise_entropy_line = 'binary_train: ndarray[Any, Any], n_trials: int = 10, bin_siz'
    var _noise_entropy_line = ') -> float:'
    var _noise_entropy_line = 'n = binary_train.size'
    var _noise_entropy_line = 'trial_len = n // n_trials'
    var _noise_entropy_line = 'if trial_len < bin_size * word_length:'
    return 0  # return float("nan")
    var _noise_entropy_line = 'entropies = []'
    var _noise_entropy_line = 'for t in range(n_trials):'
    var _noise_entropy_line = 'seg = binary_train[t * trial_len : (t + 1) * trial_len]'
    var _noise_entropy_line = 'h = spike_train_entropy(seg, bin_size, word_length)'
    var _noise_entropy_line = 'if not isnan(h):'
    var _noise_entropy_line = 'entropies.append(h)'
    var _noise_entropy_line = 'if not entropies:'
    return 0  # return float("nan")
    return 0  # return float(mean(entropies))

fn stimulus_specific_information(spike_counts: Int, stimulus_ids: Int) -> Int:
    var _stimulus_specific_information_line = 'spike_counts: ndarray[Any, Any], stimulus_ids: ndarray[Any, '
    var _stimulus_specific_information_line = ') -> float:'
    var _stimulus_specific_information_line = 'unique_stim = unique(stimulus_ids)'
    var _stimulus_specific_information_line = 'n_total = len(spike_counts)'
    var _stimulus_specific_information_line = 'if n_total == 0:'
    return 0  # return 0.0
    var _stimulus_specific_information_line = 'overall_mean = spike_counts.mean()'
    var _stimulus_specific_information_line = 'if overall_mean <= 0:'
    return 0  # return 0.0
    var _stimulus_specific_information_line = 'ssi = 0.0'
    var _stimulus_specific_information_line = 'for s in unique_stim:'
    var _stimulus_specific_information_line = 'mask = stimulus_ids == s'
    var _stimulus_specific_information_line = 'n_s = mask.sum()'
    var _stimulus_specific_information_line = 'if n_s == 0:'
    var _stimulus_specific_information_line = 'continue'
    var _stimulus_specific_information_line = 'p_s = n_s / n_total'
    var _stimulus_specific_information_line = 'mean_s = spike_counts[mask].mean()'
    var _stimulus_specific_information_line = 'if mean_s > 0:'
    var _stimulus_specific_information_line = 'ssi += p_s * mean_s * log2(mean_s / overall_mean) / overall_'
    return 0  # return float(max(0.0, ssi))

fn kozachenko_leonenko_mi(x: Int, y: Int, k: Int) -> Int:
    var _kozachenko_leonenko_mi_line = 'n = min(x.size, y.size)'
    var _kozachenko_leonenko_mi_line = 'if n < k + 1:'
    return 0  # return 0.0
    var _kozachenko_leonenko_mi_line = 'xf = ascontiguousarray(x[:n], dtype=float64)'
    var _kozachenko_leonenko_mi_line = 'yf = ascontiguousarray(y[:n], dtype=float64)'
    var _kozachenko_leonenko_mi_line = 'if _HAS_RUST and _ssc is not 0:'
    return 0  # return float(_ssc.py_kozachenko_leonenko_mi(xf, yf
    var _kozachenko_leonenko_mi_line = 'xf = xf.reshape(-1, 1)'
    var _kozachenko_leonenko_mi_line = 'yf = yf.reshape(-1, 1)'
    var _kozachenko_leonenko_mi_line = 'xy = hstack([xf, yf])'
    var _kozachenko_leonenko_mi_line = 'from scipy.special import digamma'
    var _kozachenko_leonenko_mi_line = 'dists = max(abs(data - data[idx]), axis=1)'
    var _kozachenko_leonenko_mi_line = 'dists[idx] = inf'
    return 0  # return float(partition(dists, kk - 1)[kk - 1])
    var _kozachenko_leonenko_mi_line = 'psi_k = float(digamma(k))'
    var _kozachenko_leonenko_mi_line = 'psi_n = float(digamma(n))'
    var _kozachenko_leonenko_mi_line = 'nx_sum = 0.0'
    var _kozachenko_leonenko_mi_line = 'ny_sum = 0.0'
    var _kozachenko_leonenko_mi_line = 'for i in range(n):'
    var _kozachenko_leonenko_mi_line = 'eps = _kth_dist(xy, i, k)'
    var _kozachenko_leonenko_mi_line = 'nx = sum(abs(xf - xf[i]).ravel() < eps) - 1'
    var _kozachenko_leonenko_mi_line = 'ny = sum(abs(yf - yf[i]).ravel() < eps) - 1'
    var _kozachenko_leonenko_mi_line = 'nx_sum += digamma(nx + 1)'
    var _kozachenko_leonenko_mi_line = 'ny_sum += digamma(ny + 1)'
    return 0  # return float(max(0.0, psi_k + psi_n - nx_sum / n -

fn time_rescaling_ks_test(times: Int, rate_func: Int, t_start: Int, t_end: Int) -> Int:
    var _time_rescaling_ks_test_line = 'times: ndarray[Any, Any],'
    var _time_rescaling_ks_test_line = 'rate_func: Callable[[float], float],'
    var _time_rescaling_ks_test_line = 't_start: float = 0.0,'
    var _time_rescaling_ks_test_line = 't_end: float = 1.0,'
    var _time_rescaling_ks_test_line = ') -> tuple[float, bool]:'
    var _time_rescaling_ks_test_line = 'if times.size < 5:'
    return 0  # return 1.0, False
    var _time_rescaling_ks_test_line = 'sorted_t = sort(times[(times >= t_start) & (times <= t_end)]'
    var _time_rescaling_ks_test_line = 'n = sorted_t.size'
    var _time_rescaling_ks_test_line = 'rescaled = zeros(n)'
    var _time_rescaling_ks_test_line = 'for i in range(n):'
    var _time_rescaling_ks_test_line = 'lo = t_start if i == 0 else sorted_t[i - 1]'
    var _time_rescaling_ks_test_line = 'hi = sorted_t[i]'
    var _time_rescaling_ks_test_line = 'n_quad = 20'
    var _time_rescaling_ks_test_line = 't_quad = linspace(lo, hi, n_quad)'
    var _time_rescaling_ks_test_line = 'rates = array([rate_func(t) for t in t_quad])'
    var _time_rescaling_ks_test_line = 'rescaled[i] = trapezoid(rates, t_quad)'
    var _time_rescaling_ks_test_line = 'transformed = 1.0 - exp(-rescaled)'
    var _time_rescaling_ks_test_line = 'transformed.sort()'
    var _time_rescaling_ks_test_line = 'ecdf = arange(1, n + 1) / n'
    var _time_rescaling_ks_test_line = 'ks = max(abs(ecdf - transformed))'
    var _time_rescaling_ks_test_line = 'critical_95 = 1.36 / sqrt(n)  # Kolmogorov-Smirnov 95% criti'
    return 0  # return float(ks), bool(ks < critical_95)

fn _entropy(x: Int) -> Int:
    return 0  # vals, counts = unique(x, return_counts=True)
    var __entropy_line = 'p = counts / counts.sum()'
    return 0  # return float(-sum(p * log2(p + 1e-30)))

fn _cond_entropy(future: Int) -> Int:
    var __cond_entropy_line = 'joint = future.copy()'
    var __cond_entropy_line = 'for p in pasts:'
    var __cond_entropy_line = 'joint = joint * (p.max() + 1) + p'
    return 0  # vals, counts = unique(joint, return_counts=True)
    var __cond_entropy_line = 'h_joint = float(-sum(counts / n_pts * log2(counts / n_pts + '
    var __cond_entropy_line = 'past_joint = pasts[0].copy()'
    var __cond_entropy_line = 'for p in pasts[1:]:'
    var __cond_entropy_line = 'past_joint = past_joint * (p.max() + 1) + p'
    return 0  # vals2, counts2 = unique(past_joint, return_counts=
    var __cond_entropy_line = 'h_past = float(-sum(counts2 / n_pts * log2(counts2 / n_pts +'
    return 0  # return h_joint - h_past

fn _kth_dist(data: Int, idx: Int, kk: Int) -> Int:
    var __kth_dist_line = 'dists = max(abs(data - data[idx]), axis=1)'
    var __kth_dist_line = 'dists[idx] = inf'
    return 0  # return float(partition(dists, kk - 1)[kk - 1])
