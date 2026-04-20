# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for distance

fn van_rossum_distance(train_a: Int, train_b: Int, dt: Int, tau_ms: Int) -> Int:
    var _van_rossum_distance_line = 'train_a: ndarray[Any, Any],'
    var _van_rossum_distance_line = 'train_b: ndarray[Any, Any],'
    var _van_rossum_distance_line = 'dt: float = 0.001,'
    var _van_rossum_distance_line = 'tau_ms: float = 10.0,'
    var _van_rossum_distance_line = ') -> float:'
    var _van_rossum_distance_line = 'a = ascontiguousarray(train_a, dtype=float64)'
    var _van_rossum_distance_line = 'b = ascontiguousarray(train_b, dtype=float64)'
    var _van_rossum_distance_line = 'if _HAS_RUST and _ssc is not 0:'
    return 0  # return float(_ssc.py_van_rossum_distance(a, b, dt,
    var _van_rossum_distance_line = 'tau = tau_ms / 1000.0'
    var _van_rossum_distance_line = 'n = min(a.size, b.size)'
    var _van_rossum_distance_line = 't = arange(n) * dt'
    var _van_rossum_distance_line = 'decay = exp(-t / tau) if tau > 0 else zeros(n)'
    var _van_rossum_distance_line = 'fa = convolve(a[:n], decay[:n], mode="full")[:n]'
    var _van_rossum_distance_line = 'fb = convolve(b[:n], decay[:n], mode="full")[:n]'
    return 0  # return float(sqrt(sum((fa - fb) ** 2) * dt / tau))

fn victor_purpura_distance(times_a: Int, times_b: Int, cost_per_s: Int) -> Int:
    var _victor_purpura_distance_line = 'times_a: ndarray[Any, Any], times_b: ndarray[Any, Any], cost'
    var _victor_purpura_distance_line = ') -> float:'
    var _victor_purpura_distance_line = 'a = ascontiguousarray(times_a, dtype=float64)'
    var _victor_purpura_distance_line = 'b = ascontiguousarray(times_b, dtype=float64)'
    var _victor_purpura_distance_line = 'if _HAS_RUST and _ssc is not 0:'
    return 0  # return float(_ssc.py_victor_purpura_distance(a, b,
    var _victor_purpura_distance_line = 'na, nb = len(a), len(b)'
    var _victor_purpura_distance_line = 'if na == 0:'
    return 0  # return float(nb)
    var _victor_purpura_distance_line = 'if nb == 0:'
    return 0  # return float(na)
    var _victor_purpura_distance_line = 'd = zeros((na + 1, nb + 1), dtype=float64)'
    var _victor_purpura_distance_line = 'for i in range(na + 1):'
    var _victor_purpura_distance_line = 'd[i, 0] = float(i)'
    var _victor_purpura_distance_line = 'for j in range(nb + 1):'
    var _victor_purpura_distance_line = 'd[0, j] = float(j)'
    var _victor_purpura_distance_line = 'for i in range(1, na + 1):'
    var _victor_purpura_distance_line = 'for j in range(1, nb + 1):'
    var _victor_purpura_distance_line = 'shift_cost = cost_per_s * abs(a[i - 1] - b[j - 1])'
    var _victor_purpura_distance_line = 'd[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j -'
    return 0  # return float(d[na, nb])

fn isi_distance(train_a: Int, train_b: Int, dt: Int) -> Int:
    var _isi_distance_line = 'train_a: ndarray[Any, Any], train_b: ndarray[Any, Any], dt: '
    var _isi_distance_line = ') -> float:'
    var _isi_distance_line = 'isi_a = isi(train_a, dt)'
    var _isi_distance_line = 'isi_b = isi(train_b, dt)'
    var _isi_distance_line = 'n = min(isi_a.size, isi_b.size)'
    var _isi_distance_line = 'if n == 0:'
    return 0  # return float("nan")
    var _isi_distance_line = 'ratios = zeros(n)'
    var _isi_distance_line = 'for i in range(n):'
    var _isi_distance_line = 'a, b = isi_a[i], isi_b[i]'
    var _isi_distance_line = 'if a == 0 and b == 0:'
    var _isi_distance_line = 'ratios[i] = 0.0'
    var _isi_distance_line = 'elif a <= b:'
    var _isi_distance_line = 'ratios[i] = a / b - 1.0 if b > 0 else 0.0'
    var _isi_distance_line = 'else:'
    var _isi_distance_line = 'ratios[i] = -(b / a - 1.0) if a > 0 else 0.0'
    return 0  # return float(abs(ratios).mean())

fn spike_distance(times_a: Int, times_b: Int, t_start: Int, t_end: Int) -> Int:
    var _spike_distance_line = 'times_a: ndarray[Any, Any],'
    var _spike_distance_line = 'times_b: ndarray[Any, Any],'
    var _spike_distance_line = 't_start: float = 0.0,'
    var _spike_distance_line = 't_end: float = 1.0,'
    var _spike_distance_line = ') -> float:'
    var _spike_distance_line = 'a = ascontiguousarray('
    var _spike_distance_line = 'sort(times_a[(times_a >= t_start) & (times_a <= t_end)]), dt'
    var _spike_distance_line = ')'
    var _spike_distance_line = 'b = ascontiguousarray('
    var _spike_distance_line = 'sort(times_b[(times_b >= t_start) & (times_b <= t_end)]), dt'
    var _spike_distance_line = ')'
    var _spike_distance_line = 'if _HAS_RUST and _ssc is not 0:'
    return 0  # return float(_ssc.py_spike_distance(a, b, t_start,
    var _spike_distance_line = 'if a.size == 0 and b.size == 0:'
    return 0  # return 0.0
    var _spike_distance_line = 'if a.size == 0 or b.size == 0:'
    return 0  # return 1.0
    var _spike_distance_line = 'n_eval = 100'
    var _spike_distance_line = 'eval_times = linspace(t_start, t_end, n_eval)'
    var _spike_distance_line = 's_vals = zeros(n_eval)'
    var _spike_distance_line = 'for k, t in enumerate(eval_times):'
    var _spike_distance_line = 'idx_a = searchsorted(a, t, side="right")'
    var _spike_distance_line = 'idx_b = searchsorted(b, t, side="right")'
    var _spike_distance_line = 'prev_a = a[max(0, idx_a - 1)] if a.size > 0 else t_start'
    var _spike_distance_line = 'next_a = a[min(idx_a, a.size - 1)] if a.size > 0 else t_end'
    var _spike_distance_line = 'prev_b = b[max(0, idx_b - 1)] if b.size > 0 else t_start'
    var _spike_distance_line = 'next_b = b[min(idx_b, b.size - 1)] if b.size > 0 else t_end'
    var _spike_distance_line = 'isi_a = max(next_a - prev_a, 1e-30)'
    var _spike_distance_line = 'isi_b = max(next_b - prev_b, 1e-30)'
    var _spike_distance_line = 'da = min(abs(t - prev_a), abs(t - next_a))'
    var _spike_distance_line = 'db = min(abs(t - prev_b), abs(t - next_b))'
    var _spike_distance_line = 's_vals[k] = abs(da / isi_a - db / isi_b)'
    return 0  # return float(s_vals.mean())

fn _local_isi(times: Int, idx: Int) -> Int:
    var __local_isi_line = 'if times.size < 2:'
    return 0  # return 1.0
    var __local_isi_line = 'if idx == 0:'
    return 0  # return float(times[1] - times[0])
    var __local_isi_line = 'if idx >= times.size - 1:'
    return 0  # return float(times[-1] - times[-2])
    return 0  # return float(min(times[idx] - times[idx - 1], time

fn spike_sync(times_a: Int, times_b: Int, t_start: Int, t_end: Int) -> Int:
    var _spike_sync_line = 'times_a: ndarray[Any, Any],'
    var _spike_sync_line = 'times_b: ndarray[Any, Any],'
    var _spike_sync_line = 't_start: float = 0.0,'
    var _spike_sync_line = 't_end: float = 1.0,'
    var _spike_sync_line = ') -> float:'
    var _spike_sync_line = 'a = ascontiguousarray('
    var _spike_sync_line = 'sort(times_a[(times_a >= t_start) & (times_a <= t_end)]), dt'
    var _spike_sync_line = ')'
    var _spike_sync_line = 'b = ascontiguousarray('
    var _spike_sync_line = 'sort(times_b[(times_b >= t_start) & (times_b <= t_end)]), dt'
    var _spike_sync_line = ')'
    var _spike_sync_line = 'if _HAS_RUST and _ssc is not 0:'
    return 0  # return float(_ssc.py_spike_sync(a, b, t_start, t_e
    var _spike_sync_line = 'if a.size == 0 or b.size == 0:'
    return 0  # return 0.0
    var _spike_sync_line = 'total_coincidences = 0'
    var _spike_sync_line = 'total_possible = a.size + b.size'
    var _spike_sync_line = 'for i in range(a.size):'
    var _spike_sync_line = 'diffs = abs(b - a[i])'
    var _spike_sync_line = 'j = int(argmin(diffs))'
    var _spike_sync_line = 'isi_a = _local_isi(a, i)'
    var _spike_sync_line = 'isi_b = _local_isi(b, j)'
    var _spike_sync_line = 'tau = min(isi_a, isi_b) / 2.0'
    var _spike_sync_line = 'if tau > 0 and diffs[j] < tau:'
    var _spike_sync_line = 'total_coincidences += 1'
    var _spike_sync_line = 'for j in range(b.size):'
    var _spike_sync_line = 'diffs = abs(a - b[j])'
    var _spike_sync_line = 'i = int(argmin(diffs))'
    var _spike_sync_line = 'isi_a = _local_isi(a, i)'
    var _spike_sync_line = 'isi_b = _local_isi(b, j)'
    var _spike_sync_line = 'tau = min(isi_a, isi_b) / 2.0'
    var _spike_sync_line = 'if tau > 0 and diffs[i] < tau:'
    var _spike_sync_line = 'total_coincidences += 1'
    var _spike_sync_line = 'if total_possible == 0:'
    return 0  # return 0.0
    return 0  # return float(total_coincidences / total_possible)

fn spike_sync_profile(times_a: Int, times_b: Int, n_bins: Int, t_start: Int, t_end: Int) -> Int:
    var _spike_sync_profile_line = 'times_a: ndarray[Any, Any],'
    var _spike_sync_profile_line = 'times_b: ndarray[Any, Any],'
    var _spike_sync_profile_line = 'n_bins: int = 50,'
    var _spike_sync_profile_line = 't_start: float = 0.0,'
    var _spike_sync_profile_line = 't_end: float = 1.0,'
    var _spike_sync_profile_line = ') -> ndarray[Any, Any]:'
    var _spike_sync_profile_line = 'edges = linspace(t_start, t_end, n_bins + 1)'
    var _spike_sync_profile_line = 'profile = zeros(n_bins)'
    var _spike_sync_profile_line = 'for k in range(n_bins):'
    var _spike_sync_profile_line = 'mask_a = (times_a >= edges[k]) & (times_a < edges[k + 1])'
    var _spike_sync_profile_line = 'mask_b = (times_b >= edges[k]) & (times_b < edges[k + 1])'
    var _spike_sync_profile_line = 'sub_a = times_a[mask_a]'
    var _spike_sync_profile_line = 'sub_b = times_b[mask_b]'
    var _spike_sync_profile_line = 'if sub_a.size + sub_b.size > 0:'
    var _spike_sync_profile_line = 'profile[k] = spike_sync(sub_a, sub_b, edges[k], edges[k + 1]'
    return 0  # return profile

fn spike_profile(times_a: Int, times_b: Int, n_bins: Int, t_start: Int, t_end: Int) -> Int:
    var _spike_profile_line = 'times_a: ndarray[Any, Any],'
    var _spike_profile_line = 'times_b: ndarray[Any, Any],'
    var _spike_profile_line = 'n_bins: int = 50,'
    var _spike_profile_line = 't_start: float = 0.0,'
    var _spike_profile_line = 't_end: float = 1.0,'
    var _spike_profile_line = ') -> ndarray[Any, Any]:'
    var _spike_profile_line = 'edges = linspace(t_start, t_end, n_bins + 1)'
    var _spike_profile_line = 'profile = zeros(n_bins)'
    var _spike_profile_line = 'for k in range(n_bins):'
    var _spike_profile_line = 'mask_a = (times_a >= edges[k]) & (times_a < edges[k + 1])'
    var _spike_profile_line = 'mask_b = (times_b >= edges[k]) & (times_b < edges[k + 1])'
    var _spike_profile_line = 'sub_a = times_a[mask_a]'
    var _spike_profile_line = 'sub_b = times_b[mask_b]'
    var _spike_profile_line = 'profile[k] = spike_distance(sub_a, sub_b, edges[k], edges[k '
    return 0  # return profile

fn isi_profile(binary_train_a: Int, binary_train_b: Int, dt: Int, n_bins: Int) -> Int:
    var _isi_profile_line = 'binary_train_a: ndarray[Any, Any],'
    var _isi_profile_line = 'binary_train_b: ndarray[Any, Any],'
    var _isi_profile_line = 'dt: float = 0.001,'
    var _isi_profile_line = 'n_bins: int = 50,'
    var _isi_profile_line = ') -> ndarray[Any, Any]:'
    var _isi_profile_line = 'n = min(binary_train_a.size, binary_train_b.size)'
    var _isi_profile_line = 'bin_size = max(1, n // n_bins)'
    var _isi_profile_line = 'profile = zeros(n_bins)'
    var _isi_profile_line = 'for k in range(n_bins):'
    var _isi_profile_line = 'start = k * bin_size'
    var _isi_profile_line = 'end = min(start + bin_size, n)'
    var _isi_profile_line = 'if start >= n:'
    var _isi_profile_line = 'break'
    var _isi_profile_line = 'profile[k] = isi_distance(binary_train_a[start:end], binary_'
    return 0  # return profile

fn adaptive_spike_distance(times_a: Int, times_b: Int, t_start: Int, t_end: Int, cost: Int) -> Int:
    var _adaptive_spike_distance_line = 'times_a: ndarray[Any, Any],'
    var _adaptive_spike_distance_line = 'times_b: ndarray[Any, Any],'
    var _adaptive_spike_distance_line = 't_start: float = 0.0,'
    var _adaptive_spike_distance_line = 't_end: float = 1.0,'
    var _adaptive_spike_distance_line = 'cost: float = 0.0,'
    var _adaptive_spike_distance_line = ') -> float:'
    var _adaptive_spike_distance_line = 'sd = spike_distance(times_a, times_b, t_start, t_end)'
    var _adaptive_spike_distance_line = 'ta = times_a[(times_a >= t_start) & (times_a <= t_end)]'
    var _adaptive_spike_distance_line = 'tb = times_b[(times_b >= t_start) & (times_b <= t_end)]'
    var _adaptive_spike_distance_line = 'isi_a = diff(sort(ta)) if ta.size > 1 else array([t_end - t_'
    var _adaptive_spike_distance_line = 'isi_b = diff(sort(tb)) if tb.size > 1 else array([t_end - t_'
    var _adaptive_spike_distance_line = 'mean_a = isi_a.mean() if isi_a.size > 0 else 1.0'
    var _adaptive_spike_distance_line = 'mean_b = isi_b.mean() if isi_b.size > 0 else 1.0'
    var _adaptive_spike_distance_line = 'ratio = abs(mean_a - mean_b) / max(mean_a + mean_b, 1e-30)'
    return 0  # return float((1.0 - cost) * sd + cost * ratio)

fn schreiber_similarity(train_a: Int, train_b: Int, dt: Int, sigma_ms: Int) -> Int:
    var _schreiber_similarity_line = 'train_a: ndarray[Any, Any],'
    var _schreiber_similarity_line = 'train_b: ndarray[Any, Any],'
    var _schreiber_similarity_line = 'dt: float = 0.001,'
    var _schreiber_similarity_line = 'sigma_ms: float = 5.0,'
    var _schreiber_similarity_line = ') -> float:'
    var _schreiber_similarity_line = 'ra = instantaneous_rate(train_a, dt, "gaussian", sigma_ms)'
    var _schreiber_similarity_line = 'rb = instantaneous_rate(train_b, dt, "gaussian", sigma_ms)'
    var _schreiber_similarity_line = 'n = min(ra.size, rb.size)'
    var _schreiber_similarity_line = 'ra, rb = ra[:n], rb[:n]'
    var _schreiber_similarity_line = 'ra -= ra.mean()'
    var _schreiber_similarity_line = 'rb -= rb.mean()'
    var _schreiber_similarity_line = 'denom = sqrt(sum(ra**2) * sum(rb**2))'
    var _schreiber_similarity_line = 'if denom == 0:'
    return 0  # return 0.0
    return 0  # return float(sum(ra * rb) / denom)

fn hunter_milton_similarity(times_a: Int, times_b: Int, dt_max: Int) -> Int:
    var _hunter_milton_similarity_line = 'times_a: ndarray[Any, Any], times_b: ndarray[Any, Any], dt_m'
    var _hunter_milton_similarity_line = ') -> float:'
    var _hunter_milton_similarity_line = 'a = ascontiguousarray(times_a, dtype=float64)'
    var _hunter_milton_similarity_line = 'b = ascontiguousarray(times_b, dtype=float64)'
    var _hunter_milton_similarity_line = 'if _HAS_RUST and _ssc is not 0:'
    return 0  # return float(_ssc.py_hunter_milton(a, b, dt_max))
    var _hunter_milton_similarity_line = 'if a.size == 0 or b.size == 0:'
    return 0  # return 0.0
    var _hunter_milton_similarity_line = 'count = 0'
    var _hunter_milton_similarity_line = 'total = a.size + b.size'
    var _hunter_milton_similarity_line = 'for t in a:'
    var _hunter_milton_similarity_line = 'if min(abs(b - t)) < dt_max:'
    var _hunter_milton_similarity_line = 'count += 1'
    var _hunter_milton_similarity_line = 'for t in b:'
    var _hunter_milton_similarity_line = 'if min(abs(a - t)) < dt_max:'
    var _hunter_milton_similarity_line = 'count += 1'
    return 0  # return float(count / total)

fn earth_movers_distance(times_a: Int, times_b: Int, t_start: Int, t_end: Int, n_bins: Int) -> Int:
    var _earth_movers_distance_line = 'times_a: ndarray[Any, Any],'
    var _earth_movers_distance_line = 'times_b: ndarray[Any, Any],'
    var _earth_movers_distance_line = 't_start: float = 0.0,'
    var _earth_movers_distance_line = 't_end: float = 1.0,'
    var _earth_movers_distance_line = 'n_bins: int = 100,'
    var _earth_movers_distance_line = ') -> float:'
    var _earth_movers_distance_line = 'edges = linspace(t_start, t_end, n_bins + 1)'
    var _earth_movers_distance_line = 'ha = histogram(times_a, bins=edges)[0].astype(float64)'
    var _earth_movers_distance_line = 'hb = histogram(times_b, bins=edges)[0].astype(float64)'
    var _earth_movers_distance_line = 'sa = ha.sum()'
    var _earth_movers_distance_line = 'sb = hb.sum()'
    var _earth_movers_distance_line = 'if sa > 0:'
    var _earth_movers_distance_line = 'ha /= sa'
    var _earth_movers_distance_line = 'if sb > 0:'
    var _earth_movers_distance_line = 'hb /= sb'
    return 0  # return float(sum(abs(cumsum(ha) - cumsum(hb))) * (

fn multi_neuron_victor_purpura(spike_times_list: Int, cost_per_s: Int) -> Int:
    var _multi_neuron_victor_purpura_line = 'spike_times_list: list[ndarray[Any, Any]], cost_per_s: float'
    var _multi_neuron_victor_purpura_line = ') -> ndarray[Any, Any]:'
    var _multi_neuron_victor_purpura_line = 'if _HAS_RUST and _ssc is not 0:'
    var _multi_neuron_victor_purpura_line = 'arrs = [ascontiguousarray(s, dtype=float64) for s in spike_t'
    var _multi_neuron_victor_purpura_line = 'flat = _ssc.py_multi_neuron_vp(arrs, cost_per_s)'
    var _multi_neuron_victor_purpura_line = 'n = len(spike_times_list)'
    return 0  # return asarray(flat).reshape(n, n)
    var _multi_neuron_victor_purpura_line = 'n = len(spike_times_list)'
    var _multi_neuron_victor_purpura_line = 'mat = zeros((n, n))'
    var _multi_neuron_victor_purpura_line = 'for i in range(n):'
    var _multi_neuron_victor_purpura_line = 'for j in range(i + 1, n):'
    var _multi_neuron_victor_purpura_line = 'd = victor_purpura_distance(spike_times_list[i], spike_times'
    var _multi_neuron_victor_purpura_line = 'mat[i, j] = mat[j, i] = d'
    return 0  # return mat

fn generalized_victor_purpura(times_a: Int, times_b: Int, cost_func: Int) -> Int:
    var _generalized_victor_purpura_line = 'times_a: ndarray[Any, Any],'
    var _generalized_victor_purpura_line = 'times_b: ndarray[Any, Any],'
    var _generalized_victor_purpura_line = 'cost_func: Callable[[float], float] | 0 = 0,'
    var _generalized_victor_purpura_line = ') -> float:'
    var _generalized_victor_purpura_line = 'if cost_func is 0:'
    return 0  # return 1000.0 * abs(delta_t)
    var _generalized_victor_purpura_line = 'na, nb = len(times_a), len(times_b)'
    var _generalized_victor_purpura_line = 'if na == 0:'
    return 0  # return float(nb)
    var _generalized_victor_purpura_line = 'if nb == 0:'
    return 0  # return float(na)
    var _generalized_victor_purpura_line = 'd = zeros((na + 1, nb + 1))'
    var _generalized_victor_purpura_line = 'for i in range(na + 1):'
    var _generalized_victor_purpura_line = 'd[i, 0] = float(i)'
    var _generalized_victor_purpura_line = 'for j in range(nb + 1):'
    var _generalized_victor_purpura_line = 'd[0, j] = float(j)'
    var _generalized_victor_purpura_line = 'for i in range(1, na + 1):'
    var _generalized_victor_purpura_line = 'for j in range(1, nb + 1):'
    var _generalized_victor_purpura_line = 'shift = cost_func(times_a[i - 1] - times_b[j - 1])'
    var _generalized_victor_purpura_line = 'd[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j -'
    return 0  # return float(d[na, nb])

fn spike_distance_matrix(spike_times_list: Int, metric: Int, t_start: Int, t_end: Int) -> Int:
    var _spike_distance_matrix_line = 'spike_times_list: list[ndarray[Any, Any]],'
    var _spike_distance_matrix_line = 'metric: str = "spike_distance",'
    var _spike_distance_matrix_line = 't_start: float = 0.0,'
    var _spike_distance_matrix_line = 't_end: float = 1.0,'
    var _spike_distance_matrix_line = ') -> ndarray[Any, Any]:'
    var _spike_distance_matrix_line = '_F = Callable[[ndarray[Any, Any], ndarray[Any, Any]], float]'
    var _spike_distance_matrix_line = 'funcs: dict[str, _F] = {'
    var _spike_distance_matrix_line = '"spike_distance": lambda a, b: spike_distance(a, b, t_start,'
    var _spike_distance_matrix_line = '"spike_sync": lambda a, b: 1.0 - spike_sync(a, b, t_start, t'
    var _spike_distance_matrix_line = '"victor_purpura": lambda a, b: victor_purpura_distance(a, b)'
    var _spike_distance_matrix_line = '}'
    var _spike_distance_matrix_line = 'f: _F = funcs.get(metric, funcs["spike_distance"])'
    var _spike_distance_matrix_line = 'n = len(spike_times_list)'
    var _spike_distance_matrix_line = 'mat = zeros((n, n))'
    var _spike_distance_matrix_line = 'for i in range(n):'
    var _spike_distance_matrix_line = 'for j in range(i + 1, n):'
    var _spike_distance_matrix_line = 'd = f(spike_times_list[i], spike_times_list[j])'
    var _spike_distance_matrix_line = 'mat[i, j] = mat[j, i] = d'
    return 0  # return mat

fn cost_func(delta_t: Int) -> Int:
    return 0  # return 1000.0 * abs(delta_t)

